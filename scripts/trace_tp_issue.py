#!/usr/bin/env python3
"""Trace tensor shapes through the forward pass to find the issue."""

import torch
import torch.distributed as dist
import os
import json

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.local_activation_store import LocalActivationStore
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

# Initialize distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# Monkey patch the decoder to add debugging
def debug_decode(self, a, layer_idx):
    """Wrapper to debug what the decoder receives."""
    print(f"\n[DEBUG] Rank {rank} Decoder.decode called for layer {layer_idx}")
    for src_layer, act_tensor in a.items():
        print(f"  Rank {rank}: Received activation from layer {src_layer} with shape {act_tensor.shape}")

    # Call the original decode - it's stored as an attribute on the function
    return debug_decode.original(self, a, layer_idx)


# Path to checkpoint
checkpoint_dir = "clt_training_logs/gpt2_batchtopk/step_90000"
config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"

# Load config
with open(config_path, "r") as f:
    config_dict = json.load(f)
config = CLTConfig(**config_dict)

# Create model
model = CrossLayerTranscoder(config, process_group=dist.group.WORLD, device=device)
model.eval()

# Patch the decoder
debug_decode.original = model.decoder_module.decode
model.decoder_module.decode = lambda a, layer_idx: debug_decode(model.decoder_module, a, layer_idx)

# Load checkpoint
state_dict = model.state_dict()
load_state_dict(
    state_dict=state_dict,
    storage_reader=FileSystemReader(checkpoint_dir),
    planner=DefaultLoadPlanner(),
    no_dist=False,
)
model.load_state_dict(state_dict)

if rank == 0:
    print("Model loaded, testing with real data...")

# Get real data
store = LocalActivationStore(
    dataset_path=activation_path,
    train_batch_size_tokens=10,  # Small batch for debugging
    device=device,
    dtype="float16",
    rank=0,  # All ranks see same data
    world=1,
    seed=42,
    sampling_strategy="sequential",
    normalization_method="auto",
    shard_data=False,
)

# Get one batch
inputs, targets = next(iter(store))

if rank == 0:
    print(f"\nInput shapes: {[(k, v.shape) for k, v in inputs.items()][:3]}...")

# Trace through the forward pass
with torch.no_grad():
    # Step 1: Get feature activations
    print(f"\n[TRACE] Rank {rank}: Calling get_feature_activations...")
    activations = model.get_feature_activations(inputs)

    for layer_idx in [0, 1]:  # Just check first two layers
        if layer_idx in activations:
            print(f"  Rank {rank}: Feature activations for layer {layer_idx} shape: {activations[layer_idx].shape}")

    # Step 2: The forward method calls decode with these activations
    print(f"\n[TRACE] Rank {rank}: Calling forward (which calls decode)...")

    # Let's manually do what forward does to see the issue
    reconstructions = {}
    for layer_idx in range(min(2, config.num_layers)):  # Just first 2 layers for debugging
        relevant_activations = {k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0}

        print(
            f"\n[TRACE] Rank {rank}: For layer {layer_idx}, passing activations from layers: {list(relevant_activations.keys())}"
        )

        if layer_idx in inputs and relevant_activations:
            # This is where decode gets called
            reconstructions[layer_idx] = model.decode(relevant_activations, layer_idx)

store.close()
dist.barrier()
dist.destroy_process_group()
