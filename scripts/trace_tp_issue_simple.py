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

    # Step 2: Check what happens when we manually pass these to decode
    print(f"\n[TRACE] Rank {rank}: Manually checking decode inputs...")

    # Test decode for layer 0
    layer_idx = 0
    relevant_activations = {k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0}

    print(f"\n[TRACE] Rank {rank}: About to decode layer {layer_idx}")
    print(f"  Activations being passed: {[(k, v.shape) for k, v in relevant_activations.items()]}")

    # Check if the issue is in how we access the decoder
    decoder_module = model.decoder_module
    print(f"  Decoder module type: {type(decoder_module)}")
    print(f"  Decoder expected features: {decoder_module.config.num_features}")

    # Let's check the RowParallelLinear's expected input features
    decoder_key = "0->0"
    if hasattr(decoder_module.decoders, decoder_key):
        specific_decoder = decoder_module.decoders[decoder_key]
        print(f"  Decoder 0->0 full_in_features: {specific_decoder.full_in_features}")
        print(f"  Decoder 0->0 local_in_features: {specific_decoder.local_in_features}")
        print(f"  Decoder 0->0 input_is_parallel: {specific_decoder.input_is_parallel}")

store.close()
dist.barrier()
dist.destroy_process_group()
