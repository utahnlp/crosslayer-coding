#!/usr/bin/env python3
"""Test if encoder gather operations work correctly in tensor parallel mode."""

import torch
import torch.distributed as dist
import os

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder

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

# Create a simple config
config = CLTConfig(
    num_features=32768,
    num_layers=12,
    d_model=768,
    activation_fn="batchtopk",
    batchtopk_k=200,
    clt_dtype="float32",
)

# Create model with TP
model = CrossLayerTranscoder(config, process_group=dist.group.WORLD, device=device)
model.eval()

# Create dummy input
dummy_input = {0: torch.randn(10, 768, device=device)}  # 10 tokens, 768 dims

if rank == 0:
    print(f"Testing encoder with world_size={world_size}")
    print(f"Config: num_features={config.num_features}, d_model={config.d_model}")

# Test encoder directly
with torch.no_grad():
    # Get preactivations from encoder
    preact = model.encoder_module.get_preactivations(dummy_input[0], 0)
    if rank == 0:
        print(f"\nPreactivation shape: {preact.shape}")
        print(f"Expected: [10, {config.num_features}]")

    # Get feature activations (includes BatchTopK)
    feat_acts = model.get_feature_activations(dummy_input)
    if rank == 0:
        print(f"\nFeature activation shape for layer 0: {feat_acts[0].shape}")
        print(f"Expected: [10, {config.num_features}]")

    # Test the forward pass
    outputs = model(dummy_input)
    if rank == 0:
        print(f"\nOutput shape for layer 0: {outputs[0].shape}")
        print(f"Expected: [10, {config.d_model}]")

    # Check if activations are being passed correctly to decoder
    # The decoder expects full tensors, so let's see what it's receiving
    print(f"\nRank {rank}: Activation shape being passed to decoder: {feat_acts[0].shape}")

dist.barrier()
dist.destroy_process_group()
