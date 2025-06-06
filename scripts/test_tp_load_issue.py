#!/usr/bin/env python3
"""Test to identify the issue with loaded tensor parallel models."""

import torch
import torch.distributed as dist
import os
import json

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
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

# Path to your checkpoint
checkpoint_dir = "clt_training_logs/gpt2_batchtopk/step_90000"
config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"

if rank == 0:
    print(f"Testing with world_size={world_size}")
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_dir}")

# Load config
with open(config_path, "r") as f:
    config_dict = json.load(f)
config = CLTConfig(**config_dict)

# Create dummy input
dummy_input = {0: torch.randn(10, config.d_model, device=device)}

# Test 1: Fresh model
if rank == 0:
    print("\n=== Test 1: Fresh model ===")
fresh_model = CrossLayerTranscoder(config, process_group=dist.group.WORLD, device=device)
fresh_model.eval()

with torch.no_grad():
    fresh_preact = fresh_model.encoder_module.get_preactivations(dummy_input[0], 0)
    fresh_acts = fresh_model.get_feature_activations(dummy_input)

    # Check internal state
    if rank == 0:
        print(f"Fresh model encoder world_size: {fresh_model.encoder_module.world_size}")
        print(f"Fresh model preactivation shape: {fresh_preact.shape}")
        print(f"Fresh model activation shape: {fresh_acts[0].shape}")

    # Test what shape the decoder sees
    print(f"Rank {rank}: Fresh model - shape passed to decoder: {fresh_acts[0].shape}")

dist.barrier()

# Test 2: Loaded model
if rank == 0:
    print("\n=== Test 2: Loaded model ===")
loaded_model = CrossLayerTranscoder(config, process_group=dist.group.WORLD, device=device)
loaded_model.eval()

# Load the checkpoint
state_dict = loaded_model.state_dict()
load_state_dict(
    state_dict=state_dict,
    storage_reader=FileSystemReader(checkpoint_dir),
    planner=DefaultLoadPlanner(),
    no_dist=False,
)
loaded_model.load_state_dict(state_dict)

with torch.no_grad():
    loaded_preact = loaded_model.encoder_module.get_preactivations(dummy_input[0], 0)
    loaded_acts = loaded_model.get_feature_activations(dummy_input)

    # Check internal state
    if rank == 0:
        print(f"Loaded model encoder world_size: {loaded_model.encoder_module.world_size}")
        print(f"Loaded model preactivation shape: {loaded_preact.shape}")
        print(f"Loaded model activation shape: {loaded_acts[0].shape}")

    # Test what shape the decoder sees
    print(f"Rank {rank}: Loaded model - shape passed to decoder: {loaded_acts[0].shape}")

    # Let's also check the actual encoder weights to see if they're loaded correctly
    if rank == 0:
        encoder0_weight = loaded_model.encoder_module.encoders[0].weight
        print(f"\nLoaded encoder[0] weight shape: {encoder0_weight.shape}")
        print(f"Expected shape (sharded): [{config.num_features // world_size}, {config.d_model}]")

dist.barrier()

# Test 3: Try calling forward to see where the issue occurs
if rank == 0:
    print("\n=== Test 3: Forward pass comparison ===")

with torch.no_grad():
    try:
        fresh_output = fresh_model(dummy_input)
        if rank == 0:
            print(f"Fresh model forward pass successful, output shape: {fresh_output[0].shape}")
    except Exception as e:
        print(f"Rank {rank}: Fresh model forward failed: {e}")

    try:
        loaded_output = loaded_model(dummy_input)
        if rank == 0:
            print(f"Loaded model forward pass successful, output shape: {loaded_output[0].shape}")
    except Exception as e:
        print(f"Rank {rank}: Loaded model forward failed: {e}")

dist.barrier()
dist.destroy_process_group()
