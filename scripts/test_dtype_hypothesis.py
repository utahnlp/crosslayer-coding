#!/usr/bin/env python3
"""Test if dtype mismatch is causing the issue."""

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

# Paths
checkpoint_dir = "clt_training_logs/gpt2_batchtopk/step_90000"
config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"

# Load config
with open(config_path, "r") as f:
    config_dict = json.load(f)

if rank == 0:
    print("=== Testing dtype hypothesis ===")
    print(f"Original config clt_dtype: {config_dict.get('clt_dtype', 'None/default')}")

# Test different batch sizes
batch_sizes = [10, 512, 1024]

for batch_size in batch_sizes:
    if rank == 0:
        print(f"\n--- Testing batch size {batch_size} ---")

    # Test 1: Model with float32 (default)
    config1 = CLTConfig(**config_dict)
    config1.clt_dtype = "float32"  # Explicitly set

    model1 = CrossLayerTranscoder(config1, process_group=dist.group.WORLD, device=device)
    model1.eval()

    # Load checkpoint
    state_dict1 = model1.state_dict()
    load_state_dict(
        state_dict=state_dict1,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    model1.load_state_dict(state_dict1)

    if rank == 0:
        print(f"Model 1 (float32) dtype: {next(model1.parameters()).dtype}")

    # Test 2: Model with float16
    config2 = CLTConfig(**config_dict)
    config2.clt_dtype = "float16"  # Match training

    model2 = CrossLayerTranscoder(config2, process_group=dist.group.WORLD, device=device)
    model2.eval()

    # Load checkpoint
    state_dict2 = model2.state_dict()
    load_state_dict(
        state_dict=state_dict2,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    model2.load_state_dict(state_dict2)

    if rank == 0:
        print(f"Model 2 (float16) dtype: {next(model2.parameters()).dtype}")

    # Get data
    store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=batch_size,
        device=device,
        dtype="float16",
        rank=0,
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=False,
    )

    inputs, targets = next(iter(store))

    # Test both models
    with torch.no_grad():
        # Model 1 (float32)
        try:
            acts1 = model1.get_feature_activations(inputs)
            out1 = model1(inputs)
            if rank == 0:
                print(
                    f"  Model 1 (float32): Success! Activation shape: {acts1[0].shape}, Output shape: {out1[0].shape}"
                )
        except Exception as e:
            if rank == 0:
                print(f"  Model 1 (float32): Failed with error: {str(e)[:100]}...")

        # Model 2 (float16)
        try:
            acts2 = model2.get_feature_activations(inputs)
            out2 = model2(inputs)
            if rank == 0:
                print(
                    f"  Model 2 (float16): Success! Activation shape: {acts2[0].shape}, Output shape: {out2[0].shape}"
                )
        except Exception as e:
            if rank == 0:
                print(f"  Model 2 (float16): Failed with error: {str(e)[:100]}...")

    store.close()

    # Clean up models
    del model1, model2
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    dist.barrier()

dist.destroy_process_group()
