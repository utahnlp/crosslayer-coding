#!/usr/bin/env python3
"""
Debug script to understand what the DefaultSavePlanner is doing.
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.planner import SavePlan
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    print(f"\nRank {rank}: Debugging checkpoint planner")
    
    # Create a simple model
    config = CLTConfig(
        num_features=8192,
        num_layers=12,
        d_model=768,
        activation_fn="batchtopk",
        batchtopk_k=200,
        model_name="gpt2",
        clt_dtype="float32",
    )
    
    model = CrossLayerTranscoder(
        config,
        process_group=dist.group.WORLD,
        device=device
    )
    
    # Initialize with different values per rank
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "encoder" in name and "0.weight" in name:
                # Set to rank-specific values
                param.fill_(float(rank + 1))
                print(f"Rank {rank}: Set {name} to {float(rank + 1)}")
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Check what's in the state dict
    print(f"\nRank {rank}: State dict keys (first 5):")
    for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):
        if hasattr(tensor, 'shape'):
            checksum = torch.sum(torch.abs(tensor)).item()
            print(f"  {key}: shape={tensor.shape}, checksum={checksum:.2f}")
    
    # Create planner and see what it plans
    planner = DefaultSavePlanner()
    
    # The planner needs metadata about the state dict
    # This is normally done internally by save_state_dict
    # Let's try to understand what the plan would be
    
    print(f"\nRank {rank}: Creating save plan...")
    
    # Try to create a plan (this is simplified - the real save_state_dict does more)
    # We can't easily call the planner directly, but we can at least check
    # if all ranks have the same state dict structure
    
    enc_key = "encoder_module.encoders.0.weight"
    if enc_key in state_dict:
        tensor = state_dict[enc_key]
        print(f"\nRank {rank}: {enc_key}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Sum: {torch.sum(tensor).item()}")
        print(f"  First 5 values: {tensor.flatten()[:5].tolist()}")
    
    dist.barrier()
    
    # Now actually save the checkpoint
    
    checkpoint_dir = "./debug_planner_checkpoint"
    
    print(f"\nRank {rank}: Saving checkpoint to {checkpoint_dir}")
    
    try:
        save_state_dict(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(checkpoint_dir),
            planner=DefaultSavePlanner(),
            no_dist=False,
        )
        print(f"Rank {rank}: Save completed")
    except Exception as e:
        print(f"Rank {rank}: Save failed: {e}")
    
    dist.barrier()
    
    # Check what files were created
    if rank == 0:
        import time
        time.sleep(1)  # Give filesystem time to sync
        
        print(f"\n{'='*60}")
        print("Checkpoint files created:")
        print(f"{'='*60}")
        
        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.exists():
            for f in sorted(ckpt_path.iterdir()):
                size = os.path.getsize(f) if f.is_file() else 0
                print(f"  {f.name}: {size:,} bytes")
    
    dist.barrier()
    
    # Now try to load and check
    
    print(f"\nRank {rank}: Loading checkpoint back...")
    
    # Create new model
    model2 = CrossLayerTranscoder(
        config,
        process_group=dist.group.WORLD,
        device=device
    )
    
    loaded_state = model2.state_dict()
    load_state_dict(
        state_dict=loaded_state,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    model2.load_state_dict(loaded_state)
    
    # Check what was loaded
    if enc_key in loaded_state:
        tensor = loaded_state[enc_key]
        print(f"\nRank {rank}: Loaded {enc_key}")
        print(f"  Sum: {torch.sum(tensor).item()}")
        print(f"  First 5 values: {tensor.flatten()[:5].tolist()}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Summary: Each rank should have different values if working correctly")
        print(f"{'='*60}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()