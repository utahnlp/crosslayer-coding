#!/usr/bin/env python3
"""
Full comparison script that checks ALL weights, not just the first few.
This will help us understand if the distcp files are truly correct.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np

# Imports for distributed checkpoint loading
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def get_full_weight_summary(model: CrossLayerTranscoder, prefix: str = "") -> dict:
    """Get summary of ALL weights in the model."""
    summary = {}
    state_dict = model.state_dict()
    
    for key, tensor in state_dict.items():
        if 'weight' in key:
            data = tensor.data.cpu().float().numpy()
            summary[f"{prefix}{key}"] = {
                "shape": list(tensor.shape),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "checksum": float(np.sum(np.abs(data)))
            }
    
    return summary


def main():
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Use LOCAL_RANK for device assignment
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        print("This script must be run with torchrun")
        return
    
    # Paths
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    config_path = output_dir / "cfg.json"
    
    # Load config
    with open(config_path, "r") as f:
        loaded_config_dict = json.load(f)
    loaded_config = CLTConfig(**loaded_config_dict)
    
    # Create model
    model = CrossLayerTranscoder(
        loaded_config,
        process_group=dist.group.WORLD,
        device=device
    )
    model.eval()
    
    # Load distributed checkpoint
    state_dict = model.state_dict()
    load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(str(checkpoint_dir)),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    model.load_state_dict(state_dict)
    
    # Get full summary
    summary = get_full_weight_summary(model, f"rank{rank}_")
    
    print(f"\n{'='*60}")
    print(f"Rank {rank} - Full weight summary from .distcp files:")
    print(f"{'='*60}")
    
    # Group by layer type
    encoders = {k: v for k, v in summary.items() if 'encoder' in k}
    decoders = {k: v for k, v in summary.items() if 'decoder' in k}
    
    print(f"\nEncoders ({len(encoders)} weights):")
    for key in sorted(encoders.keys()):
        val = encoders[key]
        print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.2f}")
    
    print(f"\nDecoders ({len(decoders)} weights):")
    for key in sorted(decoders.keys())[:10]:  # First 10
        val = decoders[key]
        print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.2f}")
    
    if len(decoders) > 10:
        print(f"  ... and {len(decoders) - 10} more decoder weights")
    
    # Save full summary
    summary_file = output_dir / f"weight_summary_full_rank{rank}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    dist.barrier()
    
    # On rank 0, compare the two ranks
    if rank == 0:
        import time
        time.sleep(1)  # Ensure rank 1's file is written
        
        rank1_file = output_dir / "weight_summary_full_rank1.json"
        if rank1_file.exists():
            with open(rank1_file, "r") as f:
                rank1_summary = json.load(f)
            
            print(f"\n{'='*60}")
            print("Comparing rank 0 vs rank 1 weights:")
            print(f"{'='*60}")
            
            # Find matching keys
            rank0_keys = set(k.replace('rank0_', '') for k in summary.keys())
            rank1_keys = set(k.replace('rank1_', '') for k in rank1_summary.keys())
            
            common_keys = rank0_keys & rank1_keys
            
            different_count = 0
            same_count = 0
            
            for key in sorted(common_keys):
                rank0_val = summary[f'rank0_{key}']
                rank1_val = rank1_summary[f'rank1_{key}']
                
                if abs(rank0_val['checksum'] - rank1_val['checksum']) < 0.01:
                    same_count += 1
                else:
                    different_count += 1
                    if different_count <= 5:  # Show first 5 differences
                        print(f"\n{key}:")
                        print(f"  Rank 0: checksum={rank0_val['checksum']:.2f}")
                        print(f"  Rank 1: checksum={rank1_val['checksum']:.2f}")
            
            print(f"\nSummary:")
            print(f"  Same weights: {same_count}")
            print(f"  Different weights: {different_count}")
            print(f"  Total weights: {len(common_keys)}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()