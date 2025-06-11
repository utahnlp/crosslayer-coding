#!/usr/bin/env python3
"""
Script B: Load model from .distcp files and capture weights.
Saves weight summaries to a JSON file for comparison.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
from typing import Dict, Any

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


def get_weight_summary(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, Any]:
    """Get summary statistics for key weights."""
    summary = {}
    
    # Sample a few key parameters
    key_params = [
        ("encoder_0", model.encoder_module.encoders[0].weight if len(model.encoder_module.encoders) > 0 else None),
        ("decoder_0->0", model.decoder_module.decoders["0->0"].weight if "0->0" in model.decoder_module.decoders else None),
        ("decoder_0->1", model.decoder_module.decoders["0->1"].weight if "0->1" in model.decoder_module.decoders else None),
    ]
    
    for name, param in key_params:
        if param is None:
            continue
        
        data = param.data.cpu().float().numpy()
        
        # Get a 5x5 sample and statistics
        sample = data[:5, :5] if data.ndim > 1 else data[:5]
        
        summary[f"{prefix}{name}"] = {
            "shape": list(param.shape),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "sample_5x5": sample.tolist(),
            "checksum": float(np.sum(np.abs(data)))  # Simple checksum
        }
    
    return summary


def main():
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Use LOCAL_RANK for device assignment to avoid duplicate GPU error
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    config_path = output_dir / "cfg.json"
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("STAGE B: Loading model from .distcp files")
        print(f"{'='*60}")
        print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Load config
    with open(config_path, "r") as f:
        loaded_config_dict = json.load(f)
    loaded_config = CLTConfig(**loaded_config_dict)
    
    # Create model with same distributed setup
    loaded_model_B = CrossLayerTranscoder(
        loaded_config,
        process_group=dist.group.WORLD if world_size > 1 else None,
        device=device
    )
    loaded_model_B.eval()
    
    if rank == 0:
        print(f"Created model with num_features={loaded_config.num_features}, world_size={world_size}")
    
    # Load distributed checkpoint
    state_dict_B = loaded_model_B.state_dict()
    
    print(f"Rank {rank}: Loading distributed checkpoint...")
    print(f"Rank {rank}: Model device: {device}")
    print(f"Rank {rank}: Process group size: {dist.get_world_size()}")
    
    # Debug: Check what files exist
    distcp_files = list(checkpoint_dir.glob("*.distcp"))
    print(f"Rank {rank}: Found {len(distcp_files)} .distcp files: {[f.name for f in distcp_files]}")
    
    # Debug: Check encoder weight before loading
    enc_key = "encoder_module.encoders.0.weight"
    if enc_key in state_dict_B:
        import numpy as np
        before_sum = float(torch.sum(torch.abs(state_dict_B[enc_key])).item())
        print(f"Rank {rank}: Before loading - {enc_key} checksum: {before_sum:.2f}")
    
    load_state_dict(
        state_dict=state_dict_B,
        storage_reader=FileSystemReader(str(checkpoint_dir)),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    loaded_model_B.load_state_dict(state_dict_B)
    
    # Debug: Check encoder weight after loading
    if enc_key in state_dict_B:
        after_sum = float(torch.sum(torch.abs(state_dict_B[enc_key])).item())
        print(f"Rank {rank}: After loading - {enc_key} checksum: {after_sum:.2f}")
    
    # Get weights from loaded model
    summary_B = get_weight_summary(loaded_model_B, "B_")
    
    # Always print for both ranks to see what each loads
    print(f"\nRank {rank} loaded model weight summary from .distcp files:")
    for key, val in summary_B.items():
        print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
    
    # Save summaries to files for each rank
    summary_file = output_dir / f"weight_summary_B_rank{rank}.json"
    with open(summary_file, "w") as f:
        json.dump(summary_B, f, indent=2)
    
    if rank == 0:
        print(f"\nSaved weight summary to {summary_file}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Stage B completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()