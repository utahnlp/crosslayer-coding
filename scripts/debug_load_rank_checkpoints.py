#!/usr/bin/env python3
"""
Load and compare individual rank checkpoint files.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    checkpoint_dir = Path("./debug_weight_check/latest")
    
    print(f"\n{'='*60}")
    print("Loading individual rank checkpoints")
    print(f"{'='*60}")
    
    # Load rank 0 and rank 1 checkpoints
    rank0_path = checkpoint_dir / "rank_0_model.pt"
    rank1_path = checkpoint_dir / "rank_1_model.pt"
    
    if not rank0_path.exists() or not rank1_path.exists():
        print("ERROR: Rank checkpoint files not found!")
        print(f"Looking for: {rank0_path} and {rank1_path}")
        return
    
    print(f"\nLoading {rank0_path}")
    rank0_state = torch.load(rank0_path, map_location="cpu")
    
    print(f"Loading {rank1_path}")
    rank1_state = torch.load(rank1_path, map_location="cpu")
    
    # Compare key weights
    enc_key = "encoder_module.encoders.0.weight"
    
    if enc_key in rank0_state and enc_key in rank1_state:
        enc0 = rank0_state[enc_key]
        enc1 = rank1_state[enc_key]
        
        print(f"\nComparing {enc_key}:")
        print(f"  Rank 0: shape={list(enc0.shape)}, checksum={torch.sum(torch.abs(enc0)).item():.6f}")
        print(f"  Rank 1: shape={list(enc1.shape)}, checksum={torch.sum(torch.abs(enc1)).item():.6f}")
        
        print(f"\n  Rank 0 - first 10 values: {enc0.flatten()[:10].tolist()}")
        print(f"  Rank 1 - first 10 values: {enc1.flatten()[:10].tolist()}")
        
        # Check if they're identical
        if torch.allclose(enc0, enc1):
            print("\nERROR: Rank 0 and Rank 1 have IDENTICAL encoder weights!")
        else:
            print("\nGOOD: Rank 0 and Rank 1 have DIFFERENT encoder weights")
            print(f"  Max difference: {torch.max(torch.abs(enc0 - enc1)).item():.6f}")
    
    # To recombine for a full model:
    print(f"\n{'='*60}")
    print("How to recombine:")
    print(f"{'='*60}")
    print("1. Load both rank files")
    print("2. For each parameter:")
    print("   - If it's a tensor-parallel weight, concatenate along the sharded dimension")
    print("   - If it's a replicated weight, use either rank's version")
    print("3. Save the combined state dict")
    print("\nExample for encoder weights (sharded along dim 0):")
    print("  combined_encoder = torch.cat([rank0_encoder, rank1_encoder], dim=0)")


if __name__ == "__main__":
    main()