#!/usr/bin/env python3
"""
Script C (Simple): Load the merged model and compare with A and B summaries.
"""

import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import load_file as load_safetensors_file

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.debug_weights_A_train import get_weight_summary
from scripts.debug_weights_C_merge_load import compare_summaries
from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def main():
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    
    print(f"\n{'='*60}")
    print("STAGE C: Loading merged model")
    print(f"{'='*60}")
    
    # Look for merged model
    merged_path = checkpoint_dir / "model_merged.safetensors"
    if not merged_path.exists():
        merged_path = checkpoint_dir / "model.safetensors"
    
    if not merged_path.exists():
        print(f"ERROR: No merged model found at {merged_path}")
        print("Please run: python scripts/merge_rank_checkpoints.py")
        return
    
    print(f"Found merged model: {merged_path}")
    
    # Load config
    config_path = output_dir / "cfg.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    # Create single-GPU model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossLayerTranscoder(
        config,
        process_group=None,  # Single GPU mode
        device=device
    )
    
    # Load merged state
    print(f"\nLoading merged model...")
    state_dict_C = load_safetensors_file(str(merged_path))
    model.load_state_dict(state_dict_C)
    
    # Get weights
    summary_C = get_weight_summary(model, "C_")
    
    print("\nMerged model weight summary:")
    for key, val in summary_C.items():
        print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
        print(f"    mean={val['mean']:.6f}, std={val['std']:.6f}")
        if 'sample_5x5' in val and val['sample_5x5']:
            # Show first row of the sample
            first_row = val['sample_5x5'][0] if isinstance(val['sample_5x5'][0], list) else val['sample_5x5']
            print(f"    first values: {first_row[:5]}")
    
    # Save summary
    summary_file = output_dir / "weight_summary_C.json"
    with open(summary_file, "w") as f:
        json.dump(summary_C, f, indent=2)
    print(f"\nSaved weight summary to {summary_file}")
    
    # Load previous summaries and compare
    print(f"\n{'='*60}")
    print("COMPARING ALL STAGES")
    print(f"{'='*60}")
    
    # Load A summaries (from rank 0)
    summary_A_file = output_dir / "weight_summary_A_rank0.json"
    if summary_A_file.exists():
        with open(summary_A_file, "r") as f:
            summary_A = json.load(f)
        
        # Compare A vs C
        compare_summaries(summary_A, summary_C, "In-memory (A)", "Merged model (C)")
    
    # Load B summaries (from rank 0)
    summary_B_file = output_dir / "weight_summary_B_rank0.json"
    if summary_B_file.exists():
        with open(summary_B_file, "r") as f:
            summary_B = json.load(f)
        
        # Compare B vs C
        compare_summaries(summary_B, summary_C, "Loaded from distcp (B)", "Merged model (C)")
        
        # Also compare A vs B if both exist
        if summary_A_file.exists():
            compare_summaries(summary_A, summary_B, "In-memory (A)", "Loaded from distcp (B)")
    
    # Additional check: Compare with rank checksums
    print(f"\n{'='*60}")
    print("CHECKING MERGED VS INDIVIDUAL RANKS")
    print(f"{'='*60}")
    
    # Load individual rank files to verify merge
    rank0_path = checkpoint_dir / "rank_0_model.pt"
    rank1_path = checkpoint_dir / "rank_1_model.pt"
    
    if rank0_path.exists() and rank1_path.exists():
        rank0_state = torch.load(rank0_path, map_location="cpu")
        rank1_state = torch.load(rank1_path, map_location="cpu")
        
        enc_key = "encoder_module.encoders.0.weight"
        if enc_key in rank0_state and enc_key in rank1_state:
            rank0_checksum = torch.sum(torch.abs(rank0_state[enc_key])).item()
            rank1_checksum = torch.sum(torch.abs(rank1_state[enc_key])).item()
            merged_checksum = summary_C["C_encoder_0"]["checksum"]
            
            print(f"Encoder weight checksums:")
            print(f"  Rank 0: {rank0_checksum:.6f}")
            print(f"  Rank 1: {rank1_checksum:.6f}")
            print(f"  Sum: {rank0_checksum + rank1_checksum:.6f}")
            print(f"  Merged: {merged_checksum:.6f}")
            
            if abs(merged_checksum - (rank0_checksum + rank1_checksum)) < 0.1:
                print("✓ Merged checksum matches sum of ranks!")
            else:
                print("✗ ERROR: Merged checksum doesn't match!")
    
    print(f"\n{'='*60}")
    print("Stage C completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()