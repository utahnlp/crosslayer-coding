#!/usr/bin/env python3
"""
Merge individual rank checkpoints into a single model checkpoint.
This works around the PyTorch distributed checkpoint bug.
"""

import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any
from safetensors.torch import save_file as save_safetensors_file

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def merge_tensor_parallel_weights(state_dicts: list, config: CLTConfig) -> Dict[str, torch.Tensor]:
    """
    Merge tensor-parallel weights from multiple ranks into a single state dict.
    
    Args:
        state_dicts: List of state dicts from each rank
        config: CLT configuration to understand model structure
        
    Returns:
        Merged state dict with full weights
    """
    merged_state = {}
    world_size = len(state_dicts)
    
    # Get all parameter names from first rank
    param_names = list(state_dicts[0].keys())
    
    for name in param_names:
        tensors = [sd[name] for sd in state_dicts]
        
        # Check if this is a tensor-parallel weight that needs concatenation
        if "encoder_module.encoders" in name:
            if "weight" in name:
                # Encoder weights are sharded along dim 0 (output features)
                merged_state[name] = torch.cat(tensors, dim=0)
                print(f"Merged encoder {name}: {tensors[0].shape} x {world_size} -> {merged_state[name].shape}")
            elif "bias" in name:
                # Encoder biases are also sharded along dim 0
                merged_state[name] = torch.cat(tensors, dim=0)
                print(f"Merged encoder bias {name}: {tensors[0].shape} x {world_size} -> {merged_state[name].shape}")
            else:
                # Other encoder parameters (shouldn't be any)
                merged_state[name] = tensors[0]
            
        elif "decoder_module.decoders" in name and "weight" in name:
            # Decoder weights are sharded along dim 1 (input features)
            merged_state[name] = torch.cat(tensors, dim=1)
            print(f"Merged decoder {name}: {tensors[0].shape} x {world_size} -> {merged_state[name].shape}")
            
        elif "log_threshold" in name:
            # For BatchTopK threshold, concatenate the per-layer thresholds
            merged_state[name] = torch.cat(tensors, dim=1)
            print(f"Merged threshold {name}: {tensors[0].shape} x {world_size} -> {merged_state[name].shape}")
            
        else:
            # For replicated parameters (biases, layer norms, etc.), use rank 0's version
            merged_state[name] = tensors[0]
            
            # Verify all ranks have identical replicated parameters
            for i in range(1, world_size):
                if not torch.allclose(tensors[0], tensors[i], atol=1e-6):
                    print(f"WARNING: Replicated parameter {name} differs between ranks!")
    
    return merged_state


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge tensor-parallel rank checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="./debug_weight_check/latest",
                        help="Directory containing rank checkpoint files")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path for merged model (defaults to checkpoint_dir/model_merged.safetensors)")
    parser.add_argument("--num-ranks", type=int, default=2,
                        help="Number of ranks to merge")
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory {checkpoint_dir} does not exist!")
        return
    
    # Load config
    config_path = checkpoint_dir.parent / "cfg.json"
    if not config_path.exists():
        print(f"ERROR: Config file {config_path} not found!")
        return
        
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    print(f"\n{'='*60}")
    print(f"Merging {args.num_ranks} rank checkpoints from {checkpoint_dir}")
    print(f"{'='*60}")
    
    # Load all rank checkpoints
    state_dicts = []
    for rank in range(args.num_ranks):
        rank_path = checkpoint_dir / f"rank_{rank}_model.pt"
        if not rank_path.exists():
            print(f"ERROR: Rank file {rank_path} not found!")
            print("Make sure to run training with the updated checkpointing code that saves individual rank files.")
            return
        
        print(f"Loading {rank_path}...")
        state_dict = torch.load(rank_path, map_location="cpu")
        state_dicts.append(state_dict)
    
    # Merge the state dicts
    print(f"\nMerging {args.num_ranks} rank state dicts...")
    merged_state = merge_tensor_parallel_weights(state_dicts, config)
    
    # Save merged model
    output_path = args.output_path
    if output_path is None:
        output_path = checkpoint_dir / "model_merged.safetensors"
    else:
        output_path = Path(output_path)
    
    print(f"\nSaving merged model to {output_path}...")
    save_safetensors_file(merged_state, str(output_path))
    
    # Verify the merged model
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    
    # Create a single-GPU model to verify loading works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossLayerTranscoder(
        config,
        process_group=None,  # Single GPU mode
        device=device
    )
    
    # Load the merged state
    model.load_state_dict(merged_state)
    print("✓ Successfully loaded merged state dict into single-GPU model")
    
    # Check some key parameters
    enc_weight = model.encoder_module.encoders[0].weight
    print(f"\nMerged encoder shape: {enc_weight.shape}")
    print(f"Expected shape: [{config.num_features}, {config.d_model}]")
    
    if enc_weight.shape[0] == config.num_features:
        print("✓ Encoder dimensions correct!")
    else:
        print("✗ ERROR: Encoder dimensions incorrect!")
    
    # Print checksum for comparison
    checksum = torch.sum(torch.abs(enc_weight)).item()
    print(f"\nMerged encoder checksum: {checksum:.6f}")
    
    # Compare with individual rank checksums
    for rank in range(args.num_ranks):
        rank_enc = state_dicts[rank]["encoder_module.encoders.0.weight"]
        rank_checksum = torch.sum(torch.abs(rank_enc)).item()
        print(f"  Rank {rank} contribution: {rank_checksum:.6f}")
    
    expected_checksum = sum(torch.sum(torch.abs(state_dicts[rank]["encoder_module.encoders.0.weight"])).item() 
                           for rank in range(args.num_ranks))
    print(f"  Expected sum: {expected_checksum:.6f}")
    
    if abs(checksum - expected_checksum) < 0.1:
        print("✓ Checksums match!")
    else:
        print("✗ WARNING: Checksum mismatch!")
    
    print(f"\n{'='*60}")
    print(f"Merge completed! Merged model saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()