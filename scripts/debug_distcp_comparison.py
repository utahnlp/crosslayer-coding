#!/usr/bin/env python3
"""
Simple script to check if .distcp files are correct by comparing with merged model.
Assumes training has already been done and checkpoints exist.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
import subprocess
from typing import Dict, Any

# Imports for distributed checkpoint loading
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from safetensors.torch import load_file as load_safetensors_file

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


def compare_summaries(sum1: Dict[str, Any], sum2: Dict[str, Any], label1: str, label2: str):
    """Compare two weight summaries."""
    print(f"\n{'='*60}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'='*60}")
    
    for key in sorted(set(sum1.keys()) | set(sum2.keys())):
        if key not in sum1:
            print(f"❌ {key}: Missing in {label1}")
            continue
        if key not in sum2:
            print(f"❌ {key}: Missing in {label2}")
            continue
            
        s1 = sum1[key]
        s2 = sum2[key]
        
        # Compare shapes
        if s1["shape"] != s2["shape"]:
            print(f"❌ {key}: Shape mismatch! {s1['shape']} vs {s2['shape']}")
            continue
            
        # Compare checksums
        checksum_diff = abs(s1["checksum"] - s2["checksum"]) / max(s1["checksum"], 1e-10)
        
        if checksum_diff < 1e-5:
            print(f"✅ {key}: Match (checksum diff: {checksum_diff:.2e})")
        else:
            print(f"❌ {key}: MISMATCH!")
            print(f"   Shape: {s1['shape']}")
            print(f"   Mean: {s1['mean']:.6f} vs {s2['mean']:.6f}")
            print(f"   Std: {s1['std']:.6f} vs {s2['std']:.6f}")
            print(f"   Checksum: {s1['checksum']:.6f} vs {s2['checksum']:.6f} (diff: {checksum_diff:.2%})")
            print(f"   Sample [0,0:5]: {s1['sample_5x5'][0][:5]}")
            print(f"            vs: {s2['sample_5x5'][0][:5]}")


def main():
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    config_path = output_dir / "cfg.json"
    
    # Load config
    with open(config_path, "r") as f:
        loaded_config_dict = json.load(f)
    loaded_config = CLTConfig(**loaded_config_dict)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("STAGE B: Loading model from .distcp files")
        print(f"{'='*60}")
    
    # B. Load from distributed checkpoint
    loaded_model_B = CrossLayerTranscoder(
        loaded_config,
        process_group=dist.group.WORLD if world_size > 1 else None,
        device=device
    )
    loaded_model_B.eval()
    
    # Load distributed checkpoint
    state_dict_B = loaded_model_B.state_dict()
    load_state_dict(
        state_dict=state_dict_B,
        storage_reader=FileSystemReader(str(checkpoint_dir)),
        planner=DefaultLoadPlanner(),
        no_dist=False,
    )
    loaded_model_B.load_state_dict(state_dict_B)
    
    # Get weights from loaded model
    summary_B = get_weight_summary(loaded_model_B, "B_")
    
    if rank == 0:
        print("\nLoaded model weight summary from .distcp files:")
        for key, val in summary_B.items():
            print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
    
    # C. Merge and load (only if distributed)
    if world_size > 1:
        if rank == 0:
            print(f"\n{'='*60}")
            print("STAGE C: Merging checkpoint and loading from safetensors")
            print(f"{'='*60}")
        
        dist.barrier()
        
        # Run merge
        merged_path = checkpoint_dir / "merged_model.safetensors"
        merge_script = project_root / "scripts" / "merge_tp_checkpoint.py"
        
        if rank == 0:
            # First, ensure any existing merged file is removed
            if merged_path.exists():
                merged_path.unlink()
        
        dist.barrier()
        
        # All ranks participate in merge
        merge_cmd = [
            sys.executable,  # Use same Python interpreter
            str(merge_script),
            "--ckpt-dir", str(checkpoint_dir),
            "--cfg-json", str(config_path),
            "--output", str(merged_path)
        ]
        
        # Set up environment for subprocess
        env = os.environ.copy()
        
        if rank == 0:
            print(f"Running merge on all ranks...")
        
        # Run merge script directly (all ranks)
        result = subprocess.run(merge_cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            if rank == 0:
                print(f"Merge failed on rank {rank}!")
                print(f"stderr: {result.stderr}")
        
        dist.barrier()
        
        # Only rank 0 loads and compares the merged model
        if rank == 0 and merged_path.exists():
            print("\nLoading merged model...")
            
            # Create single-GPU model
            single_model = CrossLayerTranscoder(
                loaded_config,
                process_group=None,
                device=device
            )
            single_model.eval()
            
            # Load merged checkpoint
            state_dict_C = load_safetensors_file(str(merged_path))
            single_model.load_state_dict(state_dict_C)
            
            # Get weights
            summary_C = get_weight_summary(single_model, "C_")
            
            # Compare B vs C
            compare_summaries(summary_B, summary_C, "Loaded from distcp (B)", "Loaded from merged (C)")
            
            # Check the consolidated model.safetensors file that was saved during training
            print(f"\n{'='*60}")
            print("BONUS: Checking consolidated model.safetensors from training")
            print(f"{'='*60}")
            
            consolidated_path = checkpoint_dir / "model.safetensors"
            if consolidated_path.exists():
                # Load consolidated checkpoint
                state_dict_consolidated = load_safetensors_file(str(consolidated_path))
                
                # Check shapes
                print("\nConsolidated checkpoint shapes:")
                for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                    if key in state_dict_consolidated:
                        print(f"  {key}: {state_dict_consolidated[key].shape}")
                
                # Compare with expected shapes
                print("\nExpected shapes (from merged model):")
                for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                    if key in state_dict_C:
                        print(f"  {key}: {state_dict_C[key].shape}")
    else:
        if rank == 0:
            print("\nSingle GPU run - no merging needed")
            print("Checking consolidated model.safetensors...")
            
            consolidated_path = checkpoint_dir / "model.safetensors"
            if consolidated_path.exists():
                # Load consolidated checkpoint
                state_dict_consolidated = load_safetensors_file(str(consolidated_path))
                
                # Create single-GPU model to compare
                single_model = CrossLayerTranscoder(
                    loaded_config,
                    process_group=None,
                    device=device
                )
                single_model.eval()
                single_model.load_state_dict(state_dict_consolidated)
                
                # Get weights
                summary_consolidated = get_weight_summary(single_model, "Consolidated_")
                
                # Compare
                compare_summaries(summary_B, summary_consolidated, "Loaded from distcp (B)", "Consolidated model.safetensors")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Weight comparison completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()