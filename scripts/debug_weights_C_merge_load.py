#!/usr/bin/env python3
"""
Script C: Merge distributed checkpoint and load weights.
Compares with previous stages.
"""

import os
import sys
import json
import torch
from pathlib import Path
import numpy as np
import subprocess
from typing import Dict, Any
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
    """Compare two weight summaries, ignoring prefixes."""
    print(f"\n{'='*60}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'='*60}")
    
    # Extract base names without prefixes
    def get_base_name(key):
        parts = key.split('_')
        if len(parts) >= 2 and parts[0] in ['A', 'B', 'C']:
            return '_'.join(parts[1:])
        return key
    
    # Create maps with base names
    sum1_map = {get_base_name(k): (k, v) for k, v in sum1.items()}
    sum2_map = {get_base_name(k): (k, v) for k, v in sum2.items()}
    
    all_base_names = set(sum1_map.keys()) | set(sum2_map.keys())
    
    for base_name in sorted(all_base_names):
        if base_name not in sum1_map:
            print(f"❌ {base_name}: Missing in {label1}")
            continue
        if base_name not in sum2_map:
            print(f"❌ {base_name}: Missing in {label2}")
            continue
            
        key1, s1 = sum1_map[base_name]
        key2, s2 = sum2_map[base_name]
        
        # Compare shapes
        if s1["shape"] != s2["shape"]:
            print(f"❌ {base_name}: Shape mismatch! {s1['shape']} vs {s2['shape']}")
            continue
            
        # Compare checksums
        checksum_diff = abs(s1["checksum"] - s2["checksum"]) / max(s1["checksum"], 1e-10)
        
        if checksum_diff < 1e-5:
            print(f"✅ {base_name}: Match (checksum diff: {checksum_diff:.2e})")
        else:
            print(f"❌ {base_name}: MISMATCH!")
            print(f"   Shape: {s1['shape']}")
            print(f"   Mean: {s1['mean']:.6f} vs {s2['mean']:.6f}")
            print(f"   Std: {s1['std']:.6f} vs {s2['std']:.6f}")
            print(f"   Checksum: {s1['checksum']:.6f} vs {s2['checksum']:.6f} (diff: {checksum_diff:.2%})")
            print(f"   Sample [0,0:5]: {s1['sample_5x5'][0][:5]}")
            print(f"            vs: {s2['sample_5x5'][0][:5]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    output_dir = Path("./debug_weight_check")
    checkpoint_dir = output_dir / "latest"
    config_path = output_dir / "cfg.json"
    merged_path = checkpoint_dir / "merged_model.safetensors"
    
    print(f"\n{'='*60}")
    print("STAGE C: Merging checkpoint and loading from safetensors")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path, "r") as f:
        loaded_config_dict = json.load(f)
    loaded_config = CLTConfig(**loaded_config_dict)
    
    # First, run the merge script with torchrun
    merge_script = project_root / "scripts" / "merge_tp_checkpoint.py"
    
    # Remove existing merged file
    if merged_path.exists():
        merged_path.unlink()
        print(f"Removed existing merged file")
    
    print(f"Running merge script with torchrun...")
    
    # Determine world size from existing .distcp files
    distcp_files = list(checkpoint_dir.glob("*.distcp"))
    world_size = len(distcp_files)
    print(f"Detected world_size={world_size} from {len(distcp_files)} .distcp files")
    
    merge_cmd = [
        "torchrun",
        f"--nproc-per-node={world_size}",
        str(merge_script),
        "--ckpt-dir", str(checkpoint_dir),
        "--cfg-json", str(config_path),
        "--output", str(merged_path)
    ]
    
    result = subprocess.run(merge_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Merge failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return
    else:
        print(f"Merge completed successfully")
    
    # Load merged model
    if merged_path.exists():
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
        
        print("\nMerged model weight summary:")
        for key, val in summary_C.items():
            print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
        
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
    
    print(f"\n{'='*60}")
    print("Stage C completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()