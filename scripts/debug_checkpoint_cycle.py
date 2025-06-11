#!/usr/bin/env python3
"""
Debug script to test the full checkpoint save/load/merge cycle.
This script:
1. Runs regular training for a few steps
2. Saves checkpoint and captures weight statistics
3. Loads the checkpoint back and compares
4. Merges the distributed checkpoint (if distributed)
5. Loads merged checkpoint and compares
"""

import subprocess
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
import json
import logging
import os
from typing import Dict, Any
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_weight_stats(checkpoint_path: Path, prefix: str = "") -> Dict[str, Any]:
    """Extract summary statistics from a checkpoint file."""
    stats = {}
    
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    for name, param in state_dict.items():
        if param is None:
            continue
            
        param_data = param.cpu().float().numpy()
        
        # Store summary statistics
        stats[f"{prefix}{name}"] = {
            "shape": list(param.shape),
            "mean": float(np.mean(param_data)),
            "std": float(np.std(param_data)),
            "min": float(np.min(param_data)),
            "max": float(np.max(param_data)),
            "abs_mean": float(np.mean(np.abs(param_data))),
            # Sample first few values for direct comparison
            "first_10_values": param_data.flatten()[:10].tolist() if param_data.size > 0 else []
        }
    
    return stats


def print_weight_comparison(stats1: Dict[str, Any], stats2: Dict[str, Any], label1: str, label2: str):
    """Compare two sets of weight statistics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Weight comparison: {label1} vs {label2}")
    logger.info(f"{'='*60}")
    
    all_keys = set(stats1.keys()) | set(stats2.keys())
    
    mismatches = 0
    for key in sorted(all_keys):
        if key not in stats1:
            logger.warning(f"Key {key} missing in {label1}")
            mismatches += 1
            continue
        if key not in stats2:
            logger.warning(f"Key {key} missing in {label2}")
            mismatches += 1
            continue
            
        s1 = stats1[key]
        s2 = stats2[key]
        
        # Check if shapes match
        if s1["shape"] != s2["shape"]:
            logger.error(f"{key}: Shape mismatch! {label1}={s1['shape']}, {label2}={s2['shape']}")
            mismatches += 1
            continue
            
        # Compare statistics
        mean_diff = abs(s1["mean"] - s2["mean"])
        std_diff = abs(s1["std"] - s2["std"])
        max_diff = abs(s1["max"] - s2["max"])
        
        # Compare first few values
        values_match = np.allclose(s1["first_10_values"], s2["first_10_values"], rtol=1e-5, atol=1e-6)
        
        if mean_diff > 1e-5 or std_diff > 1e-5 or not values_match:
            logger.warning(f"{key}: Statistics differ!")
            logger.warning(f"  Mean: {s1['mean']:.6f} vs {s2['mean']:.6f} (diff: {mean_diff:.6e})")
            logger.warning(f"  Std:  {s1['std']:.6f} vs {s2['std']:.6f} (diff: {std_diff:.6e})")
            logger.warning(f"  Max:  {s1['max']:.6f} vs {s2['max']:.6f} (diff: {max_diff:.6e})")
            if not values_match:
                logger.warning(f"  First values differ: {s1['first_10_values'][:3]}... vs {s2['first_10_values'][:3]}...")
            mismatches += 1
        else:
            logger.debug(f"{key}: ✓ Match (mean={s1['mean']:.6f}, std={s1['std']:.6f})")
    
    logger.info(f"\nSummary: {mismatches} mismatches out of {len(all_keys)} parameters")
    return mismatches


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Debug checkpoint save/load/merge cycle")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--output-dir", type=str, default="./debug_checkpoint_output", help="Output directory")
    parser.add_argument("--num-features", type=int, default=8192, help="Number of features")
    parser.add_argument("--training-steps", type=int, default=100, help="Training steps")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Step 1: Run training with distributed
    logger.info("="*60)
    logger.info("STEP 1: Running distributed training")
    logger.info("="*60)
    
    train_cmd = [
        "torchrun", f"--nproc-per-node={args.world_size}",
        "scripts/train_clt.py",
        "--distributed",
        "--activation-source", "local_manifest",
        "--activation-path", "./activations_local_100M/gpt2/pile-uncopyrighted_train",
        "--model-name", "gpt2",
        "--num-features", str(args.num_features),
        "--activation-fn", "batchtopk",
        "--batchtopk-k", "200",
        "--output-dir", str(output_dir),
        "--learning-rate", "1e-4",
        "--training-steps", str(args.training_steps),
        "--train-batch-size-tokens", "1024",
        "--normalization-method", "auto",
        "--sparsity-lambda", "0.0",
        "--sparsity-c", "0.0",
        "--preactivation-coef", "0.0",
        "--aux-loss-factor", "0.03125",
        "--no-apply-sparsity-penalty-to-batchtopk",
        "--optimizer", "adamw",
        "--optimizer-beta2", "0.98",
        "--lr-scheduler", "linear_final20",
        "--seed", "42",
        "--activation-dtype", "float16",
        "--precision", "fp16",
        "--sampling-strategy", "sequential",
        "--log-interval", "50",
        "--eval-interval", "1000",
        "--checkpoint-interval", "50",
        "--dead-feature-window", "10000"
    ]
    
    logger.info(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Training failed with return code {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        sys.exit(1)
    
    logger.info("Training completed successfully")
    
    # Step 2: Check what checkpoints were created
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Analyzing saved checkpoints")
    logger.info("="*60)
    
    # Find the latest checkpoint
    checkpoint_dirs = list(output_dir.glob("step_*"))
    if not checkpoint_dirs:
        # Check for final checkpoint
        final_dir = output_dir / "final"
        if final_dir.exists():
            checkpoint_dirs = [final_dir]
        else:
            logger.error("No checkpoints found!")
            sys.exit(1)
    
    latest_checkpoint = sorted(checkpoint_dirs)[-1]
    logger.info(f"Using checkpoint: {latest_checkpoint}")
    
    # Check for distributed checkpoint files (.distcp)
    distcp_files = list(latest_checkpoint.glob("*.distcp"))
    if distcp_files:
        logger.info(f"Found {len(distcp_files)} distributed checkpoint files (.distcp)")
        for f in sorted(distcp_files):
            logger.info(f"  - {f.name}")
    
    # Check for consolidated model.safetensors
    consolidated_file = latest_checkpoint / "model.safetensors"
    if consolidated_file.exists():
        logger.info(f"\nFound consolidated checkpoint: {consolidated_file}")
        logger.info(f"  Size: {consolidated_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Analyze the consolidated checkpoint
        consolidated_stats = get_weight_stats(consolidated_file, prefix="consolidated_")
        logger.info("\nConsolidated model statistics:")
        for key, values in list(consolidated_stats.items())[:5]:
            logger.info(f"  {key}: shape={values['shape']}, mean={values['mean']:.6f}, std={values['std']:.6f}")
        
        # Store for later comparison
        all_rank_stats = {"consolidated": consolidated_stats}
    
    # Step 3: Merge the distributed checkpoint
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Merging distributed checkpoint")
    logger.info("="*60)
    
    merge_script = Path("scripts/merge_tp_checkpoint.py")
    if not merge_script.exists():
        logger.error(f"Merge script not found at {merge_script}")
        sys.exit(1)
    
    merged_path = latest_checkpoint / "merged_model.safetensors"
    
    # Find config file - it should be in the parent directory
    config_path = output_dir / "cfg.json"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
    
    merge_cmd = [
        "torchrun", f"--nproc-per-node={args.world_size}",
        str(merge_script),
        "--ckpt-dir", str(latest_checkpoint),
        "--cfg-json", str(config_path),
        "--output", str(merged_path)
    ]
    
    logger.info(f"Running: {' '.join(merge_cmd)}")
    result = subprocess.run(merge_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Merge failed with return code {result.returncode}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
    else:
        logger.info("Merge completed successfully")
        
        # Step 4: Compare merged checkpoint with distributed checkpoints
        if merged_path.exists():
            logger.info("\n" + "="*60)
            logger.info("STEP 4: Analyzing merged checkpoint")
            logger.info("="*60)
            
            merged_stats = get_weight_stats(merged_path, prefix="merged_")
            
            # Log some key statistics from merged model
            logger.info("\nMerged model statistics:")
            for key, values in list(merged_stats.items())[:5]:  # Show first 5 parameters
                logger.info(f"  {key}: shape={values['shape']}, mean={values['mean']:.6f}, std={values['std']:.6f}")
            
            # Compare shapes between consolidated and merged
            if "consolidated" in all_rank_stats:
                logger.info("\nComparing parameter shapes (consolidated vs merged):")
                consolidated_stats = all_rank_stats["consolidated"]
                shape_mismatches = 0
                
                # Find matching keys between consolidated and merged
                for cons_key in sorted(consolidated_stats.keys())[:20]:
                    # Find corresponding merged key
                    merged_key = cons_key.replace("consolidated_", "merged_")
                    
                    if merged_key in merged_stats:
                        cons_shape = consolidated_stats[cons_key]["shape"]
                        merged_shape = merged_stats[merged_key]["shape"]
                        
                        if cons_shape != merged_shape:
                            logger.warning(f"  SHAPE MISMATCH: {cons_key}")
                            logger.warning(f"    Consolidated: {cons_shape}")
                            logger.warning(f"    Merged:       {merged_shape}")
                            shape_mismatches += 1
                        else:
                            logger.debug(f"  ✓ {cons_key}: {cons_shape}")
                
                logger.info(f"\nTotal shape mismatches: {shape_mismatches}")
                
                if shape_mismatches > 0:
                    logger.error("\n*** CRITICAL: The consolidated checkpoint has incorrect shapes! ***")
                    logger.error("*** It appears to only contain one rank's portion of the model. ***")
        else:
            logger.error(f"Merged checkpoint not found at {merged_path}")
    
    # Step 5: Test loading the merged checkpoint
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Testing merged checkpoint loading")
    logger.info("="*60)
    
    if merged_path.exists():
        try:
            # Load config from parent directory
            config_path = output_dir / "cfg.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded config: num_features={config.get('num_features')}, num_layers={config.get('num_layers')}")
                
                # Try to load the merged checkpoint
                from clt.config import CLTConfig
                from clt.models.clt import CrossLayerTranscoder
                
                clt_config = CLTConfig(**config)
                model = CrossLayerTranscoder(clt_config, process_group=None, device="cpu")
                
                state_dict = load_file(str(merged_path))
                model.load_state_dict(state_dict)
                logger.info("✓ Successfully loaded merged checkpoint into CLT model!")
                
                # Do a simple forward pass test
                dummy_input = torch.randn(1, 768)  # GPT-2 hidden size
                dummy_layer_idx = torch.tensor([0])
                with torch.no_grad():
                    output = model(dummy_input, dummy_layer_idx)
                logger.info(f"✓ Forward pass successful! Output shape: {output.shape}")
                
            else:
                logger.error(f"Config file not found at {config_path}")
        except Exception as e:
            logger.error(f"Failed to load merged checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("Debug script completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()