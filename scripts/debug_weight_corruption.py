#!/usr/bin/env python3
"""
Debug potential weight corruption in tensor-parallel checkpoint save/load process.
"""

import torch
import torch.distributed as dist
import os
import sys
import json
from pathlib import Path
import logging
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from safetensors.torch import save_file as save_safetensors_file, load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compare_weights(state_dict1, state_dict2, name1="Dict1", name2="Dict2"):
    """Compare two state dicts and report differences."""
    all_keys = set(state_dict1.keys()) | set(state_dict2.keys())
    
    differences = []
    for key in sorted(all_keys):
        if key not in state_dict1:
            differences.append(f"Key '{key}' missing in {name1}")
            continue
        if key not in state_dict2:
            differences.append(f"Key '{key}' missing in {name2}")
            continue
            
        t1 = state_dict1[key]
        t2 = state_dict2[key]
        
        if t1.shape != t2.shape:
            differences.append(f"Shape mismatch for '{key}': {t1.shape} vs {t2.shape}")
            continue
            
        # Compare values (move to CPU for comparison)
        t1_cpu = t1.cpu()
        t2_cpu = t2.cpu()
        if not torch.allclose(t1_cpu, t2_cpu, rtol=1e-5, atol=1e-7):
            max_diff = (t1_cpu - t2_cpu).abs().max().item()
            rel_diff = ((t1_cpu - t2_cpu).abs() / (t1_cpu.abs() + 1e-8)).max().item()
            differences.append(f"Value mismatch for '{key}': max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
            
            # Sample some differences
            if t1_cpu.numel() > 10:
                diff_indices = (t1_cpu - t2_cpu).abs().flatten().topk(min(5, t1_cpu.numel())).indices
                for idx in diff_indices[:3]:
                    idx_tuple = np.unravel_index(idx.item(), t1_cpu.shape)
                    differences.append(f"  At {idx_tuple}: {t1_cpu[idx_tuple].item():.6f} vs {t2_cpu[idx_tuple].item():.6f}")
    
    return differences


def test_simple_save_load():
    """Test basic save/load without distributed training."""
    logger.info("=== TESTING SIMPLE SAVE/LOAD ===")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create a small model
    config = CLTConfig(
        d_model=64,
        num_features=128,
        num_layers=2,
        activation_fn="relu",
    )
    
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    
    # Get original state
    original_state = model.state_dict()
    
    # Save
    temp_path = Path("/tmp/test_model.safetensors")
    save_safetensors_file(original_state, str(temp_path))
    
    # Load
    loaded_state = load_safetensors_file(str(temp_path))
    
    # Compare
    differences = compare_weights(original_state, loaded_state, "Original", "Loaded")
    
    if differences:
        logger.error(f"Found {len(differences)} differences in simple save/load:")
        for diff in differences[:10]:
            logger.error(f"  {diff}")
    else:
        logger.info("Simple save/load test PASSED - no differences found")
    
    # Clean up
    temp_path.unlink(missing_ok=True)
    
    return len(differences) == 0


def check_distributed_checkpoint_files():
    """Check the actual checkpoint files for issues."""
    logger.info("\n=== CHECKING DISTRIBUTED CHECKPOINT FILES ===")
    
    # Look for distributed checkpoint directories
    checkpoint_dirs = [
        "clt_training_logs/gpt2_batchtopk/step_20000",
        "clt_training_logs/gpt2_batchtopk/step_40000",
        "clt_training_logs/gpt2_batchtopk/step_60000",
        "clt_training_logs/gpt2_batchtopk/step_80000",
    ]
    
    for ckpt_dir in checkpoint_dirs:
        if not os.path.exists(ckpt_dir):
            continue
            
        logger.info(f"\nChecking {ckpt_dir}:")
        
        # Check for rank-specific files
        rank_files = []
        for rank in range(2):  # Assuming 2 GPUs
            rank_file = Path(ckpt_dir) / f"model_rank{rank}.safetensors"
            if rank_file.exists():
                rank_files.append(rank_file)
                logger.info(f"  Found rank file: {rank_file}")
                
                # Load and check basic stats
                state_dict = load_safetensors_file(str(rank_file))
                logger.info(f"    Keys: {len(state_dict)}")
                
                # Check a few weights
                for key in list(state_dict.keys())[:3]:
                    tensor = state_dict[key]
                    logger.info(f"    {key}: shape={tensor.shape}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        
        # Check merged file
        merged_file = Path(ckpt_dir) / "model.safetensors"
        if merged_file.exists():
            logger.info(f"  Found merged file: {merged_file}")
            state_dict = load_safetensors_file(str(merged_file))
            logger.info(f"    Keys: {len(state_dict)}")
            
            # Check if shapes are correct
            encoder_key = "encoder_module.encoders.0.weight"
            if encoder_key in state_dict:
                shape = state_dict[encoder_key].shape
                logger.info(f"    Encoder shape: {shape} (should be [32768, 768] for full model)")
                if shape[0] != 32768:
                    logger.error(f"    ERROR: Encoder has wrong feature dimension: {shape[0]}")


def check_weight_statistics():
    """Compare weight statistics between checkpoints."""
    logger.info("\n=== COMPARING WEIGHT STATISTICS ACROSS CHECKPOINTS ===")
    
    checkpoints = [
        ("clt_training_logs/gpt2_batchtopk/step_20000/model.safetensors", "Step 20k"),
        ("clt_training_logs/gpt2_batchtopk/step_40000/model.safetensors", "Step 40k"),
        ("clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors", "Step 90k"),
    ]
    
    key_weights = [
        "encoder_module.encoders.0.weight",
        "decoder_module.decoders.0->0.weight",
    ]
    
    stats_by_checkpoint = {}
    
    for ckpt_path, ckpt_name in checkpoints:
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
            
        state_dict = load_safetensors_file(ckpt_path)
        stats_by_checkpoint[ckpt_name] = {}
        
        for key in key_weights:
            if key in state_dict:
                tensor = state_dict[key]
                stats_by_checkpoint[ckpt_name][key] = {
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item(),
                    "abs_max": tensor.abs().max().item(),
                    "shape": tensor.shape,
                }
    
    # Compare statistics
    logger.info("\nWeight statistics evolution:")
    for key in key_weights:
        logger.info(f"\n{key}:")
        for ckpt_name in stats_by_checkpoint:
            if key in stats_by_checkpoint[ckpt_name]:
                stats = stats_by_checkpoint[ckpt_name][key]
                logger.info(f"  {ckpt_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                           f"abs_max={stats['abs_max']:.6f}, shape={stats['shape']}")


def check_merge_correctness():
    """Verify if the merge process is correct by comparing with individual rank files."""
    logger.info("\n=== CHECKING MERGE CORRECTNESS ===")
    
    # This would require loading the individual rank files and manually merging them
    # to compare with the merged checkpoint
    
    # For now, just check if the merged file has the right total number of features
    merged_path = "clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors"
    if os.path.exists(merged_path):
        state_dict = load_safetensors_file(merged_path)
        
        # Check encoder shapes
        for i in range(12):
            key = f"encoder_module.encoders.{i}.weight"
            if key in state_dict:
                shape = state_dict[key].shape
                if shape[0] != 32768:
                    logger.error(f"ERROR: {key} has wrong shape: {shape}, expected [32768, 768]")
                else:
                    logger.info(f"OK: {key} has correct shape: {shape}")


def main():
    logger.info("=== DEBUGGING WEIGHT CORRUPTION IN DISTRIBUTED CHECKPOINTS ===")
    
    # Test 1: Basic save/load
    simple_ok = test_simple_save_load()
    
    # Test 2: Check distributed checkpoint files
    check_distributed_checkpoint_files()
    
    # Test 3: Compare weight statistics
    check_weight_statistics()
    
    # Test 4: Check merge correctness
    check_merge_correctness()
    
    logger.info("\n=== SUMMARY ===")
    if not simple_ok:
        logger.error("Basic save/load is broken - this is a fundamental issue")
    else:
        logger.info("Basic save/load works correctly")
        logger.info("\nThe issue appears to be in the distributed training/checkpointing process.")
        logger.info("Possible causes:")
        logger.info("  1. Incorrect gradient synchronization during distributed training")
        logger.info("  2. Wrong reduction operation (sum vs mean) in tensor parallelism")
        logger.info("  3. Incorrect merging of distributed checkpoints")
        logger.info("  4. Scale factor issue in aux_loss or gradient accumulation")


if __name__ == "__main__":
    main()