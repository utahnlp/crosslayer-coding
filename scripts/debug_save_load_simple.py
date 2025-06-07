#!/usr/bin/env python3
"""
Simplified debug script to track model weights during training and after save/load.
This version uses the existing trainer infrastructure more directly.
"""

import torch
import torch.distributed as dist
import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_weight_stats(model, prefix=""):
    """Get statistics for key model weights."""
    stats = {}
    
    # Check encoder and decoder weights
    for name, param in model.named_parameters():
        if "encoder" in name and "weight" in name and "0" in name:
            stats[f"{prefix}encoder_weight_mean"] = param.data.mean().item()
            stats[f"{prefix}encoder_weight_std"] = param.data.std().item()
            stats[f"{prefix}encoder_weight_shape"] = list(param.shape)
            break
    
    for name, param in model.named_parameters():
        if "decoder" in name and "weight" in name and "0" in name:
            stats[f"{prefix}decoder_weight_mean"] = param.data.mean().item()
            stats[f"{prefix}decoder_weight_std"] = param.data.std().item()
            stats[f"{prefix}decoder_weight_shape"] = list(param.shape)
            break
    
    return stats


def run_simple_test():
    """Run a simplified distributed training test."""
    
    # Check if running with torchrun
    if "RANK" not in os.environ:
        logger.error("This script must be run with torchrun")
        logger.error("Example: torchrun --nproc_per_node=2 scripts/debug_save_load_simple.py")
        return
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    logger.info(f"Starting simple test on rank {rank}/{world_size}")
    
    # Create temporary directory
    if rank == 0:
        temp_dir = tempfile.mkdtemp(prefix="clt_debug_simple_")
        logger.info(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = None
    
    # Set CUDA device for distributed training
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    
    # Use shared temp dir path for all ranks
    if temp_dir is None:
        temp_dir = f"/tmp/clt_debug_simple_rank{rank}"
    
    # Configuration matching your actual working setup
    clt_config = CLTConfig(
        d_model=768,  # GPT-2 hidden size
        num_features=32768,  # Same as your working config
        num_layers=12,  # GPT-2 layers
        activation_fn="batchtopk",
        batchtopk_k=200,
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=10,  # Just a few steps for testing
        train_batch_size_tokens=1024,  # Same as your working config
        checkpoint_interval=5,  # Enable checkpointing to reproduce the error
        eval_interval=5,
        log_interval=1,
        enable_wandb=False,
        precision="fp16",  # Same as your working config
        optimizer="adamw",
        optimizer_beta2=0.98,  # Same as your working config
        lr_scheduler="constant",  # Simplified for testing
        aux_loss_factor=0.03125,
        sparsity_lambda=0.0,  # Same as your working config
        sparsity_c=0.0,
        preactivation_coef=0.0,
        apply_sparsity_penalty_to_batchtopk=False,
        activation_source="local_manifest",
        activation_path="./activations_local_100M/gpt2/pile-uncopyrighted_train",  # 100M dataset
        activation_dtype="float16",  # Same as your working config
        normalization_method="auto",
        sampling_strategy="sequential",
        dead_feature_window=10000,  # Same as your working config
        seed=42,
    )
    
    # Initialize trainer (handles distributed setup internally)
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=temp_dir,
        distributed=(world_size > 1),
    )
    
    # Get initial weights
    initial_stats = get_weight_stats(trainer.model, "initial_")
    if rank == 0:
        logger.info(f"Initial weight stats: {json.dumps(initial_stats, indent=2)}")
    
    # Run training
    logger.info(f"Rank {rank}: Starting training...")
    trainer.train()
    
    # Get final in-memory stats
    final_memory_stats = get_weight_stats(trainer.model, "final_memory_")
    if rank == 0:
        logger.info(f"Final in-memory weight stats: {json.dumps(final_memory_stats, indent=2)}")
    
    # Wait for all ranks to finish training
    if trainer.distributed:
        dist.barrier()
    
    # Now test checkpoint loading (only on rank 0 for simplicity)
    if rank == 0:
        logger.info("\n=== TESTING CHECKPOINT LOAD ===")
        
        # Find the latest checkpoint
        checkpoint_dirs = list(Path(temp_dir).glob("step_*"))
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda p: int(p.name.split("_")[1]))
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            
            # For distributed checkpoints, we need to merge first
            if world_size > 1:
                logger.info("Running merge script...")
                import subprocess
                
                merge_script = project_root / "scripts" / "merge_tp_checkpoint.py"
                merge_cmd = [
                    "python", str(merge_script),
                    "--checkpoint-dir", str(latest_checkpoint),
                    "--output-path", str(temp_dir / "merged_model.safetensors"),
                    "--num-features", str(clt_config.num_features),
                    "--d-model", str(clt_config.d_model),
                ]
                
                result = subprocess.run(merge_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Merge failed: {result.stderr}")
                    return
                
                logger.info("Merge successful")
                
                # Load merged checkpoint
                from safetensors.torch import load_file as load_safetensors_file
                from clt.models.clt import CrossLayerTranscoder
                
                merged_model = CrossLayerTranscoder(clt_config, device=trainer.device, process_group=None)
                state_dict = load_safetensors_file(str(temp_dir / "merged_model.safetensors"))
                merged_model.load_state_dict(state_dict)
                
                loaded_stats = get_weight_stats(merged_model, "loaded_")
                logger.info(f"Loaded weight stats: {json.dumps(loaded_stats, indent=2)}")
                
                # Compare weights
                logger.info("\n=== WEIGHT COMPARISON ===")
                for key in ["encoder_weight_mean", "encoder_weight_std", "decoder_weight_mean", "decoder_weight_std"]:
                    mem_key = f"final_memory_{key}"
                    load_key = f"loaded_{key}"
                    if mem_key in final_memory_stats and load_key in loaded_stats:
                        mem_val = final_memory_stats[mem_key]
                        load_val = loaded_stats[load_key]
                        diff = abs(load_val - mem_val)
                        rel_diff = diff / (abs(mem_val) + 1e-8) * 100
                        logger.info(f"{key}: memory={mem_val:.6f}, loaded={load_val:.6f}, diff={diff:.2e} ({rel_diff:.1f}%)")
                
                # Quick evaluation test
                logger.info("\n=== EVALUATION TEST ===")
                from clt.training.evaluator import CLTEvaluator
                evaluator = CLTEvaluator(model=merged_model, device=trainer.device)
                
                # Get one batch from trainer's activation store
                trainer.activation_store.reset_iterator()
                inputs, targets = next(trainer.activation_store)
                
                metrics = evaluator.compute_metrics(inputs, targets)
                logger.info(f"Loaded model metrics: NMSE={metrics.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}, "
                           f"EV={metrics.get('reconstruction/explained_variance', -1):.4f}")
    
    # The trainer handles process group cleanup automatically
    
    if rank == 0:
        logger.info(f"\nTest complete. Results in: {temp_dir}")
        logger.info("To keep the directory, run with --keep-temp flag")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple debug test for save/load")
    parser.add_argument("--keep-temp", action="store_true", help="Don't delete temporary directory")
    args = parser.parse_args()
    
    run_simple_test()


if __name__ == "__main__":
    main()