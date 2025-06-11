#!/usr/bin/env python3
"""
Debug script to check weight vectors before and after distributed save/load.
This is a modified version of train_clt.py that:
1. Trains a model briefly with tensor parallelism
2. Reports weight statistics before closing
3. Reloads the model and checks the same tensors
4. Merges the distributed checkpoint and checks again
"""

import argparse
import torch
import torch.distributed as dist
from pathlib import Path
import logging
import json
import numpy as np
import os
from typing import Dict, Any

# Import CLT components
from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer
from clt.models.clt import CrossLayerTranscoder
from clt.training.checkpointing import CheckpointManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_weight_stats(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, Any]:
    """Extract summary statistics from model weights."""
    stats = {}
    
    # Get some specific weight tensors and their statistics
    for name, param in model.named_parameters():
        if param is None:
            continue
            
        param_data = param.data.cpu().float().numpy()
        
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
    
    for key in sorted(all_keys):
        if key not in stats1:
            logger.warning(f"Key {key} missing in {label1}")
            continue
        if key not in stats2:
            logger.warning(f"Key {key} missing in {label2}")
            continue
            
        s1 = stats1[key]
        s2 = stats2[key]
        
        # Check if shapes match
        if s1["shape"] != s2["shape"]:
            logger.error(f"{key}: Shape mismatch! {label1}={s1['shape']}, {label2}={s2['shape']}")
            continue
            
        # Compare statistics
        mean_diff = abs(s1["mean"] - s2["mean"])
        std_diff = abs(s1["std"] - s2["std"])
        max_diff = abs(s1["max"] - s2["max"])
        
        # Compare first few values
        values_match = s1["first_10_values"] == s2["first_10_values"]
        
        if mean_diff > 1e-6 or std_diff > 1e-6 or not values_match:
            logger.warning(f"{key}: Statistics differ!")
            logger.warning(f"  Mean: {s1['mean']:.6f} vs {s2['mean']:.6f} (diff: {mean_diff:.6e})")
            logger.warning(f"  Std:  {s1['std']:.6f} vs {s2['std']:.6f} (diff: {std_diff:.6e})")
            logger.warning(f"  Max:  {s1['max']:.6f} vs {s2['max']:.6f} (diff: {max_diff:.6e})")
            if not values_match:
                logger.warning(f"  First values differ: {s1['first_10_values'][:3]}... vs {s2['first_10_values'][:3]}...")
        else:
            logger.info(f"{key}: ✓ Match (mean={s1['mean']:.6f}, std={s1['std']:.6f})")


def main():
    """Main debug function."""
    # Simplified argument parsing
    parser = argparse.ArgumentParser(description="Debug distributed CLT training save/load")
    parser.add_argument("--output-dir", type=str, default="./debug_clt_output", help="Output directory")
    parser.add_argument("--num-features", type=int, default=768, help="Number of features per layer")
    parser.add_argument("--training-steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--activation-path", type=str, default="./activations_local_100M/gpt2/pile-uncopyrighted_train", help="Path to activation data")
    args = parser.parse_args()
    
    # Initialize distributed if launched with torchrun
    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure CLT to match your training run
    clt_config = CLTConfig(
        num_features=args.num_features,  # Smaller for debug
        num_layers=12,  # GPT-2
        d_model=768,    # GPT-2
        activation_fn="batchtopk",  # Match your config
        batchtopk_k=200,  # Match your config
        model_name="gpt2",
        clt_dtype="float16"  # Match your precision
    )
    
    # Configure training to match your settings
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=args.training_steps,
        train_batch_size_tokens=1024,  # Match your config
        activation_source="local_manifest",
        activation_path=args.activation_path,
        activation_dtype="float16",  # Match your config
        normalization_method="auto",
        sparsity_lambda=0.0,  # Match your config
        sparsity_c=0.0,  # Match your config
        preactivation_coef=0.0,  # Match your config
        aux_loss_factor=0.03125,  # Match your config
        apply_sparsity_penalty_to_batchtopk=False,  # Match your no-apply setting
        optimizer="adamw",
        optimizer_beta2=0.98,  # Match your config
        lr_scheduler="linear_final20",
        precision="fp16",  # Match your config
        log_interval=10,
        eval_interval=25,
        checkpoint_interval=25,
        enable_wandb=False,
    )
    
    # Create and run trainer
    logger.info("Creating trainer...")
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=str(output_dir),
        device=device,
    )
    
    logger.info("Starting training...")
    trained_model = trainer.train()
    
    # Get weight statistics after training
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Getting weight statistics from trained model (in memory)")
    logger.info("="*60)
    trained_stats = get_weight_stats(trained_model, prefix="trained_")
    
    # Force checkpoint save
    checkpoint_dir = output_dir / "final"
    logger.info(f"\nSaving final checkpoint to {checkpoint_dir}")
    trainer.checkpoint_manager.save_checkpoint(
        trainer.clt_model,
        trainer.optimizer,
        trainer.scheduler,
        trainer.grad_scaler,
        trainer.trainer_state,
        checkpoint_dir=checkpoint_dir,
        is_final=True
    )
    
    # Wait for all ranks to finish saving
    if world_size > 1:
        dist.barrier()
    
    # Now load the checkpoint back
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Loading checkpoint and checking weights")
    logger.info("="*60)
    
    # Create a new model instance
    loaded_model = CrossLayerTranscoder(
        clt_config,
        process_group=trainer.clt_model.process_group if world_size > 1 else None,
        device=device
    )
    
    # Load the checkpoint
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir),
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size
    )
    
    # Try to load the distributed checkpoint
    if world_size > 1:
        state_dict_path = checkpoint_dir / f"rank_{rank}_model.pt"
        if state_dict_path.exists():
            logger.info(f"Loading distributed checkpoint from {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location=device)
            loaded_model.load_state_dict(state_dict)
            
            loaded_stats = get_weight_stats(loaded_model, prefix="loaded_dist_")
            print_weight_comparison(trained_stats, loaded_stats, "Trained", "Loaded (Distributed)")
    
    # Now attempt to merge and load the full model (only on rank 0)
    if rank == 0 and world_size > 1:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Attempting to merge distributed checkpoint")
        logger.info("="*60)
        
        # Check if merge script exists
        merge_script = Path(__file__).parent / "merge_tp_checkpoint.py"
        if merge_script.exists():
            import subprocess
            
            # Run the merge script
            merge_cmd = [
                "torchrun",
                f"--nproc-per-node={world_size}",
                str(merge_script),
                "--checkpoint-dir", str(checkpoint_dir),
                "--output-path", str(checkpoint_dir / "merged_model.safetensors")
            ]
            
            logger.info(f"Running merge command: {' '.join(merge_cmd)}")
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Merge successful!")
                
                # Load the merged model
                from safetensors.torch import load_file
                merged_path = checkpoint_dir / "merged_model.safetensors"
                if merged_path.exists():
                    logger.info(f"Loading merged model from {merged_path}")
                    
                    # Create a single-GPU model for comparison
                    single_model = CrossLayerTranscoder(
                        clt_config,
                        process_group=None,
                        device=device
                    )
                    
                    state_dict = load_file(str(merged_path))
                    single_model.load_state_dict(state_dict)
                    
                    merged_stats = get_weight_stats(single_model, prefix="merged_")
                    print_weight_comparison(trained_stats, merged_stats, "Trained", "Merged")
                else:
                    logger.error(f"Merged model not found at {merged_path}")
            else:
                logger.error(f"Merge failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
        else:
            logger.warning(f"Merge script not found at {merge_script}")
    
    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()
    
    logger.info("\n" + "="*60)
    logger.info("Debug script completed!")
    logger.info("="*60)


if __name__ == "__main__":
    import os
    main()