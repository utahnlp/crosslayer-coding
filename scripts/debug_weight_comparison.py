#!/usr/bin/env python3
"""
Simple test to compare model weights during training vs after save/load.
Based on smoke_train.py but focused on the weight comparison issue.
"""

import torch
import torch.distributed as dist
import os
import sys
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.trainer import CLTTrainer
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_weight_stats(model):
    """Get simple statistics about model weights."""
    stats = {}
    for name, param in model.named_parameters():
        if param is not None:
            stats[name] = {
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "shape": list(param.shape),
            }
    return stats


def main():
    # Check if running distributed - follow smoke_train.py pattern
    is_distributed_run = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    if not is_distributed_run:
        logger.error("This script must be run distributed. Use: torchrun --nproc_per_node=2 scripts/debug_weight_comparison.py")
        return
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Don't manually set device - let trainer handle it like smoke_train.py
    device_str = "cuda"
    
    # Simple config matching your actual setup
    clt_config = CLTConfig(
        d_model=768,
        num_features=8192,  # Reduced size for faster testing
        num_layers=12,
        activation_fn="batchtopk",
        batchtopk_k=200,
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=100,  # Just enough to train a bit
        train_batch_size_tokens=1024,
        checkpoint_interval=50,  # Save at step 50
        eval_interval=50,
        log_interval=10,
        enable_wandb=False,
        precision="fp16",
        optimizer="adamw",
        optimizer_beta2=0.98,
        lr_scheduler="constant",
        aux_loss_factor=0.03125,
        sparsity_lambda=0.0,
        activation_source="local_manifest",
        activation_path="./activations_local_100M/gpt2/pile-uncopyrighted_train",
        activation_dtype="float16",
        normalization_method="auto",
        sampling_strategy="sequential",
        seed=42,
    )
    
    # Initialize trainer - follow smoke_train.py pattern
    output_dir = f"/tmp/debug_weight_test"  # Single dir for all ranks
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=output_dir,
        device=device_str,
        distributed=is_distributed_run,
    )
    
    if rank == 0:
        logger.info("\n=== WEIGHT STATS BEFORE TRAINING ===")
        initial_stats = get_weight_stats(trainer.model)
        # Just show a few key weights
        for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
            if key in initial_stats:
                logger.info(f"{key}: mean={initial_stats[key]['mean']:.6f}, std={initial_stats[key]['std']:.6f}, shape={initial_stats[key]['shape']}")
    
    # Train
    logger.info(f"Rank {rank}: Starting training...")
    trainer.train()
    
    # Get in-memory stats after training
    if rank == 0:
        logger.info("\n=== WEIGHT STATS AFTER TRAINING (IN MEMORY) ===")
        trained_stats = get_weight_stats(trainer.model)
        for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
            if key in trained_stats:
                logger.info(f"{key}: mean={trained_stats[key]['mean']:.6f}, std={trained_stats[key]['std']:.6f}, shape={trained_stats[key]['shape']}")
    
    # The trainer will log metrics during training
    # We'll check them from the logs/output
    
    # Wait for checkpoint to be saved
    dist.barrier()
    
    # Now load the checkpoint and compare
    checkpoint_dir = Path(output_dir) / "step_50"
    if checkpoint_dir.exists():
        logger.info(f"\nRank {rank}: Loading checkpoint from {checkpoint_dir}")
        
        # Create fresh model
        fresh_model = CrossLayerTranscoder(
            config=clt_config,
            process_group=dist.group.WORLD,
            device=trainer.device,  # Use trainer's device
        )
        
        # Load distributed checkpoint
        tp_state_dict = fresh_model.state_dict()
        load_state_dict(
            state_dict=tp_state_dict,
            storage_reader=FileSystemReader(str(checkpoint_dir)),
            planner=DefaultLoadPlanner(),
            no_dist=False,
        )
        fresh_model.load_state_dict(tp_state_dict)
        
        if rank == 0:
            logger.info("\n=== WEIGHT STATS AFTER LOADING CHECKPOINT ===")
            loaded_stats = get_weight_stats(fresh_model)
            for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                if key in loaded_stats:
                    logger.info(f"{key}: mean={loaded_stats[key]['mean']:.6f}, std={loaded_stats[key]['std']:.6f}, shape={loaded_stats[key]['shape']}")
            
            # Compare with in-memory weights
            logger.info("\n=== WEIGHT COMPARISON (IN-MEMORY vs LOADED) ===")
            for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                if key in trained_stats and key in loaded_stats:
                    mean_diff = abs(trained_stats[key]['mean'] - loaded_stats[key]['mean'])
                    std_diff = abs(trained_stats[key]['std'] - loaded_stats[key]['std'])
                    logger.info(f"{key}: mean_diff={mean_diff:.6e}, std_diff={std_diff:.6e}")
        
        # Test evaluation on all ranks
        logger.info(f"\nRank {rank}: Testing loaded model evaluation...")
        from clt.training.evaluator import CLTEvaluator
        
        # Create evaluator with same normalization stats as trainer
        evaluator = CLTEvaluator(
            model=fresh_model,
            device=trainer.device,
            mean_tg=trainer.evaluator.mean_tg,
            std_tg=trainer.evaluator.std_tg
        )
        
        # Get a fresh batch for evaluation
        try:
            # Reset the iterator to get a fresh batch
            trainer.activation_store.reset()
            eval_inputs, eval_targets = next(iter(trainer.activation_store))
            
            # Evaluate loaded model
            loaded_metrics = evaluator.compute_metrics(eval_inputs, eval_targets)
            
            if rank == 0:
                logger.info("\n=== LOADED MODEL EVALUATION ===")
                logger.info(f"NMSE: {loaded_metrics.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}")
                logger.info(f"EV: {loaded_metrics.get('reconstruction/explained_variance', -1):.4f}")
                
                # Also check a few layer-wise L0 values
                l0_dict = loaded_metrics.get('layerwise/l0', {})
                if l0_dict:
                    logger.info("Layer-wise L0 (first 3 layers):")
                    for i in range(min(3, len(l0_dict))):
                        logger.info(f"  layer_{i}: {l0_dict.get(f'layer_{i}', 0):.2f}")
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to evaluate loaded model: {e}")
    
    # Clean up
    dist.destroy_process_group()
    
    if rank == 0:
        logger.info(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()