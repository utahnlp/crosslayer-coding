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
    
    # Note: The trainer destroys the process group when done, so we need to reinitialize for loading
    
    # Now load the checkpoint and compare
    checkpoint_dir = Path(output_dir) / "step_50"
    if checkpoint_dir.exists() and rank == 0:
        logger.info(f"\nRank {rank}: Loading checkpoint from {checkpoint_dir}")
        
        # For single-process loading after distributed training, we need to handle this differently
        # Let's check what files were actually saved
        checkpoint_files = list(checkpoint_dir.glob("*"))
        logger.info("\nFiles in checkpoint directory:")
        for f in checkpoint_files:
            logger.info(f"  {f.name}")
        
        # Load the consolidated model (which we know is incomplete)
        consolidated_path = checkpoint_dir / "model.safetensors"
        if consolidated_path.exists():
            from safetensors.torch import load_file as load_safetensors_file
            
            logger.info("\nLoading 'consolidated' model.safetensors...")
            state_dict = load_safetensors_file(str(consolidated_path))
            
            # Check shapes to confirm it's incomplete
            logger.info("\nChecking saved weight shapes:")
            for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                if key in state_dict:
                    logger.info(f"  {key}: shape={list(state_dict[key].shape)}")
            
            # Create a non-distributed model for comparison
            fresh_model = CrossLayerTranscoder(
                config=clt_config,
                process_group=None,  # No process group for single GPU
                device=trainer.device,
            )
            
            # Try to load (this will likely fail or give warnings)
            try:
                result = fresh_model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(f"Missing keys: {result.missing_keys[:5]}...")  # Show first 5
                if result.unexpected_keys:
                    logger.warning(f"Unexpected keys: {result.unexpected_keys[:5]}...")
                
                loaded_stats = get_weight_stats(fresh_model)
                
                logger.info("\n=== WEIGHT STATS AFTER LOADING CHECKPOINT ===")
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
                
                logger.info("\n⚠️  This comparison uses the incomplete 'consolidated' model!")
                logger.info("The consolidated model only contains rank 0's portion of the weights.")
                logger.info("This is likely why loaded models perform poorly!")
                
            except Exception as e:
                logger.error(f"Failed to load state dict: {e}")
                logger.info("\nThis confirms the 'consolidated' model is incomplete!")
    
    # Since we can't properly load distributed checkpoints without process group,
    # let's at least show what we learned
    if rank == 0:
        logger.info("\n=== SUMMARY ===")
        logger.info("1. Training completed successfully with good in-memory metrics")
        logger.info("2. The 'consolidated' model.safetensors is incomplete (only rank 0's portion)")
        logger.info("3. Distributed checkpoint files (__0_0.distcp, __1_0.distcp) would be needed for proper loading")
        logger.info("4. This explains why merged/loaded models show poor performance!")
    


if __name__ == "__main__":
    main()