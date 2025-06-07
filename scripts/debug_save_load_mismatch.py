#!/usr/bin/env python3
"""
Debug script to track model weights during training and after save/load.
This will help identify where the corruption happens.
"""

import torch
import torch.distributed as dist
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import argparse
import logging
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.trainer import CLTTrainer
from clt.training.evaluator import CLTEvaluator
from clt.training.data.local_activation_store import LocalActivationStore
from safetensors.torch import load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_weight_stats(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, float]:
    """Get statistics for key model weights."""
    stats = {}
    
    # Check a few key weights
    key_params = [
        "encoder_module.encoders.0.weight",
        "encoder_module.encoders.0.bias_param",
        "decoder_module.decoders.0->0.weight",
    ]
    
    for param_name in key_params:
        if hasattr(model, param_name.split('.')[0]):
            try:
                # Navigate through the module hierarchy
                parts = param_name.split('.')
                param = model
                for part in parts:
                    if '->' in part:  # Handle decoder dict keys
                        param = param[part]
                    else:
                        param = getattr(param, part)
                
                if param is not None:
                    stats[f"{prefix}{param_name}_mean"] = param.data.mean().item()
                    stats[f"{prefix}{param_name}_std"] = param.data.std().item()
                    stats[f"{prefix}{param_name}_abs_max"] = param.data.abs().max().item()
            except:
                pass
    
    return stats


def run_distributed_test():
    """Run a small distributed training test and track weights."""
    
    # Initialize distributed if not already done
    if "RANK" not in os.environ:
        logger.error("This script must be run with torchrun")
        logger.error("Example: torchrun --nproc_per_node=2 scripts/debug_save_load_mismatch.py")
        return
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    logger.info(f"Rank {rank}/{world_size}: Starting test")
    
    # Create temporary directory for this test
    if rank == 0:
        temp_dir = tempfile.mkdtemp(prefix="clt_debug_")
        logger.info(f"Using temporary directory: {temp_dir}")
    else:
        temp_dir = None
    
    # Broadcast temp_dir to all ranks
    temp_dir_list = [temp_dir]
    dist.broadcast_object_list(temp_dir_list, src=0)
    temp_dir = temp_dir_list[0]
    
    # Configuration for small test model
    d_model = 64
    num_features = 128  # Small for quick testing
    num_layers = 2
    batch_size = 32
    training_steps = 20
    
    clt_config = CLTConfig(
        d_model=d_model,
        num_features=num_features,
        num_layers=num_layers,
        activation_fn="batchtopk",
        batchtopk_k=10,
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=training_steps,
        train_batch_size_tokens=batch_size,
        checkpoint_interval=10,
        eval_interval=5,
        log_interval=5,
        enable_wandb=False,
        precision="fp32",  # Use fp32 to avoid precision issues
        optimizer="adamw",
        lr_scheduler="constant",
        aux_loss_factor=0.03125,
        sparsity_lambda=0.001,
        activation_source="local_manifest",
        activation_path="./activations_local_1M/gpt2/pile-uncopyrighted_train",
        normalization_method="auto",
    )
    
    # Create model
    process_group = dist.group.WORLD if world_size > 1 else None
    model = CrossLayerTranscoder(clt_config, device=device, process_group=process_group)
    
    # Track initial weights
    initial_stats = get_weight_stats(model, "initial_")
    if rank == 0:
        logger.info(f"Initial weight stats: {json.dumps(initial_stats, indent=2)}")
    
    # Initialize trainer
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=temp_dir,
        distributed=(world_size > 1),
    )
    
    # Custom training loop to track weights
    weight_history = []
    eval_history = []
    
    for step in range(training_steps):
        # Training step
        metrics = trainer.train_step()
        
        # Track weights every 5 steps
        if step % 5 == 0:
            current_stats = get_weight_stats(trainer.model, f"step{step}_")
            weight_history.append({"step": step, "stats": current_stats})
            
            if rank == 0:
                logger.info(f"\nStep {step} weight stats:")
                for key, val in current_stats.items():
                    if "mean" in key:
                        logger.info(f"  {key}: {val:.6f}")
        
        # Evaluation
        if step % 5 == 0 and step > 0:
            # Get evaluation metrics
            eval_metrics = trainer.evaluate(num_batches=2)
            eval_history.append({"step": step, "metrics": eval_metrics})
            
            if rank == 0:
                logger.info(f"Step {step} eval metrics: NMSE={eval_metrics.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}, "
                           f"EV={eval_metrics.get('reconstruction/explained_variance', -1):.4f}")
    
    # Get final in-memory stats
    final_memory_stats = get_weight_stats(trainer.model, "final_memory_")
    
    # Save checkpoint
    checkpoint_dir = Path(temp_dir) / "final_checkpoint"
    if rank == 0:
        logger.info(f"\nSaving checkpoint to {checkpoint_dir}")
    
    trainer.checkpoint_manager.save_checkpoint(
        step=training_steps,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        metrics={},
        checkpoint_dir=str(checkpoint_dir),
    )
    
    dist.barrier()
    
    # Now merge the checkpoint (only on rank 0)
    if rank == 0:
        logger.info("\nMerging distributed checkpoint...")
        
        # Run merge script
        merge_script = f"""
import sys
sys.path.insert(0, '{project_root}')
import torch
import torch.distributed as dist
from scripts.merge_tp_checkpoint import merge_state_dict
from clt.models.clt import CrossLayerTranscoder
from clt.config import CLTConfig
from safetensors.torch import save_file as save_safetensors_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
import json

# Initialize dist for merge
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{{rank}}")

# Load config
with open("{checkpoint_dir}/cfg.json", "r") as f:
    config_dict = json.load(f)
config = CLTConfig(**config_dict)

# Create model
model = CrossLayerTranscoder(config, device=device, process_group=dist.group.WORLD)

# Load distributed checkpoint
tp_state = model.state_dict()
load_state_dict(
    state_dict=tp_state,
    storage_reader=FileSystemReader("{checkpoint_dir}"),
    planner=DefaultLoadPlanner(),
    no_dist=False,
)
model.load_state_dict(tp_state)

# Merge
if rank == 0:
    full_state = merge_state_dict(model, config.num_features, config.d_model)
    save_safetensors_file(full_state, "{checkpoint_dir}/merged_model.safetensors")
    print("Merge complete")

dist.barrier()
dist.destroy_process_group()
"""
        
        merge_script_path = Path(temp_dir) / "merge_temp.py"
        with open(merge_script_path, 'w') as f:
            f.write(merge_script)
        
        # Run merge with torchrun
        import subprocess
        result = subprocess.run(
            ["torchrun", "--standalone", f"--nproc_per_node={world_size}", str(merge_script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Merge failed: {result.stderr}")
        else:
            logger.info("Merge successful")
    
    dist.barrier()
    
    # Load merged checkpoint and compare
    if rank == 0:
        logger.info("\nLoading merged checkpoint and comparing...")
        
        merged_path = checkpoint_dir / "merged_model.safetensors"
        if merged_path.exists():
            # Create fresh model
            fresh_model = CrossLayerTranscoder(clt_config, device=device, process_group=None)
            
            # Load merged checkpoint
            state_dict = load_safetensors_file(str(merged_path))
            fresh_model.load_state_dict(state_dict)
            
            # Get loaded stats
            loaded_stats = get_weight_stats(fresh_model, "loaded_")
            
            # Compare
            logger.info("\n=== WEIGHT COMPARISON ===")
            logger.info("Parameter: In-Memory -> Loaded (Change)")
            
            for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                mem_mean_key = f"final_memory_{key}_mean"
                loaded_mean_key = f"loaded_{key}_mean"
                
                if mem_mean_key in final_memory_stats and loaded_mean_key in loaded_stats:
                    mem_val = final_memory_stats[mem_mean_key]
                    loaded_val = loaded_stats[loaded_mean_key]
                    change = (loaded_val - mem_val) / (abs(mem_val) + 1e-8) * 100
                    logger.info(f"{key}_mean: {mem_val:.6f} -> {loaded_val:.6f} ({change:+.1f}%)")
                    
                    # Also check std
                    mem_std = final_memory_stats[f"final_memory_{key}_std"]
                    loaded_std = loaded_stats[f"loaded_{key}_std"]
                    change_std = (loaded_std - mem_std) / (mem_std + 1e-8) * 100
                    logger.info(f"{key}_std: {mem_std:.6f} -> {loaded_std:.6f} ({change_std:+.1f}%)")
            
            # Test evaluation on loaded model
            logger.info("\nTesting evaluation on loaded model...")
            
            # Create evaluator and test
            activation_store = LocalActivationStore(
                dataset_path=training_config.activation_path,
                train_batch_size_tokens=batch_size,
                device=device,
                dtype="float32",
                rank=0,
                world=1,
                seed=42,
                sampling_strategy="sequential",
                normalization_method="auto",
                shard_data=True,
            )
            
            # Get normalization stats
            mean_tg = {}
            std_tg = {}
            if hasattr(activation_store, 'mean_tg') and activation_store.mean_tg:
                for layer_idx, mean_tensor in activation_store.mean_tg.items():
                    mean_tg[layer_idx] = mean_tensor.to(device)
                    std_tg[layer_idx] = activation_store.std_tg[layer_idx].to(device)
            
            evaluator = CLTEvaluator(
                model=fresh_model,
                device=device,
                mean_tg=mean_tg,
                std_tg=std_tg,
            )
            
            # Get batch and evaluate
            inputs, targets = next(activation_store)
            loaded_metrics = evaluator.compute_metrics(inputs, targets)
            
            logger.info(f"Loaded model eval: NMSE={loaded_metrics.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}, "
                       f"EV={loaded_metrics.get('reconstruction/explained_variance', -1):.4f}")
            
            # Compare with last in-memory eval
            if eval_history:
                last_eval = eval_history[-1]
                logger.info(f"Last in-memory eval: NMSE={last_eval['metrics'].get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}, "
                           f"EV={last_eval['metrics'].get('reconstruction/explained_variance', -1):.4f}")
        
        # Save results
        results = {
            "weight_history": weight_history,
            "eval_history": eval_history,
            "final_memory_stats": final_memory_stats,
            "loaded_stats": loaded_stats if rank == 0 else {},
        }
        
        with open(Path(temp_dir) / "debug_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {temp_dir}/debug_results.json")
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        logger.info(f"\nTest complete. Results in: {temp_dir}")
        logger.info("You can manually inspect the checkpoint files if needed.")


def main():
    parser = argparse.ArgumentParser(description="Debug save/load weight mismatch")
    parser.add_argument("--keep-temp", action="store_true", help="Don't delete temporary directory")
    args = parser.parse_args()
    
    run_distributed_test()


if __name__ == "__main__":
    main()