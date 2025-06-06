#!/usr/bin/env python3
"""
Comprehensive debugging of the full tensor-parallel CLT training/evaluation cycle.
This script trains a small model, saves checkpoints, merges them, and evaluates at each stage.
"""

import torch
import torch.distributed as dist
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse
import logging
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.trainer import CLTTrainer
from clt.training.checkpointing import CheckpointManager
from clt.training.evaluator import CLTEvaluator
from clt.training.data.local_activation_store import LocalActivationStore
from safetensors.torch import save_file as save_safetensors_file, load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_model_stats(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, float]:
    """Compute summary statistics for model weights."""
    stats = {}
    
    for name, param in model.named_parameters():
        if param is None:
            continue
        
        param_cpu = param.detach().cpu().float()
        stats[f"{prefix}{name}_mean"] = param_cpu.mean().item()
        stats[f"{prefix}{name}_std"] = param_cpu.std().item()
        stats[f"{prefix}{name}_abs_max"] = param_cpu.abs().max().item()
    
    return stats


def evaluate_model_with_normalization(
    model: CrossLayerTranscoder,
    activation_store: Any,
    device: torch.device,
    num_batches: int = 5
) -> Dict[str, float]:
    """Evaluate model using proper normalization from the activation store."""
    
    # Extract normalization stats from the activation store
    mean_tg = {}
    std_tg = {}
    
    if hasattr(activation_store, 'mean_tg') and hasattr(activation_store, 'std_tg'):
        # Copy normalization stats from activation store
        for layer_idx in range(model.config.num_layers):
            if layer_idx in activation_store.mean_tg:
                mean_tg[layer_idx] = activation_store.mean_tg[layer_idx].to(device)
            if layer_idx in activation_store.std_tg:
                std_tg[layer_idx] = activation_store.std_tg[layer_idx].to(device)
    
    logger.info(f"Evaluating with normalization stats for {len(mean_tg)} layers")
    
    # Initialize evaluator WITH normalization stats
    evaluator = CLTEvaluator(
        model=model,
        device=device,
        mean_tg=mean_tg,
        std_tg=std_tg,
    )
    
    model.eval()
    total_metrics = {
        "nmse": 0.0,
        "explained_variance": 0.0,
        "avg_l0": 0.0,
        "num_batches": 0
    }
    
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        with torch.no_grad():
            for i in range(num_batches):
                try:
                    inputs, targets = next(activation_store)
                    metrics = evaluator.compute_metrics(inputs, targets)
                    
                    total_metrics["nmse"] += metrics.get(
                        "reconstruction/normalized_mean_reconstruction_error", float("nan")
                    )
                    total_metrics["explained_variance"] += metrics.get(
                        "reconstruction/explained_variance", 0.0
                    )
                    total_metrics["avg_l0"] += metrics.get("sparsity/avg_l0", 0.0)
                    total_metrics["num_batches"] += 1
                    
                except StopIteration:
                    break
    
    # Average the metrics
    if total_metrics["num_batches"] > 0:
        for key in ["nmse", "explained_variance", "avg_l0"]:
            total_metrics[key] /= total_metrics["num_batches"]
    
    return total_metrics


def main():
    parser = argparse.ArgumentParser(description="Debug full TP cycle")
    parser.add_argument("--activation-path", type=str, required=True, help="Path to activation dataset")
    parser.add_argument("--num-features", type=int, default=32768, help="Number of features")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--training-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--activation-fn", type=str, default="batchtopk", choices=["relu", "batchtopk", "topk"])
    parser.add_argument("--batchtopk-k", type=int, default=200, help="K value for BatchTopK")
    parser.add_argument("--output-dir", type=str, default="debug_tp_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize distributed if needed
    if args.world_size > 1:
        if not dist.is_initialized():
            logger.error("This script should be run with torchrun for distributed training")
            logger.error(f"Example: torchrun --nproc_per_node={args.world_size} {__file__} ...")
            return
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Rank {rank}/{world_size}: Starting debug cycle")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create and train a small model
    logger.info(f"Rank {rank}: Creating model...")
    
    # Load a sample to get dimensions
    temp_store = LocalActivationStore(
        dataset_path=args.activation_path,
        train_batch_size_tokens=args.batch_size,
        device=device,
        dtype="float16",
        rank=rank,
        world=world_size,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=(world_size > 1),
    )
    
    # Get dimensions from first batch
    sample_inputs, _ = next(temp_store)
    d_model = next(iter(sample_inputs.values())).shape[-1]
    num_layers = len(sample_inputs)
    
    # Create configs
    clt_config = CLTConfig(
        d_model=d_model,
        num_features=args.num_features,
        num_layers=num_layers,
        activation_fn=args.activation_fn,
        batchtopk_k=args.batchtopk_k if args.activation_fn == "batchtopk" else None,
    )
    
    training_config = TrainingConfig(
        training_steps=args.training_steps,
        train_batch_size_tokens=args.batch_size,
        eval_batch_size_tokens=args.batch_size,
        learning_rate=1e-4,
        checkpoint_interval=50,
        eval_interval=25,
        log_interval=10,
        output_dir=str(output_dir),
        enable_wandb=False,
        mixed_precision="fp16",
        optimizer="adamw",
        lr_scheduler="constant",
        aux_loss_factor=0.03125,
        sparsity_lambda=0.001,
    )
    
    # Create model
    process_group = dist.group.WORLD if world_size > 1 else None
    model = CrossLayerTranscoder(clt_config, device=device, process_group=process_group)
    
    # Record initial model stats
    initial_stats = compute_model_stats(model, "initial_")
    
    # Step 2: Train for a few steps
    logger.info(f"Rank {rank}: Training model...")
    
    trainer = CLTTrainer(
        model=model,
        clt_config=clt_config,
        training_config=training_config,
        activation_source="local_manifest",
        activation_path=args.activation_path,
        normalization_method="auto",
    )
    
    # Train and capture metrics during training
    training_metrics = []
    for step in range(args.training_steps):
        metrics = trainer.train_step()
        if step % 10 == 0:
            training_metrics.append({
                "step": step,
                "nmse": metrics.get("reconstruction/normalized_mean_reconstruction_error", float("nan")),
                "ev": metrics.get("reconstruction/explained_variance", 0.0),
                "loss": metrics.get("train/total_loss", float("nan")),
            })
            if rank == 0:
                logger.info(f"Step {step}: NMSE={training_metrics[-1]['nmse']:.4f}, "
                           f"EV={training_metrics[-1]['ev']:.4f}")
    
    # Get post-training model stats
    post_train_stats = compute_model_stats(model, "post_train_")
    
    # Step 3: Save checkpoint (distributed)
    checkpoint_dir = output_dir / "distributed_checkpoint"
    if rank == 0:
        logger.info(f"Saving distributed checkpoint to {checkpoint_dir}")
    
    trainer.checkpoint_manager.save_checkpoint(
        step=args.training_steps,
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        metrics={},
        checkpoint_dir=str(checkpoint_dir),
    )
    
    dist.barrier()
    
    # Step 4: Merge checkpoint (only on rank 0)
    if rank == 0:
        logger.info("Merging distributed checkpoint...")
        
        # Create a temporary script to run the merge
        merge_script = output_dir / "merge_temp.py"
        merge_output = output_dir / "merged_model.safetensors"
        
        merge_cmd = f"""
import sys
sys.path.insert(0, '{project_root}')
from scripts.merge_tp_checkpoint import main as merge_main
import argparse

# Mock argparse
class Args:
    ckpt_dir = '{checkpoint_dir}'
    cfg_json = '{checkpoint_dir}/cfg.json'
    output = '{merge_output}'

merge_main()
"""
        
        with open(merge_script, 'w') as f:
            f.write(merge_cmd)
        
        # Run merge with torchrun
        import subprocess
        result = subprocess.run(
            [
                "torchrun", "--standalone", f"--nproc_per_node={world_size}",
                str(merge_script)
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Merge failed: {result.stderr}")
        else:
            logger.info(f"Merge successful: {merge_output}")
    
    dist.barrier()
    
    # Step 5: Load merged model and evaluate (all ranks)
    if rank == 0 and (output_dir / "merged_model.safetensors").exists():
        logger.info("Loading merged model for evaluation...")
        
        # Create fresh model
        eval_model = CrossLayerTranscoder(clt_config, device=device, process_group=None)
        
        # Load merged state dict
        state_dict = load_safetensors_file(str(output_dir / "merged_model.safetensors"))
        eval_model.load_state_dict(state_dict)
        
        # Get loaded model stats
        loaded_stats = compute_model_stats(eval_model, "loaded_")
        
        # Create fresh activation store for evaluation
        eval_store = LocalActivationStore(
            dataset_path=args.activation_path,
            train_batch_size_tokens=args.batch_size,
            device=device,
            dtype="float16",
            rank=0,
            world=1,
            seed=42,
            sampling_strategy="sequential",
            normalization_method="auto",
            shard_data=True,
        )
        
        # Evaluate with proper normalization
        logger.info("Evaluating merged model...")
        eval_metrics = evaluate_model_with_normalization(
            eval_model, eval_store, device, num_batches=10
        )
        
        # Step 6: Compare results
        logger.info("\n=== DEBUGGING SUMMARY ===")
        
        # Compare weight stats
        logger.info("\n1. Weight Statistics Comparison:")
        logger.info("   Parameter: Initial -> Post-Train -> Loaded")
        
        # Compare a few key parameters
        key_params = ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]
        for param_name in key_params:
            if f"initial_{param_name}_mean" in initial_stats:
                logger.info(f"   {param_name}:")
                logger.info(f"     Mean: {initial_stats[f'initial_{param_name}_mean']:.6f} -> "
                           f"{post_train_stats[f'post_train_{param_name}_mean']:.6f} -> "
                           f"{loaded_stats[f'loaded_{param_name}_mean']:.6f}")
                logger.info(f"     Std:  {initial_stats[f'initial_{param_name}_std']:.6f} -> "
                           f"{post_train_stats[f'post_train_{param_name}_std']:.6f} -> "
                           f"{loaded_stats[f'loaded_{param_name}_std']:.6f}")
        
        # Compare metrics
        logger.info("\n2. Metrics Comparison:")
        if training_metrics:
            last_train = training_metrics[-1]
            logger.info(f"   Training (last): NMSE={last_train['nmse']:.4f}, EV={last_train['ev']:.4f}")
        logger.info(f"   Evaluation:      NMSE={eval_metrics['nmse']:.4f}, EV={eval_metrics['explained_variance']:.4f}")
        
        # Save all results
        results = {
            "config": {
                "num_features": args.num_features,
                "world_size": world_size,
                "activation_fn": args.activation_fn,
                "batch_size": args.batch_size,
            },
            "weight_stats": {
                "initial": initial_stats,
                "post_train": post_train_stats,
                "loaded": loaded_stats,
            },
            "metrics": {
                "training": training_metrics,
                "evaluation": eval_metrics,
            }
        }
        
        with open(output_dir / "debug_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}/debug_results.json")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()