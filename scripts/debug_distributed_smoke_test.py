#!/usr/bin/env python3
"""
Comprehensive distributed smoke test for CLT model save/load/eval cycle.
This test will monitor model weights, BatchTopK state, and metrics at every step.
"""

import torch
import torch.distributed as dist
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig  # noqa: E402
from clt.models.clt import CrossLayerTranscoder  # noqa: E402
from clt.training.trainer import CLTTrainer  # noqa: E402
from clt.training.checkpointing import CheckpointManager  # noqa: E402
from clt.training.evaluator import CLTEvaluator  # noqa: E402
from clt.training.wandb_logger import DummyWandBLogger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_weight_stats(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, float]:
    """Compute summary statistics for model weights."""
    stats: Dict[str, float] = {}

    for name, param in model.named_parameters():
        if param is None:
            stats[f"{prefix}{name}_is_none"] = 1.0
            continue

        param_cpu = param.detach().cpu().float()
        stats[f"{prefix}{name}_mean"] = param_cpu.mean().item()
        stats[f"{prefix}{name}_std"] = param_cpu.std().item()
        stats[f"{prefix}{name}_min"] = param_cpu.min().item()
        stats[f"{prefix}{name}_max"] = param_cpu.max().item()
        stats[f"{prefix}{name}_norm"] = param_cpu.norm().item()

        # Check for NaN/Inf
        stats[f"{prefix}{name}_has_nan"] = float(torch.isnan(param_cpu).any().item())
        stats[f"{prefix}{name}_has_inf"] = float(torch.isinf(param_cpu).any().item())

    return stats


def check_batchtopk_state(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, float]:
    """Check BatchTopK-specific state (theta values)."""
    stats = {}

    # Check if model has theta values
    if hasattr(model, "theta_manager") and model.theta_manager is not None:
        if hasattr(model.theta_manager, "log_threshold") and model.theta_manager.log_threshold is not None:
            log_theta = model.theta_manager.log_threshold.detach().cpu()
            theta = log_theta.exp()

            stats[f"{prefix}log_theta_shape"] = float(log_theta.numel())
            stats[f"{prefix}log_theta_mean"] = log_theta.mean().item()
            stats[f"{prefix}log_theta_std"] = log_theta.std().item()
            stats[f"{prefix}theta_mean"] = theta.mean().item()
            stats[f"{prefix}theta_std"] = theta.std().item()
            stats[f"{prefix}theta_min"] = theta.min().item()
            stats[f"{prefix}theta_max"] = theta.max().item()
        else:
            stats[f"{prefix}log_threshold_exists"] = 0.0
    else:
        stats[f"{prefix}theta_manager_exists"] = 0.0

    return stats


def evaluate_model(
    model: CrossLayerTranscoder, activation_store, device: torch.device, prefix: str = "", num_batches: int = 5
) -> Dict[str, float]:
    """Evaluate model on a few batches and return metrics."""
    evaluator = CLTEvaluator(model, device)

    total_metrics = {"total_loss": 0.0, "nmse": 0.0, "explained_variance": 0.0, "avg_l0": 0.0, "num_batches": 0}

    try:
        for i in range(num_batches):
            inputs, targets = next(activation_store)

            # Check input stats
            if i == 0:
                for layer_idx, inp in inputs.items():
                    total_metrics[f"input_layer{layer_idx}_mean"] = inp.float().mean().item()
                    total_metrics[f"input_layer{layer_idx}_std"] = inp.float().std().item()

            # Get metrics
            metrics = evaluator.compute_metrics(inputs, targets)

            # Aggregate key metrics
            total_metrics["nmse"] += metrics.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
            total_metrics["explained_variance"] += metrics.get("reconstruction/explained_variance", 0.0)
            total_metrics["avg_l0"] += metrics.get("sparsity/avg_l0", 0.0)
            total_metrics["num_batches"] += 1

    except StopIteration:
        logger.warning(f"Only got {total_metrics['num_batches']} batches")

    # Average the metrics
    if total_metrics["num_batches"] > 0:
        for key in ["nmse", "explained_variance", "avg_l0"]:
            total_metrics[key] /= total_metrics["num_batches"]

    # Add prefix
    return {f"{prefix}{k}": v for k, v in total_metrics.items()}


def run_smoke_test(rank: int, world_size: int, args):
    """Main smoke test logic."""
    # Initialize distributed
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create configs
    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=args.num_layers,
        d_model=args.d_model,
        activation_fn=args.activation_fn,
        batchtopk_k=args.batchtopk_k if args.activation_fn == "batchtopk" else None,
        clt_dtype=args.precision,
    )

    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=100,  # Short for smoke test
        train_batch_size_tokens=args.batch_size,
        activation_source="local_manifest",
        activation_path=args.activation_path,
        activation_dtype=args.activation_dtype,
        normalization_method="auto",
        precision=args.precision,
        seed=42,
        eval_interval=50,
        checkpoint_interval=50,
    )

    log_dir = f"smoke_test_logs/distributed_smoke_{int(time.time())}"

    # Results dictionary
    results: Dict[str, Any] = {"rank": rank, "world_size": world_size, "test_stages": {}}

    try:
        # Stage 1: Create fresh model
        logger.info(f"Rank {rank}: Creating fresh model...")
        model_fresh = CrossLayerTranscoder(
            clt_config, process_group=dist.group.WORLD if world_size > 1 else None, device=device
        )

        stage1_results = {
            **compute_weight_stats(model_fresh, "fresh_"),
            **check_batchtopk_state(model_fresh, "fresh_"),
        }
        results["test_stages"]["1_fresh_model"] = stage1_results

        # Stage 2: Initialize trainer and run a few steps
        logger.info(f"Rank {rank}: Initializing trainer...")
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=log_dir,
            device=device,
            distributed=(world_size > 1),
        )

        # Get initial evaluation metrics
        activation_store = trainer.activation_store
        stage2_results = evaluate_model(trainer.model, activation_store, device, "initial_")
        results["test_stages"]["2_initial_eval"] = stage2_results

        # Stage 3: Train for a few steps
        logger.info(f"Rank {rank}: Training for a few steps...")
        trainer.train(eval_every=50)

        stage3_results = {
            **compute_weight_stats(trainer.model, "trained_"),
            **check_batchtopk_state(trainer.model, "trained_"),
            **evaluate_model(trainer.model, activation_store, device, "trained_"),
        }
        results["test_stages"]["3_after_training"] = stage3_results

        # Stage 4: Save checkpoint
        logger.info(f"Rank {rank}: Saving checkpoint...")
        checkpoint_path = os.path.join(log_dir, "test_checkpoint")
        trainer_state = {
            "step": 100,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "wandb_run_id": None,
        }
        trainer.checkpoint_manager._save_checkpoint(100, trainer_state)

        # Stage 5: Load checkpoint into new model
        logger.info(f"Rank {rank}: Loading checkpoint...")
        model_loaded = CrossLayerTranscoder(
            clt_config, process_group=dist.group.WORLD if world_size > 1 else None, device=device
        )

        # Create new checkpoint manager for loading
        checkpoint_manager = CheckpointManager(
            model=model_loaded,
            activation_store=activation_store,
            wandb_logger=DummyWandBLogger(training_config, clt_config, log_dir, None),
            log_dir=log_dir,
            distributed=(world_size > 1),
            rank=rank,
            device=device,
            world_size=world_size,
        )

        # Load the checkpoint
        if world_size > 1:
            loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        else:
            loaded_state = checkpoint_manager.load_checkpoint(
                os.path.join(checkpoint_path, "clt_checkpoint_100.safetensors")
            )

        stage4_results = {
            "loaded_state_keys": list(loaded_state.keys()) if loaded_state else [],
            "loaded_step": loaded_state.get("step", -1) if loaded_state else -1,
        }
        results["test_stages"]["4_checkpoint_loaded"] = stage4_results

        stage5_results = {
            **compute_weight_stats(model_loaded, "loaded_"),
            **check_batchtopk_state(model_loaded, "loaded_"),
            **evaluate_model(model_loaded, activation_store, device, "loaded_"),
        }
        results["test_stages"]["5_loaded_model"] = stage5_results

        # Stage 6: Compare weights
        logger.info(f"Rank {rank}: Comparing weights...")
        weight_diffs: Dict[str, float] = {}
        for (name1, param1), (name2, param2) in zip(trainer.model.named_parameters(), model_loaded.named_parameters()):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            if param1 is not None and param2 is not None:
                diff_tensor = (param1 - param2).abs()
                max_diff = diff_tensor.max().item()
                weight_diffs[f"max_diff_{name1}"] = max_diff
                weight_diffs[f"relative_diff_{name1}"] = max_diff / (param1.abs().max().item() + 1e-8)

        results["test_stages"]["6_weight_comparison"] = weight_diffs

        # Stage 7: Test single forward pass with same data
        logger.info(f"Rank {rank}: Testing forward pass consistency...")
        test_inputs, test_targets = next(iter(activation_store))

        with torch.no_grad():
            # Get activations from both models
            acts_trained = trainer.model.get_feature_activations(test_inputs)
            acts_loaded = model_loaded.get_feature_activations(test_inputs)

            # Compare activations
            act_diffs: Dict[str, float] = {}
            for layer_idx in acts_trained:
                if layer_idx in acts_loaded:
                    diff = (acts_trained[layer_idx] - acts_loaded[layer_idx]).abs()
                    act_diffs[f"layer_{layer_idx}_max_diff"] = diff.max().item()
                    act_diffs[f"layer_{layer_idx}_mean_diff"] = diff.mean().item()
                    act_diffs[f"layer_{layer_idx}_num_different"] = float((diff > 1e-6).sum().item())

        results["test_stages"]["7_activation_comparison"] = act_diffs

    except Exception as e:
        logger.error(f"Rank {rank}: Error during smoke test: {e}")
        import traceback

        results["error"] = {"message": str(e), "traceback": traceback.format_exc()}

    # Save results
    if rank == 0:
        results_path = os.path.join(log_dir, "smoke_test_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Print summary
        print("\n=== SMOKE TEST SUMMARY ===")
        for stage, data in results["test_stages"].items():
            print(f"\n{stage}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if "mean" in key or "std" in key or "eval" in key:
                        print(f"  {key}: {value:.6f}")

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed CLT smoke test")
    parser.add_argument("--num-features", type=int, default=32768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--activation-fn", type=str, default="batchtopk")
    parser.add_argument("--batchtopk-k", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--activation-path", type=str, required=True)
    parser.add_argument("--activation-dtype", type=str, default="float16")
    parser.add_argument("--precision", type=str, default="fp16")

    args = parser.parse_args()

    # Check if running with torchrun
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    run_smoke_test(rank, world_size, args)


if __name__ == "__main__":
    main()
