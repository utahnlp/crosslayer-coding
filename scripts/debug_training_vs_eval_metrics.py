#!/usr/bin/env python3
"""
Compare metrics from training evaluation vs standalone evaluation.
This script extracts metrics from training logs and compares them to standalone evaluation.
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig  # noqa: E402
from clt.models.clt import CrossLayerTranscoder  # noqa: E402
from clt.training.evaluator import CLTEvaluator  # noqa: E402
from clt.training.data.local_activation_store import LocalActivationStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Optional[CrossLayerTranscoder]:
    """Load model from checkpoint (supports both distributed and non-distributed formats)."""

    # Check if it's a directory (distributed checkpoint) or file
    if os.path.isdir(checkpoint_path):
        # Load config from cfg.json
        config_path = os.path.join(checkpoint_path, "cfg.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}")
            return None

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        clt_config = CLTConfig(**config_dict)

        # Try to load consolidated model first
        consolidated_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(consolidated_path):
            logger.info(f"Loading consolidated model from {consolidated_path}")
            from safetensors.torch import load_file

            model = CrossLayerTranscoder(clt_config, process_group=None, device=device)
            state_dict = load_file(consolidated_path, device=str(device))
            model.load_state_dict(state_dict)
            return model
        else:
            logger.error(f"Consolidated model not found at {consolidated_path}")
            return None
    else:
        # Single file checkpoint
        logger.error("Single file checkpoint loading not implemented yet")
        return None


def extract_training_metrics(log_dir: str, step: int) -> Optional[Dict[str, float]]:
    """Extract metrics from training logs for a specific step."""

    # Look for metrics.json file
    metrics_path = os.path.join(log_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found at {metrics_path}")
        return None

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    # Find metrics for the requested step
    eval_metrics = metrics_data.get("eval_metrics", [])
    for entry in eval_metrics:
        if entry.get("step") == step:
            return {
                "nmse": entry.get("reconstruction/normalized_mean_reconstruction_error", float("nan")),
                "explained_variance": entry.get("reconstruction/explained_variance", 0.0),
                "avg_l0": entry.get("sparsity/avg_l0", 0.0),
                "sparsity_fraction": entry.get("sparsity/sparsity_fraction", 0.0),
            }

    logger.warning(f"No metrics found for step {step}")
    return None


def evaluate_standalone(
    model: CrossLayerTranscoder, activation_path: str, batch_size: int, device: torch.device, num_batches: int = 10
) -> Dict[str, float]:
    """Run standalone evaluation on the model."""

    logger.info("Initializing activation store for evaluation...")
    activation_store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=batch_size,
        device=device,
        dtype="float16",  # Match training
        rank=0,
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=True,  # Single GPU evaluation
    )

    logger.info(f"Running evaluation on {num_batches} batches...")
    evaluator = CLTEvaluator(model, device)

    total_metrics = {"nmse": 0.0, "explained_variance": 0.0, "avg_l0": 0.0, "num_batches": 0}

    # Use autocast context matching training
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        for i in range(num_batches):
            try:
                inputs, targets = next(activation_store)
                metrics = evaluator.compute_metrics(inputs, targets)

                total_metrics["nmse"] += metrics.get(
                    "reconstruction/normalized_mean_reconstruction_error", float("nan")
                )
                total_metrics["explained_variance"] += metrics.get("reconstruction/explained_variance", 0.0)
                total_metrics["avg_l0"] += metrics.get("sparsity/avg_l0", 0.0)
                total_metrics["num_batches"] += 1

            except StopIteration:
                logger.warning(f"Only got {i} batches")
                break

    # Average the metrics
    if total_metrics["num_batches"] > 0:
        for key in ["nmse", "explained_variance", "avg_l0"]:
            total_metrics[key] /= total_metrics["num_batches"]

    return total_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare training vs evaluation metrics")
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to checkpoint directory (e.g., log_dir/step_20000)"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Training log directory containing metrics.json")
    parser.add_argument("--step", type=int, required=True, help="Training step to compare")
    parser.add_argument("--activation-path", type=str, required=True, help="Path to activation dataset")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=50, help="Number of batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("\n=== Debugging Training vs Evaluation Metrics ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Step: {args.step}")
    print(f"Batch size: {args.batch_size}")

    # Load model
    print("\n1. Loading model from checkpoint...")
    model = load_model_from_checkpoint(args.checkpoint_path, device)
    if model is None:
        print("ERROR: Failed to load model")
        return

    model.eval()
    print(f"Model loaded successfully. Activation function: {model.config.activation_fn}")

    # Get training metrics
    print("\n2. Extracting training metrics...")
    training_metrics = extract_training_metrics(args.log_dir, args.step)
    if training_metrics:
        print("Training metrics:")
        for k, v in training_metrics.items():
            print(f"  {k}: {v:.6f}")
    else:
        print("WARNING: Could not extract training metrics")

    # Run standalone evaluation
    print("\n3. Running standalone evaluation...")
    eval_metrics = evaluate_standalone(model, args.activation_path, args.batch_size, device, args.num_batches)
    print("Standalone evaluation metrics:")
    for k, v in eval_metrics.items():
        if k != "num_batches":
            print(f"  {k}: {v:.6f}")

    # Compare
    print("\n4. Comparison:")
    if training_metrics:
        print("Metric          | Training    | Evaluation  | Difference")
        print("-" * 60)
        for key in ["nmse", "explained_variance", "avg_l0"]:
            train_val = training_metrics.get(key, float("nan"))
            eval_val = eval_metrics.get(key, float("nan"))
            diff = eval_val - train_val
            print(f"{key:<15} | {train_val:11.6f} | {eval_val:11.6f} | {diff:+11.6f}")

    # Additional diagnostics
    print("\n5. Model diagnostics:")

    # Check if model has theta values (BatchTopK)
    if hasattr(model, "theta_manager") and model.theta_manager is not None:
        if hasattr(model.theta_manager, "log_threshold") and model.theta_manager.log_threshold is not None:
            log_theta = model.theta_manager.log_threshold
            print(f"  Model has theta values: shape={log_theta.shape}")
            print(f"  Theta mean: {log_theta.exp().mean().item():.4f}")
            print(f"  Theta std: {log_theta.exp().std().item():.4f}")
        else:
            print("  Model does not have theta values (expected for ReLU)")

    # Check a few weights
    print("\n  Sample weight statistics:")
    for name, param in list(model.named_parameters())[:3]:
        if param is not None:
            print(f"    {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")


if __name__ == "__main__":
    main()
