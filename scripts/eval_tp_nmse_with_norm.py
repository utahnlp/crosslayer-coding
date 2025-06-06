#!/usr/bin/env python3
"""
Fixed evaluation script that properly handles normalization statistics.
This version loads norm_stats.json and passes them to the evaluator.
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.evaluator import CLTEvaluator
from clt.training.data.local_activation_store import LocalActivationStore
from safetensors.torch import load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_normalization_stats(activation_path: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Load normalization statistics from norm_stats.json."""
    norm_stats_path = Path(activation_path) / "norm_stats.json"

    if not norm_stats_path.exists():
        logger.warning(f"norm_stats.json not found at {norm_stats_path}")
        return {}, {}

    logger.info(f"Loading normalization stats from {norm_stats_path}")
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    mean_tg = {}
    std_tg = {}

    # Convert the norm stats to the format expected by the evaluator
    for layer_idx in range(len(norm_stats)):
        layer_stats = norm_stats[layer_idx]
        mean_tg[layer_idx] = torch.tensor(layer_stats["mean"], dtype=torch.float32)
        std_tg[layer_idx] = torch.tensor(layer_stats["std"], dtype=torch.float32)

    logger.info(f"Loaded normalization stats for {len(mean_tg)} layers")
    return mean_tg, std_tg


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Optional[CrossLayerTranscoder]:
    """Load a CLT model from a checkpoint directory or merged safetensors file."""
    checkpoint_path = Path(checkpoint_path)

    # Check if it's a safetensors file directly
    if checkpoint_path.suffix == ".safetensors":
        model_path = checkpoint_path
        config_path = checkpoint_path.parent / "cfg.json"
    else:
        # It's a directory, look for model.safetensors
        model_path = checkpoint_path / "model.safetensors"
        config_path = checkpoint_path / "cfg.json"

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None

    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)

    # Create model
    logger.info(f"Loading consolidated model from {model_path}")
    model = CrossLayerTranscoder(config, device=device)

    # Load state dict
    state_dict = load_safetensors_file(str(model_path), device="cpu")

    # Move to correct device and dtype
    state_dict = {
        k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)
    return model


def evaluate_model(
    model: CrossLayerTranscoder,
    activation_path: str,
    batch_size: int,
    device: torch.device,
    num_batches: int = 50,
    activation_dtype: str = "float16",
) -> Dict[str, float]:
    """Evaluate model with proper normalization handling."""
    logger.info("Initializing activation store...")

    # Initialize activation store
    activation_store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=batch_size,
        device=device,
        dtype=activation_dtype,
        rank=0,
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=True,
    )

    # Load normalization stats
    mean_tg, std_tg = load_normalization_stats(activation_path)

    # Initialize evaluator WITH normalization stats
    logger.info("Initializing evaluator with normalization stats...")
    evaluator = CLTEvaluator(
        model=model,
        device=device,
        mean_tg=mean_tg,
        std_tg=std_tg,
    )

    logger.info(f"Running evaluation on {num_batches} batches...")
    total_metrics = {"nmse": 0.0, "explained_variance": 0.0, "avg_l0": 0.0, "num_batches": 0}

    # Match training setup with autocast
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

                if i % 10 == 0:
                    logger.info(
                        f"Batch {i}: NMSE={metrics.get('reconstruction/normalized_mean_reconstruction_error', 0):.4f}, "
                        f"EV={metrics.get('reconstruction/explained_variance', 0):.4f}"
                    )

            except StopIteration:
                logger.warning(f"Only got {i} batches")
                break

    # Average the metrics
    if total_metrics["num_batches"] > 0:
        for key in ["nmse", "explained_variance", "avg_l0"]:
            total_metrics[key] /= total_metrics["num_batches"]

    return total_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLT model with proper normalization")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint directory or merged .safetensors file"
    )
    parser.add_argument("--activation-path", type=str, required=True, help="Path to activation dataset")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=50, help="Number of batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--activation-dtype", type=str, default="float16", choices=["float16", "float32"], help="Dtype for activations"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print("\n=== CLT Model Evaluation with Normalization ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Activation path: {args.activation_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(args.checkpoint, device)
    if model is None:
        print("ERROR: Failed to load model")
        return 1

    model.eval()
    print(f"Model loaded successfully")
    print(f"  Activation function: {model.config.activation_fn}")
    print(f"  Num features: {model.config.num_features}")
    print(f"  Num layers: {model.config.num_layers}")

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluate_model(
        model,
        args.activation_path,
        args.batch_size,
        device,
        args.num_batches,
        args.activation_dtype,
    )

    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Normalized MSE:      {metrics['nmse']:.6f}")
    print(f"Explained Variance:  {metrics['explained_variance']:.6f}")
    print(f"Average L0:          {metrics['avg_l0']:.2f}")
    print(f"Number of batches:   {metrics['num_batches']}")

    # Sanity check
    if metrics["nmse"] > 2.0:
        print("\nWARNING: NMSE is very high! Check if:")
        print("  1. The model was properly merged from distributed checkpoints")
        print("  2. The activation dataset matches the training data")
        print("  3. The normalization stats are correct")

    return 0


if __name__ == "__main__":
    sys.exit(main())
