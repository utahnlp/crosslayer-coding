#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) using activations from a remote server.
Handles configuration parsing from command-line arguments and initiates training.
"""

import argparse
import torch
from pathlib import Path
from typing import Literal, Optional, Dict, Any  # Added Dict, Any
import logging
import time
import json

# Attempt to import transformers for model dimension detection
try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

# Import necessary CLT components
try:
    from clt.config import CLTConfig, TrainingConfig
    from clt.training.trainer import CLTTrainer
except ImportError as e:
    print(
        f"FATAL: ImportError: {e}. Please ensure the 'clt' library is installed or "
        "the project root is in your PYTHONPATH."
    )
    raise

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_dimensions(model_name: str) -> tuple[int, int]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Falling back to gpt2 defaults (12 layers, 768 hidden size)."
            " Install transformers (`pip install transformers`) for auto-detection."
        )
        return 12, 768  # Default to gpt2 small

    try:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)

        if num_layers is None or d_model is None:
            raise ValueError(f"Could not automatically determine num_layers or d_model for {model_name}")
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            f"Falling back to gpt2 defaults (12 layers, 768 hidden size)."
        )
        return 12, 768


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT) using activations from a remote server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Parameters ---
    core_group = parser.add_argument_group("Core Training Parameters")
    # Removed --activation-path
    core_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_remote_{int(time.time())}",  # Changed default name
        help="Directory to save logs, checkpoints, and final model.",
    )
    core_group.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'gpt2', 'gpt2-medium'). Must match the model used for activation generation on the server.",
    )
    core_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps'). Auto-detected if None.",
    )
    core_group.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (requires torchrun/appropriate launcher).",
    )

    # --- Remote Activation Server Parameters ---
    remote_group = parser.add_argument_group("Remote Activation Server Parameters")
    remote_group.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="URL of the remote activation storage server (e.g., 'http://localhost:8000').",
    )
    remote_group.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Unique identifier for the dataset on the remote server (e.g., 'gpt2/pile-10k_train').",
    )
    remote_group.add_argument(
        "--remote-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for fetching batches from the remote server.",
    )
    remote_group.add_argument(
        "--remote-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed batch fetch requests.",
    )
    remote_group.add_argument(
        "--remote-prefetch-batches",
        type=int,
        default=16,
        help="Number of batches to prefetch from the server.",
    )

    # --- CLT Model Architecture ---
    clt_group = parser.add_argument_group("CLT Model Architecture (CLTConfig)")
    clt_group.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features per layer in the CLT.",
    )
    # num_layers and d_model are derived from the base model
    clt_group.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu"],
        default="jumprelu",
        help="Activation function for the CLT.",
    )
    clt_group.add_argument(
        "--jumprelu-threshold",
        type=float,
        default=0.03,
        help="Threshold for JumpReLU activation (if used).",
    )
    clt_group.add_argument(
        "--clt-dtype",
        type=str,
        default=None,
        help="Optional data type for the CLT model parameters (e.g., 'float16', 'bfloat16').",
    )

    # --- Training Hyperparameters ---
    train_group = parser.add_argument_group("Training Hyperparameters (TrainingConfig)")
    train_group.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    train_group.add_argument(
        "--training-steps",
        type=int,
        default=50000,
        help="Total number of training steps.",
    )
    train_group.add_argument(
        "--train-batch-size-tokens",
        type=int,
        default=4096,
        help="Target number of tokens per training batch.",
    )
    train_group.add_argument(
        "--normalization-method",
        type=str,
        choices=["auto", "none"],
        default="auto",
        help=(
            "Normalization for activation store. 'auto' expects the server to provide stats. "
            "'none' disables normalization."
        ),
    )
    train_group.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for the L1 sparsity penalty.",
    )
    train_group.add_argument(
        "--sparsity-c",
        type=float,
        default=1.0,
        help="Constant shaping the sparsity penalty (typically 1.0).",
    )
    train_group.add_argument(
        "--preactivation-coef",
        type=float,
        default=3e-6,
        help="Coefficient for the pre-activation MSE loss term.",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer algorithm.",
    )
    train_group.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["linear", "cosine", "linear_final20", "none"],
        default="linear",
        help=(
            "Learning rate scheduler type. 'linear_final20' keeps LR constant until the last 20% "
            "of steps then decays linearly to 0 ('none' to disable)."
        ),
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (primarily affects local operations like weight init).",
    )
    train_group.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",
        help="Data type to process fetched activations as (e.g., 'float32', 'bfloat16').",
    )
    train_group.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps of inactivity before a feature is considered 'dead' for evaluation.",
    )

    # --- Logging & Checkpointing ---
    log_group = parser.add_argument_group("Logging & Checkpointing (TrainingConfig)")
    log_group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log training metrics every N steps.",
    )
    log_group.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Run evaluation metrics computation every N steps.",
    )
    log_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save a training checkpoint every N steps.",
    )
    # WandB arguments
    log_group.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    log_group.add_argument("--wandb-project", type=str, default=None, help="WandB project name.")
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (username or team).",
    )
    log_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom name for the WandB run (defaults to a timestamp).",
    )
    log_group.add_argument("--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run.")

    args = parser.parse_args()

    # --- Validation ---
    # Server URL and dataset ID are required by argparse

    return args


def main():
    """Main function to configure and run the CLTTrainer for remote activations."""
    args = parse_args()

    # --- Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    # Save command-line arguments
    try:
        with open(output_dir / "cli_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save command-line args: {e}")

    # --- Determine Device ---
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # --- Determine Base Model Dimensions ---
    # Use the provided --model-name to get dimensions for the CLT config.
    # This ensures the CLT matches the architecture activations were generated from.
    base_model_name = args.model_name
    num_layers, d_model = get_model_dimensions(base_model_name)
    if num_layers is None or d_model is None:
        logger.error(f"Could not determine dimensions for model '{base_model_name}'. Exiting.")
        return

    # --- Create CLT Configuration ---
    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=num_layers,
        d_model=d_model,
        activation_fn=args.activation_fn,
        jumprelu_threshold=args.jumprelu_threshold,
        clt_dtype=args.clt_dtype,
    )
    logger.info(f"CLT Config: {clt_config}")

    # --- Create Training Configuration ---
    # Handle 'none' scheduler case
    lr_scheduler_arg: Optional[Literal["linear", "cosine", "linear_final20"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    # Create remote_config dictionary
    remote_config_dict: Dict[str, Any] = {
        "server_url": args.server_url,
        "dataset_id": args.dataset_id,
        "timeout": args.remote_timeout,
        "max_retries": args.remote_max_retries,
        "prefetch_batches": args.remote_prefetch_batches,
    }

    # --- Determine WandB Run Name ---
    if args.wandb_run_name:
        wandb_run_name = args.wandb_run_name
    else:
        # Construct the name based on the specified format
        # Format: {width}-width-{batch_size}-batch-{lr}-lr-{slambda}-slambda-{sc}-sc
        wandb_run_name = (
            f"{args.num_features}-width-"
            f"{args.train_batch_size_tokens}-batch-"
            f"{args.learning_rate:.1e}-lr-"  # Added learning rate
            f"{args.sparsity_lambda:.1e}-slambda-"  # Use scientific notation for lambda
            f"{args.sparsity_c:.1f}-sc"  # Use one decimal place for c
        )
        logger.info(f"Generated WandB run name: {wandb_run_name}")

    # TrainingConfig instantiation for remote source
    training_config = TrainingConfig(
        # Core Training
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        seed=args.seed,
        train_batch_size_tokens=args.train_batch_size_tokens,
        # Activation Source (hardcoded to remote)
        activation_source="remote",
        remote_config=remote_config_dict,  # Use the populated dict
        activation_dtype=args.activation_dtype,
        # Removed activation_path
        # Normalization
        normalization_method=args.normalization_method,
        # Loss Coeffs
        sparsity_lambda=args.sparsity_lambda,
        sparsity_c=args.sparsity_c,
        preactivation_coef=args.preactivation_coef,
        # Optimizer & Scheduler
        optimizer=args.optimizer,
        lr_scheduler=lr_scheduler_arg,
        # Logging & Checkpointing
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        # Dead Features
        dead_feature_window=args.dead_feature_window,
        # WandB
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=wandb_run_name,  # Use the determined name
        wandb_tags=args.wandb_tags,
    )
    logger.info(f"Training Config: {training_config}")

    # --- Initialize Trainer ---
    logger.info("Initializing CLTTrainer for remote training...")
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir),
            device=device,
            distributed=args.distributed,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CLTTrainer: {e}")
        raise

    # --- Start Training ---
    logger.info(f"Starting training from remote server {args.server_url} using dataset {args.dataset_id}...")
    try:
        trainer.train()
        logger.info("Training complete!")
        logger.info(f"Final model and logs saved in: {output_dir.resolve()}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
