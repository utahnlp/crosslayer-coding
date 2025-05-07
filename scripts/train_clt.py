#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) using activations from
either a local manifest or a remote server.
Handles configuration parsing from command-line arguments and initiates training.
"""

import argparse
import torch
from pathlib import Path
from typing import Literal, Optional, Dict, Any
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


def get_model_dimensions(model_name: str) -> tuple[Optional[int], Optional[int]]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Falling back to gpt2 defaults (12 layers, 768 hidden size) if not otherwise specified."
            " Install transformers (`pip install transformers`) for auto-detection."
        )
        return None, None  # Indicate failure to auto-detect

    try:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)

        if num_layers is None or d_model is None:
            logger.warning(
                f"Could not automatically determine num_layers or d_model for {model_name}. "
                "Will rely on defaults or error out if not sufficient."
            )
            return None, None
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            "Will rely on defaults or error out if not sufficient."
        )
        return None, None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT) from local or remote activations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Parameters ---
    core_group = parser.add_argument_group("Core Training Parameters")
    core_group.add_argument(
        "--activation-source",
        type=str,
        choices=["local_manifest", "remote"],
        required=True,
        help="Source of activations: 'local_manifest' or 'remote' server.",
    )
    core_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_{int(time.time())}",
        help="Directory to save logs, checkpoints, and final model.",
    )
    core_group.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name or path (e.g., 'gpt2', 'EleutherAI/pythia-70m'). Used for activation generation context and CLT dimension inference.",
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

    # --- Local Activation Source Parameters ---
    local_group = parser.add_argument_group(
        "Local Activation Source Parameters (if --activation-source=local_manifest)"
    )
    local_group.add_argument(
        "--activation-path",
        type=str,
        default=None,  # Required if local_manifest, checked in main
        help="Path to the directory containing pre-generated activations (e.g., .../index.bin, metadata.json).",
    )

    # --- Remote Activation Server Parameters ---
    remote_group = parser.add_argument_group("Remote Activation Server Parameters (if --activation-source=remote)")
    remote_group.add_argument(
        "--server-url",
        type=str,
        default=None,  # Required if remote, checked in main
        help="URL of the remote activation storage server (e.g., 'http://localhost:8000').",
    )
    remote_group.add_argument(
        "--dataset-id",
        type=str,
        default=None,  # Required if remote, checked in main
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
        default=16,  # Default from train_clt_remote
        help="Number of batches to prefetch from the server.",
    )

    # --- CLT Model Architecture (CLTConfig) ---
    clt_group = parser.add_argument_group("CLT Model Architecture (CLTConfig)")
    clt_group.add_argument(
        "--num-features",
        type=int,
        required=True,
        help="Number of features per layer in the CLT.",
    )
    # num_layers and d_model are derived from the base model if not explicitly set
    clt_group.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu", "batchtopk"],
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
        "--batchtopk-k",
        type=int,
        default=None,
        help="Absolute k for BatchTopK activation (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--batchtopk-frac",
        type=float,
        default=None,
        help="Fraction of features to keep for BatchTopK (if used). Only one of k or frac.",
    )
    clt_group.add_argument(
        "--disable-batchtopk-straight-through",
        action="store_true",  # If flag is present, disable is true. Default behavior is enabled.
        help="Disable straight-through estimator for BatchTopK. (BatchTopK default is True).",
    )
    clt_group.add_argument(
        "--clt-dtype",
        type=str,
        default=None,
        help="Optional data type for the CLT model parameters (e.g., 'float16', 'bfloat16').",
    )

    # --- Training Hyperparameters (TrainingConfig) ---
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
        choices=["auto", "none", "estimated_mean_std"],  # Added estimated_mean_std from TrainingConfig
        default="auto",
        help=(
            "Normalization for activation store. 'auto' expects server/local store to provide stats. "
            "'estimated_mean_std' forces estimation (if store supports it). 'none' disables."
        ),
    )
    train_group.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for the L1 sparsity penalty.",
    )
    train_group.add_argument(
        "--sparsity-lambda-schedule",
        type=str,
        choices=["linear", "delayed_linear"],
        default="linear",
        help="Schedule for applying sparsity lambda.",
    )
    train_group.add_argument(
        "--sparsity-lambda-delay-frac",
        type=float,
        default=0.1,
        help="Fraction of steps to delay lambda increase for 'delayed_linear' schedule.",
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
        "--optimizer-beta1",
        type=float,
        default=None,
        help="Optimizer beta1 value (if using Adam/AdamW).",
    )
    train_group.add_argument(
        "--optimizer-beta2",
        type=float,
        default=None,
        help="Optimizer beta2 value (if using Adam/AdamW).",
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
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Value for gradient clipping. If None, no clipping.",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_group.add_argument(
        "--activation-dtype",
        type=str,
        default="float32",  # Consistent default
        help="Data type to process/load activations as (e.g., 'float32', 'bfloat16').",
    )
    train_group.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps of inactivity before a feature is considered 'dead' for evaluation.",
    )
    train_group.add_argument(
        "--compute-sparsity-diagnostics",
        action="store_true",
        help="Enable computation of detailed sparsity diagnostics during evaluation.",
    )

    # --- Sampling Strategy ---
    sampling_group = parser.add_argument_group("Sampling Strategy (TrainingConfig)")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["sequential", "random_chunk"],
        default="sequential",
        help="Sampling strategy for manifest-based stores: 'sequential' or 'random_chunk'.",
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
        help="Custom name for the WandB run. Auto-generated if None.",
    )
    log_group.add_argument("--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run.")

    args = parser.parse_args()

    # --- Validate conditional arguments ---
    if args.activation_source == "remote":
        if not args.server_url:
            parser.error("--server-url is required when --activation-source is 'remote'")
        if not args.dataset_id:
            parser.error("--dataset-id is required when --activation-source is 'remote'")
    elif args.activation_source == "local_manifest":
        if not args.activation_path:
            parser.error("--activation-path is required when --activation-source is 'local_manifest'")

    if args.activation_fn == "batchtopk":
        if (args.batchtopk_k is None and args.batchtopk_frac is None) or (
            args.batchtopk_k is not None and args.batchtopk_frac is not None
        ):
            parser.error("For BatchTopK, exactly one of --batchtopk-k or --batchtopk-frac must be specified.")

    return args


def main():
    """Main function to configure and run the CLTTrainer."""
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
        device_str = args.device
    else:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon
            device_str = "mps"
        else:
            device_str = "cpu"
    logger.info(f"Using device: {device_str}")
    # Trainer will handle torch.device object creation

    # --- Determine Base Model Dimensions ---
    base_model_name = args.model_name
    num_layers_auto, d_model_auto = get_model_dimensions(base_model_name)
    if num_layers_auto is None or d_model_auto is None:
        # This case implies get_model_dimensions failed or returned Nones.
        # CLTConfig requires num_layers and d_model.
        # If they couldn't be auto-detected, it's a fatal error.
        logger.error(
            f"Could not determine dimensions (num_layers, d_model) for model '{base_model_name}'. "
            "These are required for CLTConfig. Please ensure the model name is correct and visible "
            "to the Hugging Face AutoConfig, or that the CLT library can derive them."
        )
        return  # Exit if dimensions are critical and not found

    # --- Create CLT Configuration ---
    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=num_layers_auto,  # d_model and num_layers are now from auto-detection
        d_model=d_model_auto,
        model_name=base_model_name,  # Store for reference
        activation_fn=args.activation_fn,
        jumprelu_threshold=args.jumprelu_threshold,
        batchtopk_k=args.batchtopk_k,
        batchtopk_frac=args.batchtopk_frac,
        batchtopk_straight_through=(not args.disable_batchtopk_straight_through),
        clt_dtype=args.clt_dtype,
    )
    logger.info(f"CLT Config: {clt_config}")

    # --- Create Training Configuration ---
    lr_scheduler_arg: Optional[Literal["linear", "cosine", "linear_final20"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    activation_path_arg: Optional[str] = None
    remote_config_dict: Optional[Dict[str, Any]] = None

    if args.activation_source == "local_manifest":
        activation_path_arg = args.activation_path
        logger.info(f"Using local activation source: {activation_path_arg}")
    elif args.activation_source == "remote":
        remote_config_dict = {
            "server_url": args.server_url,
            "dataset_id": args.dataset_id,
            "timeout": args.remote_timeout,
            "max_retries": args.remote_max_retries,
            "prefetch_batches": args.remote_prefetch_batches,
        }
        logger.info(f"Using remote activation source: {args.server_url}, dataset: {args.dataset_id}")

    # --- Determine WandB Run Name ---
    wandb_run_name = args.wandb_run_name
    if not wandb_run_name and args.enable_wandb:  # Auto-generate if not provided and wandb is enabled
        name_parts = [f"{args.num_features}-width"]
        if args.activation_fn == "batchtopk":
            name_parts.append("batchtopk")
            if args.batchtopk_k is not None:
                name_parts.append(f"k{args.batchtopk_k}")
            elif args.batchtopk_frac is not None:
                name_parts.append(f"kfrac{args.batchtopk_frac:.3f}")  # Format frac to 3 decimal places
        else:  # jumprelu or relu
            name_parts.append(args.activation_fn)
            name_parts.append(f"{args.sparsity_lambda:.1e}-slambda")
            name_parts.append(f"{args.sparsity_c:.1f}-sc")

        name_parts.append(f"{args.train_batch_size_tokens}-batch")
        name_parts.append(f"{args.learning_rate:.1e}-lr")
        if args.activation_source == "remote" and args.dataset_id:
            # Sanitize dataset_id for use in filename/run name
            sanitized_dataset_id = args.dataset_id.replace("/", "_")
            name_parts.append(f"ds_{sanitized_dataset_id[:20]}")  # Truncate if too long
        elif args.activation_source == "local_manifest" and args.activation_path:
            path_basename = Path(args.activation_path).name
            name_parts.append(f"path_{path_basename[:20]}")

        wandb_run_name = "-".join(name_parts)
        logger.info(f"Generated WandB run name: {wandb_run_name}")

    training_config = TrainingConfig(
        # Core Training
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        seed=args.seed,
        gradient_clip_val=args.gradient_clip_val,
        train_batch_size_tokens=args.train_batch_size_tokens,
        # Activation Source
        activation_source=args.activation_source,  # Directly from args
        activation_path=activation_path_arg,  # Populated if local
        remote_config=remote_config_dict,  # Populated if remote
        activation_dtype=args.activation_dtype,
        # Normalization
        normalization_method=args.normalization_method,
        # Sampling Strategy
        sampling_strategy=args.sampling_strategy,
        # Loss Coeffs
        sparsity_lambda=args.sparsity_lambda,
        sparsity_lambda_schedule=args.sparsity_lambda_schedule,
        sparsity_lambda_delay_frac=args.sparsity_lambda_delay_frac,
        sparsity_c=args.sparsity_c,
        preactivation_coef=args.preactivation_coef,
        # Optimizer & Scheduler
        optimizer=args.optimizer,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        lr_scheduler=lr_scheduler_arg,
        # Logging & Checkpointing
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        # Dead Features & Diagnostics
        dead_feature_window=args.dead_feature_window,
        compute_sparsity_diagnostics=args.compute_sparsity_diagnostics,
        # WandB
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
    logger.info(f"Training Config: {training_config}")

    # --- Initialize Trainer ---
    logger.info(f"Initializing CLTTrainer for {args.activation_source} training...")
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir),
            device=device_str,  # Pass the string, trainer handles torch.device
            distributed=args.distributed,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CLTTrainer: {e}")
        raise

    # --- Start Training ---
    if args.activation_source == "remote":
        logger.info(f"Starting training from remote server {args.server_url} using dataset {args.dataset_id}...")
    else:  # local_manifest
        logger.info(f"Starting training from local activations at {args.activation_path}...")

    try:
        trainer.train()  # eval_every is handled by eval_interval in TrainingConfig
        logger.info("Training complete!")
        logger.info(f"Final model and logs saved in: {output_dir.resolve()}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
