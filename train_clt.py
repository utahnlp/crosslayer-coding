#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) using the clt library.
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

from clt.config import CLTConfig, TrainingConfig, ActivationConfig
from clt.training.trainer import CLTTrainer

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(
            config, "n_layer", None
        )
        d_model = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )

        if num_layers is None or d_model is None:
            raise ValueError(
                f"Could not automatically determine num_layers or d_model for {model_name}"
            )
        logger.info(
            f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size."
        )
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            f"Falling back to gpt2 defaults (12 layers, 768 hidden size)."
        )
        return 12, 768


def parse_args():
    """Parse command-line arguments using distinct groups for clarity."""
    parser = argparse.ArgumentParser(
        description="Train a Cross-Layer Transcoder (CLT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core Training Mode ---
    mode_group = parser.add_argument_group("Core Training Mode")
    mode_group.add_argument(
        "--activation-source",
        type=str,
        choices=["generate", "local"],
        required=True,
        help=(
            "'generate': Generate activations on-the-fly during training. "
            "'local': Load pre-generated activations from disk."
        ),
    )
    mode_group.add_argument(
        "--activation-path",
        type=str,
        default=None,
        help="Path to the directory containing pre-generated activations (required if --activation-source=local).",
    )
    mode_group.add_argument(
        "--output-dir",
        type=str,
        default=f"clt_train_{int(time.time())}",
        help="Directory to save logs, checkpoints, and final model.",
    )
    mode_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps'). Auto-detected if None.",
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
    train_group.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Optimizer learning rate."
    )
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
        choices=["auto", "estimated_mean_std", "none"],
        default="auto",
        help=(
            "Normalization for activation store. 'auto' uses pre-calculated stats for 'local' source "
            "or estimates for 'generate' source. 'estimated_mean_std' forces estimation (only for 'generate'). "
            "'none' disables normalization."
        ),
    )
    train_group.add_argument(
        "--normalization-estimation-batches",
        type=int,
        default=50,
        help="Number of batches used to estimate normalization stats (if using 'estimated_mean_std').",
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
        choices=["linear", "cosine", "none"],
        default="linear",
        help="Learning rate scheduler type ('none' to disable).",
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
    log_group.add_argument(
        "--enable-wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    log_group.add_argument(
        "--wandb-project", type=str, default=None, help="WandB project name."
    )
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
    log_group.add_argument(
        "--wandb-tags", nargs="+", default=None, help="List of tags for the WandB run."
    )

    # --- Activation Generation Parameters (only used if --activation-source=generate) ---
    gen_group = parser.add_argument_group(
        "Activation Generation Parameters (used if --activation-source=generate)"
    )
    gen_group.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Base model name or path (e.g., 'gpt2', 'gpt2-medium'). Used for CLT dimensions and activation generation.",
    )
    gen_group.add_argument(
        "--mlp-input-template",
        type=str,
        default="transformer.h.{}.mlp.c_fc",  # Common for GPT2-like
        help="NNsight path template to the MLP input module (use '{}' for layer index).",
    )
    gen_group.add_argument(
        "--mlp-output-template",
        type=str,
        default="transformer.h.{}.mlp.c_proj",  # Common for GPT2-like
        help="NNsight path template to the MLP output module (use '{}' for layer index).",
    )
    gen_group.add_argument(
        "--model-dtype",
        type=str,
        default=None,
        help="Optional data type for loading the base model during generation (e.g., 'float16', 'bfloat16').",
    )
    gen_group.add_argument(
        "--dataset-path",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name or path for activation generation.",
    )
    gen_group.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use for generation.",
    )
    gen_group.add_argument(
        "--dataset-text-column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )
    gen_group.add_argument(
        "--context-size",
        type=int,
        default=128,
        help="Context window size for tokenization and inference during generation.",
    )
    gen_group.add_argument(
        "--inference-batch-size",
        type=int,
        default=512,
        help="Batch size (number of prompts) for model inference during activation generation.",
    )
    gen_group.add_argument(
        "--n-batches-in-buffer",
        type=int,
        default=16,
        help="Number of generation batches to buffer in the StreamingActivationStore.",
    )
    gen_group.add_argument(
        "--exclude-special-tokens",
        action="store_true",
        default=True,
        help="Exclude special tokens when extracting activations.",
    )
    gen_group.add_argument(
        "--no-exclude-special-tokens",
        action="store_false",
        dest="exclude_special_tokens",
    )
    gen_group.add_argument(
        "--prepend-bos",
        action="store_true",
        default=False,
        help="Prepend BOS token to sequences during generation.",
    )
    gen_group.add_argument("--no-prepend-bos", action="store_false", dest="prepend_bos")
    gen_group.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use dataset streaming during generation.",
    )
    gen_group.add_argument("--no-streaming", action="store_false", dest="streaming")
    gen_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading HuggingFace dataset.",
    )
    gen_group.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path for HuggingFace datasets cache (if not streaming).",
    )
    # nnsight args can be added here if needed, or configured via a file

    args = parser.parse_args()

    # --- Validation ---
    if args.activation_source == "local" and not args.activation_path:
        parser.error("--activation-path is required when --activation-source=local")
    if args.activation_source == "generate":
        # Check essential generation args are provided or have defaults
        required_gen_args = [
            "model_name",
            "mlp_input_template",
            "mlp_output_template",
            "dataset_path",
        ]
        for arg in required_gen_args:
            if getattr(args, arg) is None:
                parser.error(
                    f"--{arg.replace('_', '-')} is required when --activation-source=generate"
                )

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
    # Use the model_name specified in generation args, even if using local source,
    # as the CLT architecture needs to match the model activations were generated from.
    base_model_name = args.model_name
    num_layers, d_model = get_model_dimensions(base_model_name)

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
    lr_scheduler_arg: Optional[Literal["linear", "cosine"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    # Prepare activation source specific configs
    generation_config_dict: Optional[Dict[str, Any]] = None
    dataset_params_dict: Optional[Dict[str, Any]] = None
    activation_path_arg: Optional[str] = None

    if args.activation_source == "generate":
        generation_config_dict = {
            "model_name": args.model_name,
            "mlp_input_template": args.mlp_input_template,
            "mlp_output_template": args.mlp_output_template,
            "model_dtype": args.model_dtype,
            "context_size": args.context_size,
            "inference_batch_size": args.inference_batch_size,
            "exclude_special_tokens": args.exclude_special_tokens,
            "prepend_bos": args.prepend_bos,
            # Add nnsight args here if they were parsed
        }
        dataset_params_dict = {
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "dataset_text_column": args.dataset_text_column,
            "streaming": args.streaming,
            "dataset_trust_remote_code": args.trust_remote_code,
            "cache_path": args.cache_path,
            # Add max_samples here if needed for the generator call within the store
        }
    elif args.activation_source == "local":
        activation_path_arg = args.activation_path

    training_config = TrainingConfig(
        # Core Training
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        train_batch_size_tokens=args.train_batch_size_tokens,
        # Activation Source
        activation_source=args.activation_source,
        generation_config=generation_config_dict,
        dataset_params=dataset_params_dict,
        activation_path=activation_path_arg,
        # Buffer Size (only used for 'generate')
        n_batches_in_buffer=args.n_batches_in_buffer,
        # Normalization
        normalization_method=args.normalization_method,
        normalization_estimation_batches=args.normalization_estimation_batches,
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
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        # Remote config is not handled by this script
        remote_config=None,
    )
    logger.info(f"Training Config: {training_config}")

    # --- Initialize Trainer ---
    logger.info("Initializing CLTTrainer...")
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir),
            device=device,
        )
    except Exception as e:
        logger.exception(
            f"Failed to initialize CLTTrainer: {e}"
        )  # Use logger.exception
        raise

    # --- Start Training ---
    logger.info("Starting training...")
    try:
        trained_model = trainer.train()  # eval_every is handled internally now
        logger.info("Training complete!")
        logger.info(f"Final model and logs saved in: {output_dir.resolve()}")
    except Exception as e:
        logger.exception(f"Training failed: {e}")  # Use logger.exception
        raise


if __name__ == "__main__":
    main()
