#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) on GPT-2.
"""

import argparse
import torch
from pathlib import Path
from typing import Literal, Optional
import logging

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a CLT on GPT-2")

    # --- CLT Configuration ---
    parser.add_argument(
        "--num-features", type=int, default=300, help="Number of features per layer"
    )
    parser.add_argument(
        "--activation-fn",
        type=str,
        choices=["jumprelu", "relu"],
        default="jumprelu",
        help="Activation function for CLT",
    )
    parser.add_argument(
        "--jumprelu-threshold",
        type=float,
        default=0.03,
        help="Threshold for JumpReLU activation",
    )
    parser.add_argument(
        "--clt-dtype",
        type=str,
        default=None,
        help="Data type for the CLT model (e.g., float16)",
    )

    # --- Base Model Configuration ---
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Base model to extract activations from",
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        default=None,
        help="Data type for the base model (e.g., bfloat16, float16)",
    )

    # --- Dataset Configuration ---
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path or name of the HuggingFace dataset",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--dataset-text-column",
        type=str,
        default="text",
        help="Name of the column containing text data",
    )
    parser.add_argument(
        "--streaming", action="store_true", default=True, help="Use streaming dataset"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Do not use streaming dataset",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code for dataset loading",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process from dataset",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache extracted activations",
    )

    # --- Tokenization & Batching Configuration ---
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Context window size for processing",
    )
    parser.add_argument(
        "--prepend-bos", action="store_true", default=True, help="Prepend BOS token"
    )
    parser.add_argument(
        "--no-prepend-bos",
        action="store_false",
        dest="prepend_bos",
        help="Do not prepend BOS token",
    )
    parser.add_argument(
        "--exclude-special-tokens",
        action="store_true",
        default=False,
        help="Exclude special tokens during activation extraction",
    )
    parser.add_argument(
        "--store-batch-size-prompts",
        type=int,
        default=4,
        help="Number of prompts per activation extraction batch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for CLT training"
    )
    parser.add_argument(
        "--n-batches-in-buffer",
        type=int,
        default=16,
        help="Number of extraction batches to keep in the buffer",
    )

    # --- Normalization Configuration ---
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["mean_std", "estimated_mean_std", "none"],
        default="mean_std",
        help="Method for normalizing activations",
    )
    parser.add_argument(
        "--normalization-estimation-batches",
        type=int,
        default=50,
        help="Number of batches to estimate normalization stats (if method='estimated_mean_std')",
    )

    # --- Training & Optimization Configuration ---
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--training-steps", type=int, default=50000, help="Number of training steps"
    )
    parser.add_argument(
        "--sparsity-lambda",
        type=float,
        default=1e-3,
        help="Coefficient for sparsity penalty",
    )
    parser.add_argument(
        "--sparsity-c",
        type=float,
        default=1.0,
        help="Parameter for sparsity penalty shape",
    )
    parser.add_argument(
        "--preactivation-coef",
        type=float,
        default=3e-6,
        help="Coefficient for pre-activation loss",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer to use",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["linear", "cosine", "none"],
        default="linear",
        help="Learning rate scheduler type ('none' to disable)",
    )

    # --- Logging & Evaluation Configuration ---
    parser.add_argument(
        "--output-dir", type=str, default="clt_output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Log metrics every N steps"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=1000, help="Evaluate model every N steps"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--dead-feature-window",
        type=int,
        default=1000,
        help="Number of steps until a feature is considered dead",
    )

    # --- WandB Configuration ---
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        default=False,
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity name"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="WandB run name"
    )
    parser.add_argument(
        "--wandb-tags", nargs="+", default=None, help="List of WandB tags"
    )

    # --- Device Configuration ---
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'mps')",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Create configurations
    # Use the model to determine number of layers and d_model
    print(f"Creating model configuration for {args.model}...")

    # Here we're assuming a simple way to get model dimensions for GPT-2 variants
    # In a real implementation, you might want to detect this dynamically
    if args.model == "gpt2":
        num_layers = 12
        d_model = 768
    elif args.model == "gpt2-medium":
        num_layers = 24
        d_model = 1024
    elif args.model == "gpt2-large":
        num_layers = 36
        d_model = 1280
    elif args.model == "gpt2-xl":
        num_layers = 48
        d_model = 1600
    else:
        # For other models, you might need a more sophisticated approach
        # This is a simplification for the example
        print(f"Warning: Model {args.model} not recognized, using default values")
        num_layers = 12
        d_model = 768

    clt_config = CLTConfig(
        num_features=args.num_features,
        num_layers=num_layers,
        d_model=d_model,
        activation_fn=args.activation_fn,
        jumprelu_threshold=args.jumprelu_threshold,
        clt_dtype=args.clt_dtype,
    )

    # Handle 'none' scheduler case
    lr_scheduler_arg: Optional[Literal["linear", "cosine"]] = (
        args.lr_scheduler if args.lr_scheduler != "none" else None
    )

    training_config = TrainingConfig(
        # Core training parameters
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        # Model parameters
        model_name=args.model,
        model_dtype=args.model_dtype,
        # Dataset parameters
        dataset_path=args.dataset,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
        streaming=args.streaming,
        dataset_trust_remote_code=args.trust_remote_code,
        max_samples=args.max_samples,
        cache_path=args.cache_path,
        # Tokenization parameters
        context_size=args.context_size,
        prepend_bos=args.prepend_bos,
        exclude_special_tokens=args.exclude_special_tokens,
        # Batch size parameters
        batch_size=args.batch_size,
        store_batch_size_prompts=args.store_batch_size_prompts,
        n_batches_in_buffer=args.n_batches_in_buffer,
        # Normalization parameters
        normalization_method=args.normalization_method,
        normalization_estimation_batches=args.normalization_estimation_batches,
        # Loss function coefficients
        sparsity_lambda=args.sparsity_lambda,
        sparsity_c=args.sparsity_c,
        preactivation_coef=args.preactivation_coef,
        # Optimizer parameters
        optimizer=args.optimizer,
        lr_scheduler=lr_scheduler_arg,
        # Logging parameters
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        # Dead feature tracking
        dead_feature_window=args.dead_feature_window,
        # WandB logging configuration
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=str(output_dir),
        device=device,
    )

    # Train model
    print("Starting training...")
    trainer.train(eval_every=args.eval_interval)

    print("Training complete!")


if __name__ == "__main__":
    main()
