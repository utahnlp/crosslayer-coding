#!/usr/bin/env python3
"""
Script to train a Cross-Layer Transcoder (CLT) on GPT-2.
"""

import argparse
import torch
from pathlib import Path

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a CLT on GPT-2")

    # Model configuration
    parser.add_argument(
        "--num-features", type=int, default=300, help="Number of features per layer"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to extract activations from"
    )

    # Training configuration
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
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
        "--jumprelu-threshold",
        type=float,
        default=0.03,
        help="Threshold for JumpReLU activation",
    )

    # Data configuration
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to text dataset"
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
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help="Context window size for processing",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="clt_output", help="Directory to save outputs"
    )

    # Device configuration
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')"
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
        activation_fn="jumprelu",
        jumprelu_threshold=args.jumprelu_threshold,
    )

    training_config = TrainingConfig(
        # Core training parameters
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        # Model parameters
        model_name=args.model,
        # Dataset parameters
        dataset_path=args.dataset,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
        streaming=True,
        max_samples=args.max_samples,
        # Tokenization parameters
        context_size=args.context_size,
        prepend_bos=True,
        exclude_special_tokens=True,
        # Batch size parameters
        batch_size=args.batch_size,
        store_batch_size_prompts=4,
        n_batches_in_buffer=16,
        # Normalization parameters
        normalization_method="mean_std",
        # Loss function coefficients
        sparsity_lambda=args.sparsity_lambda,
        sparsity_c=1.0,
        preactivation_coef=3e-6,
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
    trainer.train(eval_every=1000)

    print("Training complete!")


if __name__ == "__main__":
    main()
