#!/usr/bin/env python3
"""
Smoke test script for CLT training.
Trains a small CLT model for a few steps, using BatchTopK activation.
Sources activations from a local path for single-process runs,
and from a remote server for distributed runs.
Uses float32 precision for activations and training for consistency.
"""
import torch
from pathlib import Path
import os
import logging
from typing import Optional, Tuple, Literal, Dict, Any

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None  # Will be handled in get_model_dimensions

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)


def get_model_dimensions(model_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Attempt to dynamically get num_layers and d_model from model_name."""
    if AutoConfig is None:
        logger.warning(
            "Transformers library not found. Cannot dynamically detect model dimensions."
            " Install transformers (`pip install transformers`) for auto-detection."
            " Falling back to model-specific defaults if not otherwise specified."
        )
        if model_name == "gpt2":  # Should not be used now, but kept for robustness
            return 12, 768
        elif model_name == "EleutherAI/pythia-70m":
            return 6, 512
        return None, None

    try:
        config = AutoConfig.from_pretrained(model_name)
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
        d_model = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)

        if num_layers is None or d_model is None:
            logger.warning(
                f"Could not automatically determine num_layers or d_model for {model_name}. "
                "Will use hardcoded model-specific defaults if available."
            )
            if model_name == "gpt2":
                return 12, 768
            elif model_name == "EleutherAI/pythia-70m":
                return 6, 512
            return None, None
        logger.info(f"Detected model dimensions for {model_name}: {num_layers} layers, {d_model} hidden size.")
        return num_layers, d_model
    except Exception as e:
        logger.warning(
            f"Failed to get model dimensions for {model_name}: {e}. "
            "Will use hardcoded model-specific defaults if available."
        )
        if model_name == "gpt2":
            return 12, 768
        elif model_name == "EleutherAI/pythia-70m":
            return 6, 512
        return None, None


def main():
    """Runs the smoke test."""

    # Determine if running in a distributed environment first
    is_distributed_run = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    run_type = "distributed (remote)" if is_distributed_run else "single-process (local)"
    logger.info(f"Starting CLT smoke test script ({run_type} activation source with WandB)...")

    # --- Configuration ---
    base_model_name = "EleutherAI/pythia-70m"

    activation_source_for_config: Literal["local_manifest", "remote"]
    activation_path_for_config: Optional[str] = None
    remote_config_dict_for_config: Optional[Dict[str, Any]] = None

    # Consistent dtypes for this smoke test, matching typical server storage
    consistent_clt_dtype = "float32"
    consistent_activation_dtype_tc: Literal["bfloat16", "float16", "float32"] = "float32"
    consistent_precision_tc: Literal["fp32", "fp16", "bf16"] = "fp32"

    if is_distributed_run:
        output_dir_name = "clt_smoke_output_remote_wandb_batchtopk"
        server_url = "http://34.41.125.189:8000"
        dataset_id = "EleutherAI/pythia-70m/pile-uncopyrighted_train"
        activation_source_for_config = "remote"
        remote_config_dict_for_config = {
            "server_url": server_url,
            "dataset_id": dataset_id,
            "timeout": 120,
            "max_retries": 3,
            "prefetch_batches": 4,
        }
        logger.warning(
            f"Distributed Run: Using REMOTE server {server_url} for dataset '{dataset_id}'. Using {consistent_precision_tc} precision."
        )
        logger.warning("PREREQUISITE: Ensure the remote server is running and accessible.")
        logger.warning(
            f"PREREQUISITE: Ensure the dataset '{dataset_id}' is available on the server (likely as float32)."
        )
    else:
        output_dir_name = "clt_smoke_output_local_wandb_batchtopk"
        local_activation_path = "/Users/curttigges/Projects/crosslayer-coding/tutorials/tutorial_activations_local_1M_pythia/EleutherAI/pythia-70m/pile-uncopyrighted_train"
        activation_source_for_config = "local_manifest"
        activation_path_for_config = local_activation_path
        logger.info(
            f"Single-Process Run: Using LOCAL activations from {local_activation_path}. Using {consistent_precision_tc} precision."
        )
        if not Path(local_activation_path).exists():
            logger.error(f"Local activation data not found at {local_activation_path}")
            logger.error("Please ensure this path is correct and activations (for Pythia-70m, likely float32) exist.")
            logger.error("Refer to tutorial 1B for generating these activations if needed.")
            return

    logger.info("W&B Logging Enabled: Project='refactor-smoke-tests'")

    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    num_layers, d_model = get_model_dimensions(base_model_name)
    if num_layers is None or d_model is None:
        logger.error(f"Could not determine model dimensions for {base_model_name}. Exiting.")
        return

    clt_config = CLTConfig(
        num_features=d_model * 2,  # Small expansion (e.g., 512*2 = 1024)
        num_layers=num_layers,
        d_model=d_model,
        activation_fn="batchtopk",
        batchtopk_k=200,  # Using BatchTopK with k=200
        # jumprelu_threshold is not used for batchtopk
        model_name=base_model_name,
        clt_dtype=consistent_clt_dtype,
    )

    # Auto-generate WandB run name
    run_name_parts = [
        "smoke",
        base_model_name.split("/")[-1],  # e.g. pythia-70m
        f"{clt_config.num_features}f",
        clt_config.activation_fn,  # Will be batchtopk
        f"k{clt_config.batchtopk_k}" if clt_config.activation_fn == "batchtopk" else "",
        f"lr{1e-4:.0e}",
        f"b{1024}",
        "dist" if is_distributed_run else "local",
        consistent_precision_tc,  # Add precision to name
    ]
    wandb_run_name = "-".join(filter(None, run_name_parts))  # Filter out empty strings if any

    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=40,
        train_batch_size_tokens=1024,
        activation_source=activation_source_for_config,
        activation_path=activation_path_for_config,
        remote_config=remote_config_dict_for_config,
        activation_dtype=consistent_activation_dtype_tc,
        normalization_method="auto",
        sparsity_lambda=1e-4,  # For BatchTopK, ensure apply_sparsity_penalty_to_batchtopk is False if this is not desired
        apply_sparsity_penalty_to_batchtopk=False,  # Typically False if AuxK or other sparsity mechanism used with BatchTopK
        log_interval=5,
        eval_interval=10,
        checkpoint_interval=20,
        enable_wandb=True,
        wandb_project="refactor-smoke-tests",
        wandb_run_name=wandb_run_name,
        seed=42,
        precision=consistent_precision_tc,
    )

    # Determine device for CLTTrainer (used if not distributed, otherwise trainer figures it out)
    if torch.cuda.is_available():
        trainer_device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        trainer_device_str = "mps"
    else:
        trainer_device_str = "cpu"

    logger.info(f"Effective Run Type: {run_type}")
    logger.info(f"CLT Config for smoke test: {clt_config}")
    logger.info(f"Training Config for smoke test: {training_config}")
    logger.info(f"Distributed run detected by script logic: {is_distributed_run}")
    logger.info(f"Trainer device string (for single GPU/main process): {trainer_device_str}")

    # --- Trainer Initialization ---
    try:
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(output_dir),
            device=trainer_device_str,
            distributed=is_distributed_run,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CLTTrainer: {e}")
        return

    # --- Run Training ---
    logger.info(f"Starting smoke training ({run_type}, WandB enabled, {consistent_precision_tc} precision)...")
    try:
        trainer.train()
        logger.info(f"Smoke training complete! Final model and logs saved in {output_dir.resolve()}")
    except Exception as e:
        logger.exception(f"Smoke training failed: {e}")
        return


if __name__ == "__main__":
    main()
