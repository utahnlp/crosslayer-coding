"""Minimal, fast-running configs for testing."""

from typing import Literal, Optional
from clt.config import CLTConfig, TrainingConfig

ActivationFn = Literal["jumprelu", "relu", "batchtopk", "topk"]
SparsitySchedule = Literal["linear", "delayed_linear"]
ActivationSource = Literal["local_manifest", "remote"]
Precision = Literal["fp32", "fp16", "bf16"]
ActivationDtype = Literal["bfloat16", "float16", "float32"]


def create_tiny_clt_config(
    num_layers: int = 2,
    num_features: int = 8,
    d_model: int = 4,
    activation_fn: ActivationFn = "relu",
) -> CLTConfig:
    """Creates a minimal CLTConfig for fast tests."""
    return CLTConfig(
        num_layers=num_layers,
        num_features=num_features,
        d_model=d_model,
        activation_fn=activation_fn,
        # Keep other params at default for simplicity unless needed
    )


def create_tiny_training_config(
    training_steps: int = 10,
    train_batch_size_tokens: int = 16,
    learning_rate: float = 1e-4,
    sparsity_lambda: float = 0.01,
    sparsity_lambda_schedule: SparsitySchedule = "linear",
    sparsity_lambda_delay_frac: float = 0.0,
    preactivation_coef: float = 0.0,
    eval_interval: int = 1000,
    checkpoint_interval: int = 1000,
    activation_source: ActivationSource = "local_manifest",
    activation_path: Optional[str] = None,
    activation_dtype: ActivationDtype = "bfloat16",
    precision: Precision = "fp32",
    dead_feature_window: int = 1000000,  # Set very high to disable dead neuron tracking
) -> TrainingConfig:
    """Creates a minimal TrainingConfig for fast tests."""
    return TrainingConfig(
        training_steps=training_steps,
        train_batch_size_tokens=train_batch_size_tokens,
        learning_rate=learning_rate,
        sparsity_lambda=sparsity_lambda,
        sparsity_lambda_schedule=sparsity_lambda_schedule,
        sparsity_lambda_delay_frac=sparsity_lambda_delay_frac,
        preactivation_coef=preactivation_coef,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        activation_source=activation_source,
        activation_path=activation_path,
        activation_dtype=activation_dtype,
        precision=precision,
        dead_feature_window=dead_feature_window,
        # Keep other params at default
    )
