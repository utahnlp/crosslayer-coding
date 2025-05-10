from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal


@dataclass
class CLTConfig:
    """Configuration for the CrossLayerTranscoder model."""

    d_model: int
    num_layers: int
    num_features: int
    activation_fn: Literal["relu", "jumprelu", "batchtopk"] = "relu"
    clt_dtype: Optional[str] = None  # e.g. "float32", "bfloat16"

    # JumpReLU specific
    jumprelu_threshold: float = 0.01

    # BatchTopK specific
    batchtopk_k: Optional[int] = None
    batchtopk_frac: Optional[float] = None
    batchtopk_straight_through: bool = True

    # Optional: if model was trained with a specific name, store it
    model_name: Optional[str] = None
    # Optional: normalization method used for training data (if applicable)
    normalization_method: Optional[str] = None
    # Optional: expected dtype of input activations for this CLT (if applicable)
    expected_input_dtype: Optional[str] = None
    # Optional: hook templates and context size if relevant
    mlp_input_template: Optional[str] = None
    mlp_output_template: Optional[str] = None
    context_size: Optional[int] = None


@dataclass
class TrainingConfig:
    """Configuration for the CLT training process."""

    # --- Core Training Parameters ---
    training_steps: int = 10000
    learning_rate: float = 1e-4
    train_batch_size_tokens: int = 4096
    seed: int = 42
    optimizer: Literal["adam", "adamw"] = "adamw"
    optimizer_beta1: Optional[float] = None  # Default 0.9 if Adam/AdamW
    optimizer_beta2: Optional[float] = None  # Default 0.999 if Adam/AdamW
    lr_scheduler: Optional[Literal["linear", "cosine", "linear_final20"]] = None
    lr_scheduler_params: Optional[Dict[str, Any]] = None  # e.g. {"end_factor": 0.1} for linear
    gradient_clip_val: Optional[float] = 1.0

    # --- Activation Store & Data Handling ---
    activation_source: Literal["generate", "local_manifest", "remote"] = "generate"
    activation_path: Optional[str] = None  # Path to local dataset (manifest or directory)
    activation_dtype: Optional[str] = "float16"  # Dtype for activations from store, e.g. float16, bfloat16, float32
    sampling_strategy: Literal["uniform_per_token", "uniform_per_batch"] = "uniform_per_token"
    normalization_method: Literal["none", "estimated_mean_std", "loaded_mean_std"] = "estimated_mean_std"
    normalization_estimation_batches: int = 100  # Batches for estimating norm stats
    n_batches_in_buffer: int = 10  # For StreamingActivationStore

    # --- Loss Configuration ---
    reconstruction_loss_weight: float = 1.0
    sparsity_loss_type: Literal["l1_norm_std", "l1_tanh_norm_std"] = "l1_tanh_norm_std"
    sparsity_lambda: float = 0.01
    sparsity_c: float = 20.0  # Tanh coefficient for sparsity
    sparsity_warmup_steps: Optional[int] = None  # Linear ramp-up from 0, defaults to 20% of training_steps
    preactivation_loss_type: Literal["none", "relu_sum_abs", "relu_sum_sq"] = "none"
    preactivation_loss_lambda: float = 0.0
    auxiliary_loss_weight: float = 0.0  # For custom auxiliary losses

    # --- Dead Feature Handling ---
    dead_feature_window: int = 1000  # Steps of inactivity before considered dead
    dead_feature_penalty_lambda: float = 0.0  # Optional penalty for dead features
    jumprelu_default_theta_on_convert: float = 1e6  # For BatchTopK->JumpReLU, theta for never-fired features

    # --- Checkpointing & Logging ---
    log_interval: int = 100  # Log training metrics every N steps
    eval_interval: int = 1000  # Evaluate model every N steps
    checkpoint_interval: int = 1000  # Save checkpoint every N steps
    log_dir: Optional[str] = None  # Base directory for all logs and checkpoints
    diag_every_n_eval_steps: Optional[int] = None  # How often to run detailed diagnostics (every N eval steps)
    max_features_for_diag_hist: Optional[int] = None  # Max features for histogram diagnostics

    # --- WandB Logging ---
    enable_wandb: bool = False
    wandb_project: Optional[str] = "clt_training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = field(default_factory=list)

    # --- Distributed Training ---
    distributed: bool = False  # Whether to use distributed training

    # --- Activation Generation Specific (if activation_source='generate') ---
    generation_config: Optional[Dict[str, Any]] = None
    dataset_params: Optional[Dict[str, Any]] = None

    # --- Remote Activation Store Specific (if activation_source='remote') ---
    remote_config: Optional[Dict[str, Any]] = None

    # --- Activation Config (for models trained on pre-generated activations) ---
    # This might hold info like model_name, hook_templates if not using 'generate'
    activation_config: Optional[Dict[str, Any]] = None

    # --- Sparsity Diagnostics during Eval ---
    compute_sparsity_diagnostics: bool = (
        False  # Whether to compute and log detailed sparsity tanh/z-score stats during eval
    )


@dataclass
class InferenceConfig:
    """Configuration for CLT inference/evaluation using a trained model."""

    clt_checkpoint_path: str  # Path to the .pt CLT model checkpoint or sharded checkpoint directory
    # data_path can be a manifest.json, a directory of .pt files, or path to Streaming/Remote config for eval
    data_path: str
    eval_batch_size_tokens: int = 4096
    max_eval_batches: Optional[int] = None  # Limit number of batches for evaluation
    device: Optional[str] = None  # "cuda", "cpu", "mps"
    output_log_dir: str = "clt_inference_results"
    # If data_path is for Streaming/Remote, these configs are needed:
    data_source_type: Literal["local_manifest", "generate_from_hf", "remote_server"] = "local_manifest"
    # For generate_from_hf
    generation_config_path: Optional[str] = (
        None  # Path to a YAML/JSON file with generation_config for ActivationExtractorCLT
    )
    dataset_params_path: Optional[str] = (
        None  # Path to a YAML/JSON file with dataset_params for extractor.stream_activations
    )
    # For remote_server
    remote_server_config_path: Optional[str] = None  # Path to YAML/JSON with remote_config (server_url, dataset_id etc)

    activation_dtype: Optional[str] = "float16"  # Dtype for activations from store, e.g. float16, bfloat16, float32
    normalization_method: Literal["none", "loaded_mean_std"] = "loaded_mean_std"  # For eval, usually use loaded or none

    # WandB options for logging evaluation results
    enable_wandb: bool = False
    wandb_project: Optional[str] = "clt_evaluation"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = field(default_factory=list)

    # Optional: If evaluating a specific model name for context in WandB
    model_name_for_wandb: Optional[str] = None

    # If using sharded model, world_size for reconstructing model state
    # For non-sharded models or if CLTConfig contains TP info, this might not be needed or can be 1.
    # Primarily for loading a sharded model checkpoint into a non-distributed InferenceRunner.
    model_world_size_for_load: int = 1
