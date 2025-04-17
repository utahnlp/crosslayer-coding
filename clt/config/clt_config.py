from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


@dataclass
class CLTConfig:
    """Configuration for a Cross-Layer Transcoder."""

    num_features: int  # Number of features per layer
    num_layers: int  # Number of transformer layers
    d_model: int  # Dimension of model's hidden state
    activation_fn: Literal["jumprelu", "relu"] = "jumprelu"
    jumprelu_threshold: float = 0.03  # Threshold for JumpReLU activation
    clt_dtype: Optional[str] = (
        None  # Optional dtype for the CLT model itself (e.g., "float16")
    )

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.num_features > 0, "Number of features must be positive"
        assert self.num_layers > 0, "Number of layers must be positive"
        assert self.d_model > 0, "Model dimension must be positive"
        assert self.jumprelu_threshold > 0, "JumpReLU threshold must be positive"


@dataclass
class TrainingConfig:
    """Configuration for training a Cross-Layer Transcoder."""

    # Basic training parameters
    learning_rate: float  # Learning rate for optimizer
    training_steps: int  # Total number of training steps

    # Training batch size (tokens)
    train_batch_size_tokens: int = 4096  # Number of tokens per training step batch
    # Buffer size for streaming store
    n_batches_in_buffer: int = 16  # Number of extraction batches in buffer

    # Normalization parameters
    normalization_method: Literal["auto", "estimated_mean_std", "none"] = "auto"
    # 'auto': Use pre-calculated from mapped store, or estimate for streaming store.
    # 'estimated_mean_std': Always estimate for streaming store (ignored for mapped).
    # 'none': Disable normalization.
    normalization_estimation_batches: int = 50  # Batches for normalization estimation

    # --- Activation Store Source --- #
    activation_source: Literal["generate", "local", "remote"] = "generate"
    # Config for "generate" source (on-the-fly)
    generation_config: Optional[Dict[str, Any]] = (
        None  # Dict matching ActivationConfig fields needed for extractor
    )
    dataset_params: Optional[Dict[str, Any]] = (
        None  # Dict matching dataset fields for stream_activations
    )
    # Config for "local" source (pre-generated)
    activation_path: Optional[str] = (
        None  # Path to pre-generated activation dataset directory
    )
    # Config for "remote" source (STUBBED)
    remote_config: Optional[Dict[str, Any]] = (
        None  # Dict with server_url, dataset_id, etc.
    )
    # How many batches to prefetch for remote source
    remote_prefetch_batches: Optional[int] = 4
    # --- End Activation Store Source --- #

    # Loss function coefficients
    sparsity_lambda: float = 1e-3  # Coefficient for sparsity penalty
    sparsity_c: float = 1.0  # Parameter affecting sparsity penalty shape
    preactivation_coef: float = 3e-6  # Coefficient for pre-activation loss

    # Optimizer parameters
    optimizer: Literal["adam", "adamw"] = "adamw"
    lr_scheduler: Optional[Literal["linear", "cosine"]] = "linear"

    # Logging parameters
    log_interval: int = 100  # How often to log metrics
    eval_interval: int = 1000  # How often to run evaluation
    checkpoint_interval: int = 1000  # How often to save checkpoints

    # Dead feature tracking
    dead_feature_window: int = 1000  # Steps until a feature is considered dead

    # WandB logging configuration
    enable_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_project: Optional[str] = None  # WandB project name
    wandb_entity: Optional[str] = None  # WandB entity/organization name
    wandb_run_name: Optional[str] = (
        None  # WandB run name, defaults to timestamp if None
    )
    wandb_tags: Optional[list] = None  # Tags for the WandB run

    def __post_init__(self):
        """Validate training parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.training_steps > 0, "Training steps must be positive"
        assert (
            self.train_batch_size_tokens > 0
        ), "Training batch size (tokens) must be positive"
        assert self.n_batches_in_buffer > 0, "Buffer size must be positive"
        assert self.sparsity_lambda >= 0, "Sparsity lambda must be non-negative"
        assert self.dead_feature_window > 0, "Dead feature window must be positive"

        # Validate activation source configuration
        if self.activation_source == "generate":
            assert (
                self.generation_config is not None
            ), "generation_config dict must be provided when activation_source is 'generate'"
            assert (
                self.dataset_params is not None
            ), "dataset_params dict must be provided when activation_source is 'generate'"
            # Basic check for essential keys in the dicts (can be expanded)
            assert (
                "model_name" in self.generation_config
            ), "generation_config missing 'model_name'"
            assert (
                "dataset_path" in self.dataset_params
            ), "dataset_params missing 'dataset_path'"
        elif self.activation_source == "local":
            assert (
                self.activation_path is not None
            ), "activation_path must be specified when activation_source is 'local'"
        elif self.activation_source == "remote":
            assert (
                self.remote_config is not None
            ), "remote_config dict must be provided when activation_source is 'remote'"
            assert (
                "server_url" in self.remote_config
                and "dataset_id" in self.remote_config
            ), "remote_config must contain 'server_url' and 'dataset_id'"
            if self.remote_prefetch_batches is not None:
                assert (
                    self.remote_prefetch_batches >= 1
                ), "remote_prefetch_batches must be at least 1"
