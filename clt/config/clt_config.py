from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CLTConfig:
    """Configuration for a Cross-Layer Transcoder."""

    num_features: int  # Number of features per layer
    num_layers: int  # Number of transformer layers
    d_model: int  # Dimension of model's hidden state
    activation_fn: Literal["jumprelu", "relu"] = "jumprelu"
    jumprelu_threshold: float = 0.03  # Threshold for JumpReLU activation

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

    # Model parameters
    model_name: str = "gpt2"  # Name of the model to extract activations from

    # Dataset parameters
    dataset_path: str = "NeelNanda/pile-10k"  # Path or name of the HuggingFace dataset
    dataset_split: str = "train"  # Dataset split to use
    dataset_text_column: str = "text"  # Name of the column containing text data
    streaming: bool = True  # Whether to use streaming mode for the dataset
    dataset_trust_remote_code: Optional[bool] = None  # Argument for load_dataset
    max_samples: Optional[int] = None  # Maximum number of samples to use from dataset

    # Tokenization parameters
    context_size: int = 1024  # Context window size for processing
    prepend_bos: bool = True  # Whether to prepend BOS token to texts
    exclude_special_tokens: bool = False  # Whether to filter out special tokens

    # Batch size parameters
    store_batch_size_prompts: int = 4  # Number of prompts per extraction batch
    batch_size: int = 64  # Number of sequences per training batch
    n_batches_in_buffer: int = 16  # Number of extraction batches in buffer

    # Normalization parameters
    normalization_method: Literal["mean_std", "estimated_mean_std", "none"] = "mean_std"
    normalization_estimation_batches: int = 50  # Batches for normalization estimation

    # Activation caching parameters
    cache_path: Optional[str] = None  # Path to save/load extracted activations

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

    @property
    def train_batch_size_tokens(self) -> int:
        """Calculate the number of tokens per training batch.

        Returns:
            The product of batch_size and context_size
        """
        return self.batch_size * self.context_size

    def __post_init__(self):
        """Validate training parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.training_steps > 0, "Training steps must be positive"
        assert (
            self.store_batch_size_prompts > 0
        ), "Extraction batch size must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.n_batches_in_buffer > 0, "Buffer size must be positive"
        assert self.context_size > 0, "Context size must be positive"
        assert self.sparsity_lambda >= 0, "Sparsity lambda must be non-negative"
        assert isinstance(self.streaming, bool), "Streaming must be boolean"
        assert self.dead_feature_window > 0, "Dead feature window must be positive"
