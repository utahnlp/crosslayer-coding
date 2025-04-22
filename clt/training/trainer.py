import torch
import torch.optim as optim
from typing import Dict, Optional, Union, Any
from tqdm import tqdm  # type: ignore
import os
import json
import time
import importlib.util
import sys
import logging  # Add logging import
import datetime  # Import datetime for formatting
import torch.distributed as dist  # Add distributed import
from torch.nn.parallel import DistributedDataParallel as DDP  # Import DDP

# Needed for isinstance checks
from torch.nn import Module

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data import (
    BaseActivationStore,
    StreamingActivationStore,
    # MappedActivationStore, # Removed legacy store
)

# Import the new manifest-based stores
from clt.training.local_activation_store import LocalActivationStore
from clt.training.remote_activation_store import RemoteActivationStore

# Import Manifest base class for type checking
# from clt.training.manifest_activation_store import ManifestActivationStore

from clt.training.losses import LossManager
from clt.nnsight.extractor import (
    ActivationExtractorCLT,
)  # Keep for StreamingStore usage
from .evaluator import CLTEvaluator  # Import the new evaluator

# Get logger for this module
logger = logging.getLogger(__name__)


# Helper function to format elapsed time
def _format_elapsed_time(seconds: float) -> str:
    """Formats elapsed seconds into HH:MM:SS or MM:SS."""
    td = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0 or hours > 0:
        return f"{td.days * 24 + hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


class WandBLogger:
    """Wrapper class for Weights & Biases logging. Ensures init only on rank 0."""

    def __init__(
        self,
        training_config: TrainingConfig,
        clt_config: CLTConfig,
        log_dir: str,
        rank: int = 0,  # Add rank
    ):
        """Initialize the WandB logger, only on rank 0.

        Args:
            training_config: Training configuration
            clt_config: CLT model configuration
            log_dir: Directory to save logs
            rank: Process rank for distributed training.
        """
        self.enabled = training_config.enable_wandb
        self.log_dir = log_dir
        self.rank = rank
        self.is_rank_zero = rank == 0

        if not self.enabled or not self.is_rank_zero:
            self.enabled = False  # Disable if not rank 0 or explicitly disabled
            return

        # Check if wandb is installed
        if not importlib.util.find_spec("wandb"):
            print(
                "Warning (Rank 0): WandB logging requested but wandb not installed. "
                "Install with 'pip install wandb'. Continuing without WandB."
            )
            self.enabled = False
            return

        # Import wandb
        import wandb

        # Set up run name with timestamp if not provided
        run_name = training_config.wandb_run_name
        if run_name is None:
            run_name = f"clt-{time.strftime('%Y%m%d-%H%M%S')}-rank{self.rank}"  # Include rank?

        # Initialize wandb only on rank 0
        try:
            wandb.init(
                project=training_config.wandb_project,
                entity=training_config.wandb_entity,
                name=run_name,
                dir=log_dir,  # Ensure logs are saved in the correct directory
                tags=training_config.wandb_tags,
                config={
                    **training_config.__dict__,
                    **clt_config.__dict__,
                    "log_dir": log_dir,
                },
                mode="online" if self.is_rank_zero else "disabled",  # Disable on non-zero ranks
            )
            if wandb.run is not None:
                print(f"WandB logging initialized on Rank 0: {wandb.run.name}")
        except Exception as e:
            print(f"Warning (Rank 0): Failed to initialize WandB: {e}. Disabling WandB.")
            self.enabled = False

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        lr: Optional[float] = None,
        sparsity_lambda: Optional[float] = None,
        total_tokens_processed: Optional[int] = None,
    ):
        """Log metrics for a training step under the 'training/' group. Only logs if enabled (rank 0)."""
        if not self.enabled:  # Already checks for rank 0 internally
            return

        # Import wandb only when needed and enabled
        try:
            import wandb

            # Rename loss keys for clarity and add 'training/' prefix
            metrics = {}
            for key, value in loss_dict.items():
                if key == "total":
                    metrics["training/total_loss"] = value
                elif key == "sparsity":
                    metrics["training/sparsity_loss"] = value
                elif key == "reconstruction":
                    # Reconstruction loss is part of training, log it here too if present
                    metrics["training/reconstruction_loss"] = value
                elif key == "preactivation":
                    metrics["training/preactivation_loss"] = value
                else:
                    # Keep other potential keys, prepending 'training/'
                    metrics[f"training/{key}"] = value

            # Add learning rate
            if lr is not None:
                metrics["training/learning_rate"] = lr

            # Add sparsity lambda
            if sparsity_lambda is not None:
                metrics["training/sparsity_lambda"] = sparsity_lambda

            # Add total tokens processed
            # Note: In DDP, this will be the total tokens processed across *all* ranks
            if total_tokens_processed is not None:
                metrics["training/total_tokens_processed_global"] = total_tokens_processed

            # Log to wandb
            wandb.log(metrics, step=step)

        except ImportError:
            # This shouldn't happen if initialization succeeded, but as a safeguard
            self.enabled = False
            print("Warning (Rank 0): wandb not found during log_step. Disabling WandB.")
        except Exception as e:
            print(f"Warning (Rank 0): Error during WandB log_step: {e}")

    def log_evaluation(self, step: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics. Only logs if enabled (rank 0)."""
        if not self.enabled:  # Already checks for rank 0 internally
            return

        try:
            import wandb

            # Log metrics directly, assuming keys are already structured
            # e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'layerwise/l0/layer_0'
            wandb_log_dict: Dict[str, Any] = {}
            for key, value in eval_metrics.items():
                if key.startswith("layerwise/"):
                    # Handle nested layerwise data (histograms and scalars)
                    if isinstance(value, dict):
                        for layer_key, layer_value in value.items():
                            # Construct wandb key: e.g., layerwise/l0/layer_0
                            wandb_key = f"{key}/{layer_key}"
                            if isinstance(layer_value, list):
                                # Log list data as histogram
                                try:
                                    wandb_log_dict[wandb_key] = wandb.Histogram(layer_value)
                                except Exception as e:
                                    print(f"Wandb (Rank 0): Error creating histogram for {wandb_key}: {e}")
                                    # Fallback: log mean or placeholder
                                    try:
                                        mean_val = sum(layer_value) / len(layer_value) if layer_value else 0.0
                                        wandb_log_dict[f"{wandb_key}_mean"] = mean_val
                                    except TypeError:
                                        wandb_log_dict[f"{wandb_key}_mean"] = -1.0
                            elif isinstance(layer_value, (float, int)):
                                # Log scalar layerwise data
                                wandb_log_dict[wandb_key] = layer_value
                    else:
                        # If the top level key itself is scalar (shouldn't happen with current structure)
                        wandb_log_dict[key] = value
                elif key.endswith("_agg_hist") and isinstance(value, list):
                    # Handle aggregate histogram data
                    try:
                        wandb_log_dict[key] = wandb.Histogram(value)
                    except Exception as e:
                        print(f"Wandb (Rank 0): Error creating aggregate histogram for {key}: {e}")
                        # Optional Fallback: log mean
                        try:
                            mean_val = sum(value) / len(value) if value else 0.0
                            wandb_log_dict[f"{key}_mean"] = mean_val
                        except TypeError:
                            wandb_log_dict[f"{key}_mean"] = -1.0
                elif isinstance(value, (float, int)):  # Handle top-level scalars
                    wandb_log_dict[key] = value
                # Add other specific handling if needed

            # Log the prepared dictionary to wandb
            if wandb_log_dict:
                wandb.log(wandb_log_dict, step=step)

        except ImportError:
            self.enabled = False
            print("Warning (Rank 0): wandb not found during log_evaluation. Disabling WandB.")
        except Exception as e:
            print(f"Warning (Rank 0): Error during WandB log_evaluation: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        """Log an artifact to WandB. Only logs if enabled (rank 0)."""
        if not self.enabled:  # Already checks for rank 0 internally
            return

        try:
            import wandb

            # Use filename if name not provided
            if name is None:
                name = os.path.basename(artifact_path)

            # Create and log artifact
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
            print(f"Wandb (Rank 0): Logged artifact '{name}' ({artifact_type}) from {artifact_path}")

        except ImportError:
            self.enabled = False
            print("Warning (Rank 0): wandb not found during log_artifact. Disabling WandB.")
        except Exception as e:
            print(f"Warning (Rank 0): Error logging artifact {name} to WandB: {e}")

    def finish(self):
        """Finish the WandB run. Only finishes if enabled (rank 0)."""
        if not self.enabled:  # Already checks for rank 0 internally
            return

        try:
            import wandb

            wandb.finish()
            print("WandB (Rank 0): Run finished.")
        except ImportError:
            pass  # Wandb wasn't imported, nothing to finish
        except Exception as e:
            print(f"Warning (Rank 0): Error finishing WandB run: {e}")


class CLTTrainer:
    """Trainer for Cross-Layer Transcoder models, supporting DDP."""

    # Add type hint for the activation store attribute
    activation_store: BaseActivationStore
    # Model can be base or DDP wrapped. Using Module for broader compatibility initially.
    model: Module

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        rank: int = 0,  # Add rank
        world: int = 1,  # Add world size
        ddp: bool = False,  # Add ddp flag
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training (e.g., 'cuda:0')
            rank: Process rank for distributed training.
            world: World size for distributed training.
            ddp: Flag indicating if DistributedDataParallel is being used.
        """
        self.clt_config = clt_config
        self.training_config = training_config
        self.rank = rank
        self.world = world
        self.ddp = ddp
        self.is_rank_zero = rank == 0

        # Ensure self.device is a torch.device object
        _device_input = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input

        # Set up log directory (all ranks create it for potential local logging)
        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        if self.is_rank_zero:  # Only rank 0 creates the directory to avoid race conditions
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Log directory: {self.log_dir}")

        # Record start time
        self.start_time = time.time()

        # Initialize model on the correct device *before* potentially wrapping with DDP
        # The CLT model now takes device/dtype in __init__
        # Initialize as CrossLayerTranscoder first
        _model: CrossLayerTranscoder = CrossLayerTranscoder(clt_config, device=self.device)
        logger.info(f"Rank {self.rank}: Base CrossLayerTranscoder initialized.")

        # --- DEBUG: Check parameter count before DDP ---
        try:
            num_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
            logger.info(f"Rank {self.rank}: Parameter count BEFORE DDP wrap: {num_params}")
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error counting parameters BEFORE DDP wrap: {e}")

        # --- DEBUG: Add barrier before DDP initialization ---
        if self.ddp:
            logger.info(f"Rank {self.rank}: Entering barrier before DDP initialization...")
            dist.barrier()
            logger.info(f"Rank {self.rank}: Passed barrier before DDP initialization.")

        # --- Wrap model with DDP if needed ---
        if self.ddp:
            if not isinstance(self.device, torch.device) or self.device.type != "cuda":
                raise ValueError("DDP requires CUDA device.")
            # Ensure model is on the correct device before wrapping
            _model = _model.to(self.device)
            self.model = DDP(  # Assign to self.model here
                _model,
                device_ids=[self.device.index],  # device is e.g., torch.device('cuda:1')
                output_device=self.device.index,
                find_unused_parameters=False,  # Set to True if graph has unused outputs
            )
            logger.info(f"Rank {self.rank}: Wrapped model with DDP.")
        else:
            # Ensure model is on device even if not using DDP
            self.model = _model.to(self.device)  # Assign to self.model here
            logger.info(f"Rank {self.rank}: Model placed on device {self.device} (DDP disabled).")

        logger.info(f"Rank {self.rank}: Initializing optimizer...")
        # Initialize optimizer (after model is potentially wrapped and moved to device)
        # DDP handles parameter synchronization, so using self.model.parameters() is correct
        if training_config.optimizer == "adam":
            self.optimizer: Any = optim.Adam(self.model.parameters(), lr=training_config.learning_rate)
        else:  # "adamw"
            self.optimizer = optim.AdamW(self.model.parameters(), lr=training_config.learning_rate)
        logger.info(f"Rank {self.rank}: Optimizer '{training_config.optimizer}' initialized.")

        logger.info(f"Rank {self.rank}: Initializing scheduler...")
        # Initialize scheduler
        self.scheduler: Optional[Any] = None
        scheduler_type = training_config.lr_scheduler
        scheduler_params = training_config.lr_scheduler_params or {}

        if scheduler_type == "linear":
            default_linear_params = {"start_factor": 1.0, "end_factor": 0.1}
            final_params = {**default_linear_params, **scheduler_params}
            final_params.pop("total_iters", None)
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=training_config.training_steps,
                **final_params,
            )
            logger.info(
                f"Rank {self.rank}: Using LinearLR scheduler with params: {final_params}, total_iters={training_config.training_steps}"
            )
        elif scheduler_type == "cosine":
            default_cosine_params = {"eta_min": 0}
            final_params = {**default_cosine_params, **scheduler_params}
            t_max = final_params.pop("T_max", training_config.training_steps)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, **final_params)
            logger.info(
                f"Rank {self.rank}: Using CosineAnnealingLR scheduler with params: {final_params}, T_max={t_max}"
            )
        logger.info(f"Rank {self.rank}: Scheduler initialized.")

        logger.info(f"Rank {self.rank}: Creating activation store...")
        # Initialize activation store based on config, passing rank/world
        self.activation_store = self._create_activation_store(self.start_time, self.rank, self.world)
        logger.info(f"Rank {self.rank}: Activation store created.")

        logger.info(f"Rank {self.rank}: Initializing loss manager...")
        # Initialize loss manager
        self.loss_manager = LossManager(training_config)
        logger.info(f"Rank {self.rank}: Loss manager initialized.")

        logger.info(f"Rank {self.rank}: Initializing evaluator...")
        # Initialize Evaluator - Evaluation typically done on rank 0
        # Pass the underlying CrossLayerTranscoder instance
        underlying_model = self.model.module if self.ddp else self.model
        assert isinstance(underlying_model, CrossLayerTranscoder), "Model for Evaluator must be CrossLayerTranscoder"
        self.evaluator = CLTEvaluator(underlying_model, self.device, self.start_time)
        logger.info(f"Rank {self.rank}: Evaluator initialized.")

        logger.info(f"Rank {self.rank}: Initializing dead neuron counters...")
        # Initialize dead neuron counters (local per rank)
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,  # Local device
            dtype=torch.long,
        )
        logger.info(f"Rank {self.rank}: Dead neuron counters initialized.")

        # Training metrics (only populated on rank 0)
        self.metrics: Dict[str, list] = {
            "train_losses": [],
            "eval_metrics": [],
        }

        logger.info(f"Rank {self.rank}: Initializing WandB logger...")
        # Initialize WandB logger (handles rank internally)
        self.wandb_logger = WandBLogger(
            training_config=training_config,
            clt_config=clt_config,
            log_dir=self.log_dir,
            rank=self.rank,
        )
        logger.info(f"Rank {self.rank}: CLTTrainer initialization complete.")

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window (local to rank)."""
        # Ensure counter is initialized
        if not hasattr(self, "n_forward_passes_since_fired") or self.n_forward_passes_since_fired is None:
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        return self.n_forward_passes_since_fired > self.training_config.dead_feature_window

    def _get_underlying_model(self) -> CrossLayerTranscoder:
        """Helper to get the underlying CrossLayerTranscoder model, asserting type."""
        model = self.model.module if self.ddp else self.model
        assert isinstance(model, CrossLayerTranscoder), "Model must be a CrossLayerTranscoder instance"
        return model

    def _create_activation_store(
        self,
        start_time: float,
        rank: int,  # Add rank
        world: int,  # Add world size
    ) -> BaseActivationStore:
        """Create the appropriate activation store based on training config, passing rank/world."""

        activation_source = self.training_config.activation_source
        store: BaseActivationStore  # Correct type hint for the variable

        # Note: Rank and world are now passed into the method

        if activation_source == "generate":
            # --- Streaming Store --- #
            # Note: StreamingActivationStore needs modification to handle rank/world for sampling
            logger.info(f"Rank {rank}: Using StreamingActivationStore (generating on-the-fly).")
            gen_cfg = self.training_config.generation_config
            ds_params = self.training_config.dataset_params
            if gen_cfg is None or ds_params is None:
                raise ValueError(
                    "generation_config and dataset_params must be provided in TrainingConfig for on-the-fly generation."
                )

            extractor = ActivationExtractorCLT(
                model_name=gen_cfg["model_name"],
                mlp_input_module_path_template=gen_cfg["mlp_input_template"],
                mlp_output_module_path_template=gen_cfg["mlp_output_template"],
                device=self.device,  # Use the rank-specific device
                model_dtype=gen_cfg.get("model_dtype"),
                context_size=gen_cfg.get("context_size", 128),
                inference_batch_size=gen_cfg.get("inference_batch_size", 512),
                exclude_special_tokens=gen_cfg.get("exclude_special_tokens", True),
                prepend_bos=gen_cfg.get("prepend_bos", False),
                nnsight_tracer_kwargs=gen_cfg.get("nnsight_tracer_kwargs"),
                nnsight_invoker_args=gen_cfg.get("nnsight_invoker_args"),
            )

            activation_generator = extractor.stream_activations(
                dataset_path=ds_params["dataset_path"],
                dataset_split=ds_params.get("dataset_split", "train"),
                dataset_text_column=ds_params.get("dataset_text_column", "text"),
                streaming=ds_params.get("streaming", True),
                dataset_trust_remote_code=ds_params.get("dataset_trust_remote_code", False),
                cache_path=ds_params.get("cache_path"),
                # Pass rank/world for potential dataset sharding if supported by underlying library
                # rank=rank, # Example: if datasets lib supports this
                # world_size=world,
            )

            stream_norm_method = self.training_config.normalization_method
            if stream_norm_method == "auto":
                stream_norm_method = "estimated_mean_std"

            # TODO: Modify StreamingActivationStore to accept and use rank/world for sampling/buffering
            store = StreamingActivationStore(  # Assign to store
                activation_generator=activation_generator,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                n_batches_in_buffer=self.training_config.n_batches_in_buffer,
                normalization_method=str(stream_norm_method),
                normalization_estimation_batches=(self.training_config.normalization_estimation_batches),
                device=self.device,  # Use rank-specific device
                start_time=start_time,
                # rank=rank, # Pass rank/world when implemented in StreamingActivationStore
                # world=world,
            )
            logger.info(f"Rank {rank}: Initialized StreamingActivationStore.")
            if store.normalization_method == "estimated_mean_std":
                # Normalization estimation needs coordination in DDP (e.g., estimate on rank 0, broadcast stats)
                logger.warning(
                    f"Rank {rank}: Normalization estimation in StreamingActivationStore "
                    f"needs modification for DDP. Currently estimating locally."
                )
                # Add dist.barrier() here if needed after rank 0 broadcasts stats
            logger.info(f"Rank {rank}:   Normalization method: {store.normalization_method}")

        elif activation_source == "local_manifest":
            # --- Local Manifest Store --- #
            logger.info(f"Rank {rank}: Using LocalActivationStore (reading local manifest/chunks).")
            if not self.training_config.activation_path:
                raise ValueError(
                    "activation_path must be set in TrainingConfig when activation_source is 'local_manifest'."
                )
            store = LocalActivationStore(  # Assign to store
                dataset_path=self.training_config.activation_path,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                device=self.device,  # Use rank-specific device
                dtype=self.training_config.activation_dtype,
                rank=rank,  # Pass rank
                world=world,  # Pass world size
                seed=self.training_config.seed,
            )
            logger.info(f"Rank {rank}: Initialized LocalActivationStore from path: {store.dataset_path}")
            if store.apply_normalization:
                logger.info(f"Rank {rank}:   Normalization ENABLED using loaded norm_stats.json.")
            else:
                logger.warning(f"Rank {rank}:   Normalization DISABLED (norm_stats.json not found/failed).")

        elif activation_source == "remote":
            # --- Remote Manifest Store --- #
            logger.info(f"Rank {rank}: Using RemoteActivationStore (remote slice server).")
            remote_cfg = self.training_config.remote_config
            if remote_cfg is None:
                raise ValueError("remote_config dict must be set in TrainingConfig when activation_source is 'remote'.")
            server_url = remote_cfg.get("server_url")
            dataset_id = remote_cfg.get("dataset_id")
            if not server_url or not dataset_id:
                raise ValueError("remote_config must contain 'server_url' and 'dataset_id'.")

            store = RemoteActivationStore(  # Assign to store
                server_url=server_url,
                dataset_id=dataset_id,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                device=self.device,  # Use rank-specific device
                dtype=self.training_config.activation_dtype,
                rank=rank,  # Pass rank
                world=world,  # Pass world size
                seed=self.training_config.seed,
                timeout=remote_cfg.get("timeout", 60),
            )
            # Accessing specific attributes requires isinstance check or careful typing
            # For logging, it's often acceptable to assume the type or use getattr
            did_raw = getattr(store, "did_raw", "unknown")
            logger.info(f"Rank {rank}: Initialized RemoteActivationStore for dataset: {did_raw}")
            apply_norm = getattr(store, "apply_normalization", False)
            if apply_norm:
                logger.info(f"Rank {rank}:   Normalization ENABLED using fetched norm_stats.json.")
            else:
                logger.warning(f"Rank {rank}:   Normalization DISABLED (norm_stats.json not found/failed).")
        else:
            raise ValueError(
                f"Unknown activation_source: {activation_source}. Valid options: 'generate', 'local_manifest', 'remote'."
            )

        return store

    def _log_metrics(self, step: int, loss_dict: Dict[str, float]):
        """Log training metrics (only on rank 0)."""
        # --- Save metrics locally (only on rank 0) ---
        if self.is_rank_zero:
            # Store loss for saving to JSON later
            self.metrics["train_losses"].append({"step": step, **loss_dict})

            # --- Gather additional metrics for logging --- #
            current_lr = None
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]

            current_lambda = self.loss_manager.get_current_sparsity_lambda()

            # Calculate total tokens processed across all ranks
            total_tokens_processed_global = self.training_config.train_batch_size_tokens * (step + 1) * self.world

            # --- Log to WandB (rank 0 only) ---
            self.wandb_logger.log_step(
                step,
                loss_dict,
                lr=current_lr,
                sparsity_lambda=current_lambda,
                total_tokens_processed=total_tokens_processed_global,
            )

            # --- Save metrics file periodically (rank 0 only) ---
            log_interval = self.training_config.log_interval
            if step > 0 and step % log_interval == 0:
                self._save_metrics()

    def _save_metrics(self):
        """Save training metrics to disk (only on rank 0)."""
        if not self.is_rank_zero:
            return

        metrics_path = os.path.join(self.log_dir, "metrics.json")
        try:
            with open(metrics_path, "w") as f:
                # Use default=str to handle potential non-serializable types like torch tensors
                json.dump(self.metrics, f, indent=2, default=str)
            # logger.debug(f"Rank 0: Saved metrics to {metrics_path}") # Less verbose
        except Exception as e:
            logger.error(f"Rank 0: Failed to save metrics to {metrics_path}: {e}", exc_info=True)

    def _save_checkpoint(self, step: int):
        """Save a checkpoint (model, optimizer, store state) (only on rank 0)."""
        if not self.is_rank_zero:
            return

        # Ensure log directory exists (should already exist from __init__)
        os.makedirs(self.log_dir, exist_ok=True)

        # --- Prepare state dicts ---
        # Get model state dict, handling DDP wrapper
        underlying_model = self._get_underlying_model()
        model_state_dict = underlying_model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        store_state_dict = self.activation_store.state_dict()

        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "store": store_state_dict,
            "step": step,
            # Include configs for easier reloading/inspection
            "clt_config": self.clt_config.__dict__,
            "training_config": self.training_config.__dict__,
            # Optional: Add scheduler state if needed
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

        # --- Save checkpoint file ---
        checkpoint_filename = f"clt_checkpoint_{step}.pt"
        checkpoint_path = os.path.join(self.log_dir, checkpoint_filename)
        latest_path = os.path.join(self.log_dir, "clt_checkpoint_latest.pt")

        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Rank 0: Saved checkpoint to {checkpoint_path}")

            # Save a copy as latest
            torch.save(checkpoint, latest_path)
            # logger.debug(f"Rank 0: Updated latest checkpoint link to {latest_path}")

            # --- Log checkpoint artifact to WandB ---
            self.wandb_logger.log_artifact(
                artifact_path=checkpoint_path,
                artifact_type="model_checkpoint",
                name=f"clt_checkpoint_{step}",
            )

        except Exception as e:
            logger.error(f"Rank 0: Failed to save checkpoint to {checkpoint_path}: {e}", exc_info=True)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        # store_checkpoint_path: Optional[str] = None # Store state is now inside main checkpoint
    ):
        """Load model, optimizer, and activation store state from a checkpoint.

        Handles loading state dict whether the checkpoint was saved with DDP or not,
        and whether the current trainer is using DDP or not.

        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found at {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load checkpoint onto the current rank's device
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Rank {self.rank}: Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
            raise

        # --- Load Model State ---
        # Get the state dict from the checkpoint
        model_state_dict = checkpoint.get("model")
        if model_state_dict is None:
            logger.error(f"Rank {self.rank}: Checkpoint {checkpoint_path} does not contain model state_dict.")
            raise ValueError("Invalid checkpoint: Missing model state.")

        # Adjust keys if loading a DDP-saved checkpoint into a non-DDP model
        # Or if loading a non-DDP checkpoint into a DDP model (DDP wrapper handles this case)
        if self.ddp:
            # Current model is DDP-wrapped. DDP wrapper's load_state_dict handles
            # both cases: loading from DDP-saved (module. prefix) or non-DDP saved.
            try:
                # DDP model expects state_dict of the underlying module
                self.model.load_state_dict(model_state_dict)
                logger.info(f"Rank {self.rank}: Loaded model state into DDP model.")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to load model state into DDP model: {e}", exc_info=True)
                raise
        else:
            # Current model is plain CrossLayerTranscoder.
            # If checkpoint was saved from DDP, keys might start with 'module.'
            # We need to remove this prefix.
            if isinstance(self.model, CrossLayerTranscoder) and all(
                k.startswith("module.") for k in model_state_dict.keys()
            ):
                logger.info(f"Rank {self.rank}: Removing 'module.' prefix from checkpoint keys for non-DDP model.")
                model_state_dict = {k[len("module.") :]: v for k, v in model_state_dict.items()}

            try:
                # Load into the plain model
                self.model.load_state_dict(model_state_dict)
                logger.info(f"Rank {self.rank}: Loaded model state into non-DDP model.")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to load model state into non-DDP model: {e}", exc_info=True)
                raise

        # --- Load Optimizer State ---
        optimizer_state_dict = checkpoint.get("optimizer")
        if optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(optimizer_state_dict)
                logger.info(f"Rank {self.rank}: Loaded optimizer state.")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to load optimizer state: {e}", exc_info=True)
                # Don't raise, maybe training can continue without optimizer state
        else:
            logger.warning(f"Rank {self.rank}: Optimizer state not found in checkpoint.")

        # --- Load Scheduler State (Optional) ---
        scheduler_state_dict = checkpoint.get("scheduler")
        if scheduler_state_dict and self.scheduler:
            try:
                self.scheduler.load_state_dict(scheduler_state_dict)
                logger.info(f"Rank {self.rank}: Loaded scheduler state.")
            except Exception as e:
                logger.warning(f"Rank {self.rank}: Failed to load scheduler state: {e}")
        elif self.scheduler:
            logger.warning(f"Rank {self.rank}: Scheduler state not found in checkpoint.")

        # --- Load Activation Store State ---
        store_state_dict = checkpoint.get("store")
        if store_state_dict:
            try:
                # Ensure activation_store is initialized before loading state
                if not hasattr(self, "activation_store") or self.activation_store is None:
                    logger.error("Rank {self.rank}: Activation store not initialized. Cannot load state.")
                else:
                    self.activation_store.load_state_dict(store_state_dict)
                    # Check for epoch attribute before logging it
                    current_epoch = getattr(self.activation_store, "epoch", "N/A")
                    logger.info(f"Rank {self.rank}: Loaded activation store state (epoch {current_epoch}).")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Failed to load activation store state: {e}", exc_info=True)
                # Activation store state is crucial for correctness, consider raising
                raise ValueError("Failed to load critical activation store state.") from e
        else:
            logger.error(f"Rank {self.rank}: Activation store state not found in checkpoint.")
            raise ValueError("Invalid Checkpoint: Missing activation store state.")

        # --- Load Step ---
        start_step = checkpoint.get("step", -1) + 1
        if start_step > 0:
            logger.info(f"Rank {self.rank}: Resuming training from step {start_step}")
            # Note: The training loop needs to handle starting from this step
        else:
            logger.warning(f"Rank {self.rank}: Step not found in checkpoint, starting from step 0.")
            start_step = 0

        # --- Add barrier to ensure all ranks load before proceeding ---
        if self.ddp:
            dist.barrier()
            logger.info(f"Rank {self.rank}: Barrier passed after checkpoint load.")

        # Return the step to resume from
        return start_step

    def train(self) -> CrossLayerTranscoder:  # , eval_every: int = 1000 # eval_every is in config now
        """Train the CLT model."""

        start_step = 0  # Default start step
        # Example of how to integrate checkpoint loading:
        # if self.training_config.resume_from_checkpoint:
        #    try:
        #        start_step = self.load_checkpoint(self.training_config.resume_from_checkpoint)
        #    except Exception as e:
        #        logger.error(f"Rank {self.rank}: Failed to load checkpoint {self.training_config.resume_from_checkpoint}. Starting from scratch.")
        #        start_step = 0

        if self.is_rank_zero:
            print(f"Starting CLT training on {self.world} processes (Device: {self.device})...")
            # Access num_features and num_layers via clt_config
            print(f"Model: {self.clt_config.num_features} features/layer, {self.clt_config.num_layers} layers")
            print(f"Training for {self.training_config.training_steps} steps.")
            print(f"Logging to {self.log_dir}")
            if start_step > 0:
                print(f"Resuming from step {start_step}")

        # Barrier to ensure setup and prints (on rank 0) complete before training loop
        if self.ddp:
            dist.barrier()

        # Check if using normalization and notify user (only rank 0)
        # Note: Normalization estimation coordination for Streaming Store in DDP is needed
        if self.is_rank_zero and self.training_config.normalization_method == "estimated_mean_std":
            print("\n>>> NORMALIZATION PHASE (if applicable) <<<\n")
            if self.training_config.activation_source == "generate":
                print("StreamingStore: Normalization estimation needs DDP coordination (currently local estimate).")
            else:  # Manifest stores load pre-computed stats
                print("ManifestStore: Using pre-computed normalization stats if available.")
            # print(f"Using {self.training_config.normalization_estimation_batches} batches for estimation (if estimating).\n")
            sys.stdout.flush()
            # Barrier might be needed here if rank 0 estimates and broadcasts

        # Training loop using ActivationStore as iterator
        if self.is_rank_zero:
            print("\n>>> TRAINING PHASE <<<\n")
            sys.stdout.flush()

        # Use tqdm progress bar only on rank 0
        pbar = tqdm(
            range(start_step, self.training_config.training_steps),
            desc=f"Training CLT (Rank {self.rank})",
            leave=True,
            disable=(not self.is_rank_zero),  # Disable bar on non-zero ranks
            initial=start_step,  # Start counter from resume step
            total=self.training_config.training_steps,  # Set total steps
        )

        # --- Main Training Loop ---
        try:
            for step in pbar:
                # Set epoch for manifest samplers (needed for DDP consistency)
                # Check if it's a manifest store before setting epoch
                # We removed the external call to set_epoch here.
                # The ManifestActivationStore internally handles sampler epoch resets.
                # if isinstance(self.activation_store, ManifestActivationStore):
                #     current_epoch = getattr(self.activation_store, 'epoch', 0)
                #     self.activation_store.set_epoch(current_epoch)

                # Force display update of progress bar (rank 0)
                if self.is_rank_zero:
                    pbar.refresh()

                try:
                    # Get batch directly from the iterator (handles DDP sharding internally)
                    inputs, targets = next(self.activation_store)

                except StopIteration:
                    logger.warning(
                        f"Rank {self.rank}: Activation store exhausted at step {step}. Training finished early."
                    )
                    break  # Exit training loop if data runs out
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Error getting batch at step {step}: {e}. Skipping step.", exc_info=True
                    )
                    # Consider a barrier here? If one rank fails, others might hang.
                    # If an error is persistent, might need to terminate all ranks.
                    continue  # Skip this step if batch fetching fails

                # --- Check for empty batch ---
                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    logger.warning(f"Rank {self.rank}: Received empty batch at step {step}. Skipping.")
                    continue

                # --- Forward pass and compute loss ---
                # DDP automatically syncs gradients during backward pass
                self.optimizer.zero_grad()

                # Get the underlying CrossLayerTranscoder model instance
                underlying_model = self._get_underlying_model()

                # Pass the *underlying* model to loss manager methods that expect CrossLayerTranscoder
                loss, loss_dict = self.loss_manager.compute_total_loss(
                    underlying_model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
                )

                # --- Update Dead Neuron Counters (local per rank) ---
                if hasattr(self, "n_forward_passes_since_fired"):
                    with torch.no_grad():
                        try:
                            # Use underlying model to get activations
                            feature_activations_batch = underlying_model.get_feature_activations(inputs)

                            for layer_idx, layer_acts in feature_activations_batch.items():
                                if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                    if layer_acts.numel() > 0:
                                        fired_mask_per_token = layer_acts > 1e-6
                                        fired_features_this_layer = fired_mask_per_token.any(dim=0)

                                        if (
                                            fired_features_this_layer.shape[0]
                                            == self.n_forward_passes_since_fired.shape[1]
                                        ):
                                            self.n_forward_passes_since_fired[layer_idx] += 1
                                            self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                        else:
                                            logger.warning(
                                                f"Rank {self.rank}: Shape mismatch dead neuron update L{layer_idx}. "
                                                f"Acts: {layer_acts.shape}, Fired: {fired_features_this_layer.shape}, "
                                                f"Counter: {self.n_forward_passes_since_fired.shape}"
                                            )
                        except Exception as e:
                            logger.error(f"Rank {self.rank}: Error updating dead neuron counts: {e}", exc_info=True)

                # --- Backward pass ---
                if torch.isnan(loss):
                    logger.warning(
                        f"Rank {self.rank}: NaN loss encountered at step {step}. Skipping backward/optimizer."
                    )
                else:
                    try:
                        # Pass the potentially DDP-wrapped model to backward()
                        loss.backward()  # DDP handles gradient averaging here

                        # --- Gradient clipping --- #
                        if self.training_config.gradient_clip_val is not None:
                            # Clip grads on the DDP model directly is fine
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_val,
                            )

                    except RuntimeError as e:
                        logger.error(
                            f"Rank {self.rank}: Error during backward pass at step {step}: {e}. Skipping optimizer step.",
                            exc_info=True,
                        )
                        continue  # Skip optimizer step if backward fails

                    # --- Optimizer step ---
                    try:
                        self.optimizer.step()
                    except Exception as e:
                        logger.error(
                            f"Rank {self.rank}: Error during optimizer step at step {step}: {e}", exc_info=True
                        )
                        continue  # Avoid scheduler step if optimizer failed

                # --- Scheduler step ---
                if self.scheduler:
                    try:
                        self.scheduler.step()
                    except Exception as e:
                        logger.error(
                            f"Rank {self.rank}: Error during scheduler step at step {step}: {e}", exc_info=True
                        )

                # --- Update progress bar (rank 0) ---
                if self.is_rank_zero:
                    # Aggregate loss for display? Or just show rank 0 loss?
                    # Showing rank 0 loss is simpler.
                    description = (
                        f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)

                    # Force update to display progress every step
                    pbar.refresh()
                    sys.stdout.flush()

                # --- Log metrics (rank 0 handles logging) ---
                # Loss dict is from rank 0's calculation (if not aggregated)
                self._log_metrics(step, loss_dict)

                # --- Evaluation & Checkpointing (only on rank 0) ---
                if self.is_rank_zero:
                    eval_interval = self.training_config.eval_interval
                    checkpoint_interval = self.training_config.checkpoint_interval

                    is_last_step = step == self.training_config.training_steps - 1
                    save_checkpoint_flag = (step > 0 and step % checkpoint_interval == 0) or is_last_step
                    run_eval_flag = (step > 0 and step % eval_interval == 0) or is_last_step

                    if run_eval_flag:
                        logger.info(f"Rank 0: Running evaluation at step {step}...")
                        # Ensure evaluator uses the non-DDP model for consistency
                        current_underlying_model = self._get_underlying_model()
                        self.evaluator.model = current_underlying_model
                        # Detach mask for evaluator, which runs with no_grad
                        # Use local rank 0's dead neuron mask for evaluation
                        current_dead_mask = self.dead_neurons_mask.detach().clone()

                        try:
                            # Use the evaluator to compute metrics
                            # Inputs/targets are from rank 0's last batch
                            eval_metrics = self.evaluator.compute_metrics(
                                inputs,
                                targets,
                                dead_neuron_mask=current_dead_mask,
                            )

                            # Store evaluation metrics (for saving to JSON)
                            self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                            # --- Update Progress Bar Postfix --- #
                            l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                            ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                            avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                            dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                            eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                            eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                            pbar.set_postfix_str(eval_msg)
                            pbar.refresh()  # Force update

                            # --- Log evaluation metrics to WandB --- #
                            self.wandb_logger.log_evaluation(step, eval_metrics)

                            # --- Save metrics JSON after evaluation --- #
                            self._save_metrics()
                            logger.info(f"Rank 0: Evaluation complete at step {step}.")

                        except Exception as e:
                            logger.error(f"Rank 0: Error during evaluation at step {step}: {e}", exc_info=True)

                    if save_checkpoint_flag:
                        self._save_checkpoint(step)
                        # Optionally remove older checkpoints here if desired

                # --- Barrier before next step? ---
                # Might be useful if ranks can get very desynchronized,
                # e.g., due to I/O waits or errors on some ranks.
                # if self.ddp and step % 100 == 0: # Example: sync every 100 steps
                #    dist.barrier()

            # --- End of training loop ---

        except KeyboardInterrupt:
            logger.warning(f"Rank {self.rank}: Training interrupted by user at step {step}.")
            # Perform cleanup if needed? Barrier?
        finally:
            if self.is_rank_zero:
                pbar.close()
                # Use the value of 'step' from the last completed iteration
                last_processed_step = step if "step" in locals() else start_step - 1
                print(f"\nTraining loop finished. Last step processed: {last_processed_step}.")

        # --- Final Save Operations (Rank 0 Only) ---
        if self.is_rank_zero:
            final_model_path = os.path.join(self.log_dir, "clt_final.pt")
            print(f"Saving final checkpoint to {final_model_path}...")
            # Use the value of 'step' from the last completed iteration
            last_completed_step = step if "step" in locals() else start_step - 1
            # Save final checkpoint (includes model, optimizer, store state)
            if last_completed_step >= 0:  # Avoid saving checkpoint if no steps were completed
                self._save_checkpoint(last_completed_step)  # Use last completed step index
            else:
                logger.warning("No training steps completed. Final checkpoint not saved.")

            # Log final model artifact to WandB (optional, could log the checkpoint file)
            # Get underlying model state dict for final artifact logging
            # final_model_state = self.model.module.state_dict() if self.ddp else self.model.state_dict()
            # torch.save(final_model_state, final_model_path) # Save just state dict for artifact?
            # self.wandb_logger.log_artifact(artifact_path=final_model_path, artifact_type="model", name="clt_final_model_state")

            print("Saving final metrics...")
            self._save_metrics()

            # Finish WandB logging
            self.wandb_logger.finish()

            print(f"Training completed! Final checkpoint saved to {final_model_path}")

        # --- Final Barrier ---
        if self.ddp:
            dist.barrier()
            logger.info(f"Rank {self.rank}: Final barrier passed.")

        # Return the underlying model instance
        return self._get_underlying_model()


# Ensure cleanup of distributed group if initialized
# This might be better placed in the calling script (e.g., train_clt.py)
# def cleanup():
#     if dist.is_initialized():
#         dist.destroy_process_group()
#         logger.info("Destroyed distributed process group.")
#
# # Example of cleanup registration
# import atexit
# atexit.register(cleanup)
