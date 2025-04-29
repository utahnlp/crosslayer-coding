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
import torch.distributed as dist  # Import torch.distributed
from torch.distributed import ProcessGroup  # Import ProcessGroup
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader  # Storage for checkpointing

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


# Define the dummy logger class explicitly for better type checking
class DummyWandBLogger:
    def log_step(self, *args, **kwargs):
        pass

    def log_evaluation(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class WandBLogger:
    """Wrapper class for Weights & Biases logging."""

    def __init__(self, training_config: TrainingConfig, clt_config: CLTConfig, log_dir: str):
        """Initialize the WandB logger.

        Args:
            training_config: Training configuration
            clt_config: CLT model configuration
            log_dir: Directory to save logs
        """
        self.enabled = training_config.enable_wandb
        self.log_dir = log_dir

        if not self.enabled:
            return

        # Check if wandb is installed
        if not importlib.util.find_spec("wandb"):
            print(
                "Warning: WandB logging requested but wandb not installed. "
                "Install with 'pip install wandb'. Continuing without WandB."
            )
            self.enabled = False
            return

        # Import wandb
        import wandb

        # Set up run name with timestamp if not provided
        run_name = training_config.wandb_run_name
        if run_name is None:
            run_name = f"clt-{time.strftime('%Y%m%d-%H%M%S')}"

        # Initialize wandb
        wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            name=run_name,
            dir=log_dir,
            tags=training_config.wandb_tags,
            config={
                **training_config.__dict__,
                **clt_config.__dict__,
                "log_dir": log_dir,
            },
        )

        if wandb.run is not None:
            print(f"WandB logging initialized: {wandb.run.name}")

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        lr: Optional[float] = None,
        sparsity_lambda: Optional[float] = None,
        total_tokens_processed: Optional[int] = None,
    ):
        """Log metrics for a training step under the 'training/' group.

        Args:
            step: Current training step
            loss_dict: Dictionary of loss values (e.g., total, reconstruction, sparsity)
            lr: Current learning rate
            sparsity_lambda: Current sparsity coefficient lambda
            total_tokens_processed: Total tokens processed up to this step
        """
        if not self.enabled:
            return

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
        if total_tokens_processed is not None:
            metrics["training/total_tokens_processed"] = total_tokens_processed

        # Log to wandb
        wandb.log(metrics, step=step)

    def log_evaluation(self, step: int, eval_metrics: Dict[str, Any]):
        """Log evaluation metrics, organized by the structure from CLTEvaluator.

        Args:
            step: Current training step
            eval_metrics: Dictionary of evaluation metrics from CLTEvaluator
                          (keys like 'reconstruction/', 'sparsity/', 'layerwise/')
        """
        if not self.enabled:
            return

        import wandb

        # Log metrics directly, assuming keys are already structured
        # e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'layerwise/l0/layer_0'
        wandb_log_dict: Dict[str, Any] = {}
        for key, value in eval_metrics.items():
            if key.startswith("layerwise/"):
                # Handle nested layerwise data (histograms and scalars)
                # layerwise_category = key.split("/")[
                #     1
                # ]  # e.g., 'l0', 'log_feature_density' # Removed unused variable
                if isinstance(value, dict):
                    for layer_key, layer_value in value.items():
                        # Construct wandb key: e.g., layerwise/l0/layer_0
                        wandb_key = f"{key}/{layer_key}"  # Correctly forms e.g. layerwise/log_feature_density/layer_0
                        if isinstance(layer_value, list):
                            # Log list data as histogram
                            try:
                                wandb_log_dict[wandb_key] = wandb.Histogram(layer_value)
                            except Exception as e:
                                print(f"Wandb: Error creating histogram for {wandb_key}: {e}")
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
                # Handle aggregate histogram data (e.g., sparsity/log_feature_density_agg_hist)
                try:
                    wandb_log_dict[key] = wandb.Histogram(value)
                except Exception as e:
                    print(f"Wandb: Error creating aggregate histogram for {key}: {e}")
                    # Optional Fallback: log mean of aggregate data
                    try:
                        mean_val = sum(value) / len(value) if value else 0.0
                        wandb_log_dict[f"{key}_mean"] = mean_val
                    except TypeError:
                        wandb_log_dict[f"{key}_mean"] = -1.0

            elif isinstance(value, (float, int)):  # Handle top-level scalars
                # Log directly, e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'dead_features/total_eval'
                wandb_log_dict[key] = value
            # Add other specific handling if needed (e.g., for specific non-scalar, non-layerwise data)

        # Log the prepared dictionary to wandb
        if wandb_log_dict:
            wandb.log(wandb_log_dict, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        """Log an artifact to WandB.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset")
            name: Name of the artifact (defaults to filename)
        """
        if not self.enabled:
            return

        import wandb

        # Use filename if name not provided
        if name is None:
            name = os.path.basename(artifact_path)

        # Create and log artifact
        artifact = wandb.Artifact(name=name, type=artifact_type)
        # Check if it's a directory (for sharded checkpoints)
        if os.path.isdir(artifact_path):
            artifact.add_dir(artifact_path)
        else:
            artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Finish the WandB run."""
        if not self.enabled:
            return

        import wandb

        wandb.finish()


class CLTTrainer:
    """Trainer for Cross-Layer Transcoder models."""

    # Add type hint for the activation store attribute
    activation_store: BaseActivationStore
    # Model type hint
    model: CrossLayerTranscoder

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,  # Add distributed flag
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training (ignored if distributed)
            distributed: Whether to use distributed training
        """
        self.clt_config = clt_config
        self.training_config = training_config
        self.distributed = distributed

        # Initialize distributed training if enabled
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.process_group: Optional[ProcessGroup] = None  # For tensor parallelism

        if self.distributed:
            if not dist.is_initialized():
                # Default backend, consider NCCL for NVIDIA GPUs
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))  # Get local rank if available

            # Set device based on local_rank when distributed
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.local_rank}")
                torch.cuda.set_device(self.device)
            else:
                # Fallback for CPU distributed testing (not typical)
                self.device = torch.device("cpu")
                logger.warning("Distributed training requested but CUDA not available. Using CPU.")
            # Set the process group for tensor parallelism (using WORLD for now)
            self.process_group = dist.group.WORLD
        else:
            # Original device handling for non-distributed case
            _device_input = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input
            # Process group is None when not distributed
            self.process_group = None

        # Set up log directory - only rank 0 creates it
        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

        # Record start time
        self.start_time = time.time()

        # Initialize model, passing device and process group for direct initialization
        # self.process_group is correctly set to None if not distributed
        self.model = CrossLayerTranscoder(
            clt_config, process_group=self.process_group, device=self.device  # Pass the potentially None group
        )

        # Initialize optimizer - works on local parameters
        # Explicitly type the kwargs dict for clarity and linting
        optimizer_kwargs: Dict[str, Any] = {"lr": training_config.learning_rate}
        beta1 = training_config.optimizer_beta1  # Could be None
        beta2 = training_config.optimizer_beta2  # Could be None

        # Only add 'betas' if at least one is specified
        if beta1 is not None or beta2 is not None:
            # Get defaults if one is None
            # Default Adam/AdamW betas are (0.9, 0.999)
            final_beta1 = beta1 if beta1 is not None else 0.9
            final_beta2 = beta2 if beta2 is not None else 0.999
            optimizer_kwargs["betas"] = (final_beta1, final_beta2)
            logger.info(f"Rank {self.rank}: Using optimizer betas: ({final_beta1}, {final_beta2})")

        if training_config.optimizer == "adam":
            self.optimizer: Any = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        else:  # "adamw"
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_kwargs)

        # Initialize scheduler
        self.scheduler: Optional[Any] = None
        scheduler_type = training_config.lr_scheduler
        # Get scheduler params from config, default to empty dict if None
        scheduler_params = training_config.lr_scheduler_params or {}

        if scheduler_type == "linear":
            # Default params for LinearLR
            default_linear_params = {
                "start_factor": 1.0,
                "end_factor": 0.1,
                # total_iters is always training_steps for this setup
            }
            # Update defaults with user-provided params
            final_params = {**default_linear_params, **scheduler_params}
            # Ensure total_iters is not overridden by user params
            final_params.pop("total_iters", None)

            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                total_iters=training_config.training_steps,
                **final_params,  # Pass start_factor, end_factor, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using LinearLR scheduler with params: {final_params}, total_iters={training_config.training_steps}"
            )

        elif scheduler_type == "cosine":
            # Default params for CosineAnnealingLR
            default_cosine_params = {
                # T_max defaults to training_steps
                "eta_min": 0,  # Default minimum LR
            }
            # Update defaults with user-provided params
            final_params = {**default_cosine_params, **scheduler_params}
            # Set T_max explicitly, allowing override but defaulting to training_steps
            t_max = final_params.pop("T_max", training_config.training_steps)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, **final_params  # Pass eta_min, etc.
            )
            logger.info(
                f"Rank {self.rank}: Using CosineAnnealingLR scheduler with params: {final_params}, T_max={t_max}"
            )

        elif scheduler_type == "linear_final20":
            # This scheduler keeps LR constant for the initial fraction of training
            # and then linearly decays it to 0 over the remaining steps (default 20%).
            # The fraction can be customized via lr_scheduler_params["decay_start_frac"].
            decay_start_frac = scheduler_params.get("decay_start_frac", 0.8)  # 0.8 means last 20% decays
            assert 0.0 < decay_start_frac < 1.0, "decay_start_frac must be between 0 and 1"
            total_steps = training_config.training_steps
            decay_start_step = int(decay_start_frac * total_steps)

            def lr_lambda(current_step: int):
                if current_step < decay_start_step:
                    return 1.0  # Keep LR constant
                # Linearly decay from 1 -> 0 over the remaining steps
                remaining = total_steps - current_step
                decay_steps = total_steps - decay_start_step
                return max(remaining / decay_steps, 0.0)

            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info(
                "Rank %d: Using linear_final20 LR scheduler with decay_start_frac=%s (start step %d of %d)",
                self.rank,
                decay_start_frac,
                decay_start_step,
                total_steps,
            )

        # Add elif blocks here for other potential schedulers

        # Initialize activation store based on config - uses self.rank/world_size now
        self.activation_store = self._create_activation_store(self.start_time)

        # Pass normalisation statistics (if available) so the loss can be computed in
        # the *original* scale even when inputs/targets are stored normalised.
        mean_tg_stats = getattr(self.activation_store, "mean_tg", {})  # type: ignore[arg-type]
        std_tg_stats = getattr(self.activation_store, "std_tg", {})  # type: ignore[arg-type]

        self.loss_manager = LossManager(
            training_config,
            mean_tg=mean_tg_stats,
            std_tg=std_tg_stats,
        )

        # Initialize Evaluator - Pass norm stats here too
        self.evaluator = CLTEvaluator(
            model=self.model,
            device=self.device,
            start_time=self.start_time,
            mean_tg=mean_tg_stats,  # Pass the same stats
            std_tg=std_tg_stats,  # Pass the same stats
        )

        # Initialize dead neuron counters (replicated for now, consider sharding later if needed)
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
        )

        # Training metrics (only rank 0 saves, but others might need local copies for some logic)
        self.metrics: Dict[str, list] = {
            "train_losses": [],
            "eval_metrics": [],
        }

        # Initialize WandB logger - only on rank 0
        if not self.distributed or self.rank == 0:
            self.wandb_logger: Union[WandBLogger, DummyWandBLogger] = WandBLogger(
                training_config=training_config, clt_config=clt_config, log_dir=self.log_dir
            )
        else:
            # Dummy logger for non-rank-0 processes
            self.wandb_logger = DummyWandBLogger()

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window."""
        # Ensure counter is initialized
        if not hasattr(self, "n_forward_passes_since_fired") or self.n_forward_passes_since_fired is None:
            # Return an all-false mask if counter doesn't exist yet
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        return self.n_forward_passes_since_fired > self.training_config.dead_feature_window

    def _create_activation_store(self, start_time: float) -> BaseActivationStore:
        """Create the appropriate activation store based on training config.

        Valid activation_source values:
        - "generate": Use StreamingActivationStore with on-the-fly generation.
        - "local_manifest": Use LocalActivationStore with local manifest/chunks.
        - "remote": Use RemoteActivationStore with remote server.

        Args:
            start_time: The training start time for elapsed time logging.

        Returns:
            Configured instance of a BaseActivationStore subclass.
        """
        activation_source = self.training_config.activation_source
        sampling_strategy = self.training_config.sampling_strategy

        shard_data = False  # add a flag if you like
        row_rank = 0 if shard_data is False else self.rank
        row_world = 1 if shard_data is False else self.world_size

        # Temporary fix to stop sharding data
        rank = row_rank
        world = row_world

        # Fix: Declare store with Base type hint
        store: BaseActivationStore

        if activation_source == "generate":
            if self.distributed:
                # Activation generation with NNsight doesn't easily support multi-GPU generation
                # without more complex setup (e.g., one rank generates, others receive).
                # For simplicity, disable generation in distributed mode for now.
                raise NotImplementedError(
                    "On-the-fly activation generation ('generate') is not currently supported "
                    "in distributed training mode. Please pre-generate activations using "
                    "'local_manifest' or 'remote' source."
                )

            logger.info("Using StreamingActivationStore (generating on-the-fly).")
            # --- Validate required config dicts --- #
            gen_cfg = self.training_config.generation_config
            ds_params = self.training_config.dataset_params
            if gen_cfg is None or ds_params is None:
                raise ValueError(
                    "generation_config and dataset_params must be provided in TrainingConfig for on-the-fly generation."
                )

            # --- Create Extractor from generation_config --- #
            extractor = ActivationExtractorCLT(
                model_name=gen_cfg["model_name"],  # Required
                mlp_input_module_path_template=gen_cfg["mlp_input_template"],  # Required
                mlp_output_module_path_template=gen_cfg["mlp_output_template"],  # Required
                device=self.device,  # Trainer manages device
                model_dtype=gen_cfg.get("model_dtype"),
                context_size=gen_cfg.get("context_size", 128),
                inference_batch_size=gen_cfg.get("inference_batch_size", 512),
                exclude_special_tokens=gen_cfg.get("exclude_special_tokens", True),
                prepend_bos=gen_cfg.get("prepend_bos", False),
                nnsight_tracer_kwargs=gen_cfg.get("nnsight_tracer_kwargs"),
                nnsight_invoker_args=gen_cfg.get("nnsight_invoker_args"),
            )

            # --- Create Generator from dataset_params --- #
            activation_generator = extractor.stream_activations(
                dataset_path=ds_params["dataset_path"],  # Required
                dataset_split=ds_params.get("dataset_split", "train"),
                dataset_text_column=ds_params.get("dataset_text_column", "text"),
                streaming=ds_params.get("streaming", True),
                dataset_trust_remote_code=ds_params.get("dataset_trust_remote_code", False),
                cache_path=ds_params.get("cache_path"),
                # max_samples=ds_params.get("max_samples"), # max_samples is not a valid arg for stream_activations
            )

            # --- Create Streaming Store --- #
            stream_norm_method = self.training_config.normalization_method
            if stream_norm_method == "auto":
                stream_norm_method = "estimated_mean_std"

            store = StreamingActivationStore(
                activation_generator=activation_generator,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                n_batches_in_buffer=self.training_config.n_batches_in_buffer,
                normalization_method=str(stream_norm_method),
                normalization_estimation_batches=(self.training_config.normalization_estimation_batches),
                device=self.device,
                start_time=start_time,
            )
            logger.info("Initialized StreamingActivationStore.")
            logger.info(f"  Normalization method: {store.normalization_method}")
            logger.info("  Uses internal estimation, DOES NOT use norm_stats.json.")
        # Use the new LocalActivationStore for manifest-based local datasets
        elif activation_source == "local_manifest":
            logger.info(f"Rank {rank}: Using LocalActivationStore (reading local manifest/chunks).")
            if not self.training_config.activation_path:
                raise ValueError(
                    "activation_path must be set in TrainingConfig when activation_source is 'local_manifest'."
                )
            store = LocalActivationStore(
                dataset_path=self.training_config.activation_path,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                device=self.device,
                dtype=self.training_config.activation_dtype,  # Use explicit dtype from config
                rank=rank,  # Use trainer's rank
                world=world,  # Use trainer's world size
                seed=self.training_config.seed,
                sampling_strategy=sampling_strategy,
                normalization_method=self.training_config.normalization_method,
            )
            # Fix: Check instance type before accessing subclass-specific attributes
            if isinstance(store, LocalActivationStore):
                logger.info(f"Rank {rank}: Initialized LocalActivationStore from path: {store.dataset_path}")
                if store.apply_normalization:
                    logger.info(f"Rank {rank}:   Normalization ENABLED using loaded norm_stats.json.")
                else:
                    # Make the warning more generic as the file might be loaded but processing failed
                    logger.warning(
                        f"Rank {rank}:   Normalization DISABLED (processing failed or file incomplete/invalid)."
                    )
        elif activation_source == "remote":
            logger.info(f"Rank {rank}: Using RemoteActivationStore (remote slice server).")
            remote_cfg = self.training_config.remote_config
            if remote_cfg is None:
                raise ValueError("remote_config dict must be set in TrainingConfig when activation_source is 'remote'.")
            server_url = remote_cfg.get("server_url")
            dataset_id = remote_cfg.get("dataset_id")
            if not server_url or not dataset_id:
                raise ValueError("remote_config must contain 'server_url' and 'dataset_id'.")

            store = RemoteActivationStore(
                server_url=server_url,
                dataset_id=dataset_id,
                train_batch_size_tokens=self.training_config.train_batch_size_tokens,
                device=self.device,
                dtype=self.training_config.activation_dtype,  # Use explicit dtype from config
                rank=rank,  # Use trainer's rank
                world=world,  # Use trainer's world size
                seed=self.training_config.seed,
                timeout=remote_cfg.get("timeout", 60),
                sampling_strategy=sampling_strategy,
            )
            # Fix: Check instance type before accessing subclass-specific attributes
            if isinstance(store, RemoteActivationStore):
                logger.info(f"Rank {rank}: Initialized RemoteActivationStore for dataset: {store.did_raw}")
                if store.apply_normalization:
                    logger.info(f"Rank {rank}:   Normalization ENABLED using fetched norm_stats.json.")
                else:
                    logger.warning(
                        f"Rank {rank}:   Normalization DISABLED (norm_stats.json not found on server or failed to load)."
                    )
        else:
            raise ValueError(
                f"Unknown activation_source: {activation_source}. Valid options: 'generate', 'local_manifest', 'remote'."
            )

        return store

    def _log_metrics(self, step: int, loss_dict: Dict[str, float]):
        """Log training metrics, including current LR and lambda. Only Rank 0 logs to WandB/file."""

        # Add step to training loss record (for saving to JSON) - all ranks might need this locally?
        # Let's keep it local for now. Rank 0 will save the full history.
        self.metrics["train_losses"].append({"step": step, **loss_dict})

        # Only log to WandB/File from rank 0
        if not self.distributed or self.rank == 0:
            # --- Gather additional metrics for logging --- #
            current_lr = None
            if self.scheduler is not None:
                # Assuming one parameter group
                current_lr = self.scheduler.get_last_lr()[0]

            current_lambda = self.loss_manager.get_current_sparsity_lambda()

            # Calculate total tokens processed (global perspective)
            total_tokens_processed = self.training_config.train_batch_size_tokens * self.world_size * (step + 1)

            # Use the wandb_logger which handles the dummy case
            self.wandb_logger.log_step(
                step,
                loss_dict,
                lr=current_lr,
                sparsity_lambda=current_lambda,
                total_tokens_processed=total_tokens_processed,
            )

            # --- Save metrics periodically --- #
            log_interval = self.training_config.log_interval
            if step % log_interval == 0:
                self._save_metrics()

    def _save_metrics(self):
        """Save training metrics to disk - only on rank 0 when distributed."""
        if self.distributed and self.rank != 0:
            return

        metrics_path = os.path.join(self.log_dir, "metrics.json")
        try:
            with open(metrics_path, "w") as f:
                # Use default=str to handle potential non-serializable types
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save metrics to {metrics_path}: {e}")

    def _save_checkpoint(self, step: int):
        """Save a distributed checkpoint of the model and activation store state.

        Uses torch.distributed.checkpoint to save sharded state directly.

        Args:
            step: Current training step
        """
        if not self.distributed:  # Non-distributed save
            os.makedirs(self.log_dir, exist_ok=True)
            model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.pt")
            latest_model_path = os.path.join(self.log_dir, "clt_checkpoint_latest.pt")
            store_checkpoint_path = os.path.join(self.log_dir, f"activation_store_checkpoint_{step}.pt")
            latest_store_path = os.path.join(self.log_dir, "activation_store_checkpoint_latest.pt")

            try:
                # In non-distributed, model state_dict is the full dict
                torch.save(self.model.state_dict(), model_checkpoint_path)
                torch.save(self.model.state_dict(), latest_model_path)
                # Log checkpoint as artifact to WandB
                self.wandb_logger.log_artifact(
                    artifact_path=model_checkpoint_path, artifact_type="model", name=f"clt_checkpoint_{step}"
                )
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                print(f"Warning: Failed to save non-distributed checkpoint at step {step}: {e}")
            return

        # --- Distributed Save ---
        # Define checkpoint directory for this step
        checkpoint_dir = os.path.join(self.log_dir, f"step_{step}")
        latest_checkpoint_dir = os.path.join(self.log_dir, "latest")

        # Save model state dict using distributed checkpointing
        # All ranks participate in saving their shard
        try:
            model_state_dict = self.model.state_dict()
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,  # Ensure distributed save
            )
            # Also save latest (overwrites previous latest) - maybe link instead?
            # For simplicity, save again. Rank 0 can handle linking later if needed.
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(latest_checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,
            )
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save distributed model checkpoint at step {step}: {e}")

        # Save activation store state (only rank 0)
        if self.rank == 0:
            store_checkpoint_path = os.path.join(checkpoint_dir, "activation_store.pt")  # Save inside step dir
            latest_store_path = os.path.join(latest_checkpoint_dir, "activation_store.pt")
            try:
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save activation store state at step {step}: {e}")

            # Log checkpoint directory as artifact to WandB (only rank 0)
            self.wandb_logger.log_artifact(
                artifact_path=checkpoint_dir,  # Log the directory
                artifact_type="model_checkpoint",
                name=f"dist_checkpoint_{step}",
            )

        # Barrier to ensure all ranks finish saving before proceeding
        dist.barrier()

    def load_checkpoint(self, checkpoint_path: str):
        """Load a distributed checkpoint for model and activation store state.

        Args:
            checkpoint_path: Path to the *directory* containing the sharded checkpoint.
                             This should be the directory saved by _save_checkpoint (e.g., .../step_N).
        """
        if not os.path.isdir(checkpoint_path):
            print(
                f"Error: Checkpoint path {checkpoint_path} is not a directory. Distributed checkpoints are saved as directories."
            )
            # Check if it's a non-distributed .pt file for backward compatibility or single GPU runs
            if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt") and not self.distributed:
                print("Attempting to load as non-distributed checkpoint file...")
                self._load_non_distributed_checkpoint(checkpoint_path)
                return
            else:
                return

        if not self.distributed:
            print("Attempting to load a distributed checkpoint directory in non-distributed mode.")
            print("Loading only rank 0 state from the directory (if possible)...")
            # Attempt to load rank 0 shard if structure is known, or just load store state
            # model_state_dict: Dict[str, Any] = {} # Placeholder - not used
            # Try loading rank 0 shard - needs specific knowledge of file structure inside dir
            # load_state_dict(model_state_dict, storage_reader=FileSystemReader(checkpoint_path)) # This might not work easily for single shard
            try:
                # Create an empty state dict matching the *full* model structure
                # Ensure the model is on the correct device before getting state_dict
                self.model.to(self.device)
                state_dict_to_load = self.model.state_dict()  # Non-dist model has full state dict
                load_state_dict(
                    state_dict=state_dict_to_load,
                    storage_reader=FileSystemReader(checkpoint_path),
                    planner=DefaultLoadPlanner(),  # Standard planner might reconstruct
                    no_dist=True,  # Key flag: Treat the checkpoint as containing a full state dict
                )
                # state_dict_to_load is modified in-place by load_state_dict
                print(
                    f"Successfully loaded and reconstructed full model state from sharded checkpoint {checkpoint_path}"
                )
            except Exception as e:
                print(f"Error loading distributed checkpoint into non-distributed model from {checkpoint_path}: {e}")
                print(
                    "This might indicate that the checkpoint format requires manual reconstruction or saving the full state dict separately on rank 0 during distributed save."
                )
                # Fallback or error handling could go here
                return  # Stop if model load failed

            # Load activation store (if exists)
            store_checkpoint_path = os.path.join(checkpoint_path, "activation_store.pt")
            if os.path.exists(store_checkpoint_path):
                try:
                    store_state = torch.load(store_checkpoint_path, map_location=self.device)
                    if hasattr(self, "activation_store") and self.activation_store is not None:
                        self.activation_store.load_state_dict(store_state)
                        print(f"Loaded activation store state from {store_checkpoint_path}")
                    else:
                        print("Warning: Activation store not initialized. Cannot load state.")
                except Exception as e:
                    print(f"Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
            else:
                print(f"Warning: Activation store checkpoint not found in {checkpoint_path}")
            # Model loading in non-dist from dist checkpoint is complex, skipping full implementation here.
            print("Warning: Loading sharded model parameters into non-distributed model is not fully implemented.")
            return

        # --- Distributed Load ---
        # Load model state dict using distributed checkpointing
        try:
            # Create an empty state dict matching the *local* shard structure
            state_dict_to_load = self.model.state_dict()
            load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=FileSystemReader(checkpoint_path),
                planner=DefaultLoadPlanner(),
                no_dist=False,  # Ensure distributed load
            )
            # state_dict_to_load is modified in-place
            print(f"Rank {self.rank}: Loaded distributed model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Rank {self.rank}: Error loading distributed model checkpoint from {checkpoint_path}: {e}")
            # Barrier even on error? Maybe not, let ranks fail individually.
            return

        # Load activation store state (only rank 0 needs it, but load happens on rank 0)
        if self.rank == 0:
            store_checkpoint_path = os.path.join(checkpoint_path, "activation_store.pt")
            if os.path.exists(store_checkpoint_path):
                try:
                    store_state = torch.load(store_checkpoint_path, map_location=self.device)  # Load to rank 0's device
                    if hasattr(self, "activation_store") and self.activation_store is not None:
                        self.activation_store.load_state_dict(store_state)
                        print(f"Rank 0: Loaded activation store state from {store_checkpoint_path}")
                    else:
                        print("Rank 0: Warning: Activation store not initialized. Cannot load state.")
                except Exception as e:
                    print(f"Rank 0: Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
            else:
                print(f"Rank 0: Warning: Activation store checkpoint not found in {checkpoint_path}")

        # Barrier to ensure all ranks have loaded before proceeding
        dist.barrier()

    # Helper for backward compatibility / non-distributed runs
    def _load_non_distributed_checkpoint(self, checkpoint_path: str, store_checkpoint_path: Optional[str] = None):
        """Loads a standard single-file model checkpoint."""
        if self.distributed:
            print("Error: Attempting to load non-distributed checkpoint in distributed mode.")
            return

        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            return
        try:
            full_state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(full_state_dict)  # Assumes model has standard load_state_dict
            print(f"Loaded non-distributed model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading non-distributed model checkpoint from {checkpoint_path}: {e}")
            return

        # Determine and load store checkpoint path (same logic as before)
        if store_checkpoint_path is None:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            if basename.startswith("clt_checkpoint_"):
                store_basename = basename.replace("clt_checkpoint_", "activation_store_checkpoint_")
                store_checkpoint_path = os.path.join(dirname, store_basename)
            elif basename == "clt_checkpoint_latest.pt":
                store_checkpoint_path = os.path.join(dirname, "activation_store_checkpoint_latest.pt")
            else:
                store_checkpoint_path = None

        if store_checkpoint_path and os.path.exists(store_checkpoint_path):
            try:
                store_state = torch.load(store_checkpoint_path, map_location=self.device)
                if hasattr(self, "activation_store") and self.activation_store is not None:
                    self.activation_store.load_state_dict(store_state)
                    print(f"Loaded activation store state from {store_checkpoint_path}")
                else:
                    print("Warning: Activation store not initialized. Cannot load state.")
            except Exception as e:
                print(f"Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
        else:
            print(
                f"Warning: Activation store checkpoint path not found or specified: {store_checkpoint_path}. Store state not loaded."
            )

    def _average_shared_parameter_grads(self):
        """Average gradients of parameters that are **replicated** across ranks.

        Tensor-parallel layers shard their weights so those gradients must **not** be
        synchronised.  However parameters that are kept identical on every rank –
        e.g. the JumpReLU `log_threshold` vector (shape `[num_features]`) and any
        unsharded bias vectors – must have their gradients reduced or they will
        diverge between ranks.
        """
        if not self.distributed or self.world_size == 1:
            return

        for p in self.model.parameters():
            if p.grad is None:
                continue
            if p.dim() == 1:  # replicated vectors
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= self.world_size

    def train(self, eval_every: int = 1000) -> CrossLayerTranscoder:
        """Train the CLT model.

        Args:
            eval_every: Evaluate model every N steps

        Returns:
            Trained CLT model (local shard)
        """
        # Print startup message from rank 0 only
        if not self.distributed or self.rank == 0:
            print(f"Starting CLT training on {self.device}...")
            print(
                f"Model has {self.clt_config.num_features} features per layer "
                f"and {self.clt_config.num_layers} layers"
            )
            print(f"Training for {self.training_config.training_steps} steps.")
            print(f"Logging to {self.log_dir}")
            if self.distributed:
                print(f"Distributed training with {self.world_size} processes (Tensor Parallelism)")

            # Check if using normalization and notify user
            if self.training_config.normalization_method == "estimated_mean_std":
                print("\n>>> NORMALIZATION PHASE <<<")
                print("Normalization statistics are being estimated from dataset activations.")
                print("This may take some time, but happens only once before training begins.")
                print(f"Using {self.training_config.normalization_estimation_batches} batches for estimation.\n")

            # Make sure we flush stdout to ensure prints appear immediately,
            # especially important in Jupyter/interactive environments
            sys.stdout.flush()
            # Wait for 1 second to ensure output is displayed before training starts
            time.sleep(1)
            print("\n>>> TRAINING PHASE <<<")
            sys.stdout.flush()

        # After the existing startup messages
        if self.distributed:
            print("\n!!! DIAGNOSTIC INFO !!!")
            print(f"Rank {self.rank}: Process group type: {type(self.process_group)}")
            print(f"Rank {self.rank}: RowParallelLinear _reduce does NOT divide by world_size")
            print(f"Rank {self.rank}: Using weight regularization in sparsity penalty")
            print(f"Rank {self.rank}: Averaging replicated parameter gradients")
            # Check if activation store has rank/world attributes before accessing
            store_rank = getattr(self.activation_store, "rank", "N/A")
            store_world = getattr(self.activation_store, "world", "N/A")
            print(f"Rank {self.rank}: Data sharding: rank={store_rank}, world={store_world}")
            print(f"Rank {self.rank}: Batch size tokens: {self.training_config.train_batch_size_tokens}")
            print(f"Rank {self.rank}: Sparsity lambda: {self.training_config.sparsity_lambda}")

            # Check if activation store actually loaded correctly
            batch_avail = next(iter(self.activation_store), None)
            print(f"Rank {self.rank}: First batch available: {batch_avail is not None}")

            # Force torch to compile/execute our code by running a tiny forward/backward pass
            dummy = torch.ones(1, device=self.device, requires_grad=True)
            dummy_out = dummy * 2
            dummy_out.backward()
            print("!!! END DIAGNOSTIC !!!\n")

        # Create progress bar only on rank 0
        pbar: Union[tqdm, range]
        if not self.distributed or self.rank == 0:
            pbar = tqdm(
                range(self.training_config.training_steps),
                desc="Training CLT",
                leave=True,
            )
        else:
            pbar = range(self.training_config.training_steps)

        step = 0
        try:
            for step in pbar:
                # Refresh progress bar on rank 0
                if isinstance(pbar, tqdm):
                    pbar.refresh()

                try:
                    # Get batch directly from the iterator (handles distributed sampling internally)
                    inputs, targets = next(self.activation_store)

                except StopIteration:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print("Activation store exhausted. Training finished early.")
                    if self.distributed:
                        dist.barrier()  # Ensure all ranks see this
                    break  # Exit training loop if data runs out
                except Exception as e:
                    # Rank 0 prints message
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Error getting batch at step {step}: {e}. Skipping step.")
                    # Maybe barrier here too? If one rank fails, others might hang?
                    # Let's continue for now, assuming store handles internal errors.
                    continue

                # --- Check for empty batch --- (Optional but good practice)
                # This check should ideally happen *before* moving data potentially
                if not inputs or not targets or not any(v.numel() > 0 for v in inputs.values()):
                    if not self.distributed or self.rank == 0:
                        print(f"\nRank {self.rank}: Warning: Received empty batch at step {step}. Skipping.")
                    continue

                # --- BEGIN: One-time Normalization Check ---
                if step == 0 and (not self.distributed or self.rank == 0):
                    logger.info("--- Running Post-Normalization Check (First Batch) ---")
                    norm_applied = getattr(self.activation_store, "apply_normalization", None)
                    if isinstance(self.activation_store, (LocalActivationStore, RemoteActivationStore)):
                        logger.info(f"ActivationStore reports apply_normalization={norm_applied}")
                    elif isinstance(self.activation_store, StreamingActivationStore):
                        logger.info(
                            f"Streaming store normalization method: {self.activation_store.normalization_method}"
                        )

                    for li in range(self.clt_config.num_layers):
                        mean_in, std_in, mean_tg, std_tg = float("nan"), float("nan"), float("nan"), float("nan")
                        try:
                            if li in inputs and inputs[li].numel() > 0:
                                input_tensor = inputs[li].float()
                                mean_in = input_tensor.mean().item()
                                std_in = input_tensor.std().item()
                            if li in targets and targets[li].numel() > 0:
                                target_tensor = targets[li].float()
                                mean_tg = target_tensor.mean().item()
                                std_tg = target_tensor.std().item()

                            if not (
                                torch.isnan(torch.tensor(mean_in)) and torch.isnan(torch.tensor(mean_tg))
                            ):  # Log if at least one value is valid
                                logger.info(
                                    f"  Layer {li:>2}: Input Mean={mean_in:+.4f}, Std={std_in:.4f} | Target Mean={mean_tg:+.4f}, Std={std_tg:.4f}"
                                )
                        except Exception as e:
                            logger.error(f"  Layer {li}: Error during normalization check: {e}")
                    logger.info("--- End Post-Normalization Check ---")
                # --- END: One-time Normalization Check ---

                # --- Forward pass and compute loss --- (All ranks)
                self.optimizer.zero_grad()

                # Compute feature activations **once** per step to avoid redundant encoder forward passes.
                feature_activations_batch = self.model.get_feature_activations(inputs)

                # Compute total loss using the pre-computed activations
                loss, loss_dict = self.loss_manager.compute_total_loss(
                    self.model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
                    precomputed_activations=feature_activations_batch,
                )

                # --- Update Dead Neuron Counters --- (All ranks, counter is replicated)
                # We need *full* feature activations *after* non-linearity
                if hasattr(self, "n_forward_passes_since_fired"):
                    with torch.no_grad():
                        for layer_idx, layer_acts in feature_activations_batch.items():
                            # Ensure layer index is within bounds of the counter tensor
                            if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                if layer_acts.numel() > 0:
                                    # layer_acts shape: [batch_tokens, num_features]
                                    fired_mask_per_token = layer_acts > 1e-6
                                    fired_features_this_layer = fired_mask_per_token.any(dim=0)

                                    if fired_features_this_layer.shape[0] == self.n_forward_passes_since_fired.shape[1]:
                                        self.n_forward_passes_since_fired[layer_idx] += 1
                                        self.n_forward_passes_since_fired[layer_idx][fired_features_this_layer] = 0
                                    else:
                                        if not self.distributed or self.rank == 0:  # Only rank 0 logs warning
                                            print(
                                                f"Rank {self.rank}: Warning: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                                f"Acts shape: {layer_acts.shape}, Fired mask: {fired_features_this_layer.shape}, "
                                                f"Counter: {self.n_forward_passes_since_fired.shape}"
                                            )

                # --- Backward pass --- (All ranks, handles communication implicitly)
                if torch.isnan(loss):
                    if not self.distributed or self.rank == 0:
                        print(
                            f"\nRank {self.rank}: Warning: NaN loss encountered at step {step}. "
                            f"Skipping backward pass and optimizer step."
                        )
                else:
                    try:
                        loss.backward()

                        # --- Synchronise gradients of replicated parameters --- #
                        self._average_shared_parameter_grads()

                        # --- Gradient clipping --- #
                        if self.training_config.gradient_clip_val is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_val,
                            )
                    except RuntimeError as e:
                        if not self.distributed or self.rank == 0:
                            print(
                                f"\nRank {self.rank}: Error during backward pass at step {step}: {e}. Skipping optimizer step."
                            )
                        continue

                    # --- Optimizer step --- (Applied to local parameters using local gradients)
                    self.optimizer.step()

                    # --- Invalidate Caches --- #
                    if hasattr(self.model, "_cached_decoder_norms"):
                        self.model._cached_decoder_norms = None

                # --- Scheduler step --- (All ranks)
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar --- (Rank 0 only)
                if isinstance(pbar, tqdm):
                    description = (
                        f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                        f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                        f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                        f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    )
                    pbar.set_description(description)
                    # Force update to display progress
                    if step % 1 == 0:  # Update every step
                        pbar.refresh()
                        sys.stdout.flush()

                # --- Log metrics --- (Rank 0 logs to WandB/file)
                self._log_metrics(step, loss_dict)

                # --- Evaluation & Checkpointing ---
                eval_interval = self.training_config.eval_interval
                checkpoint_interval = self.training_config.checkpoint_interval

                save_checkpoint_flag = (step > 0 and step % checkpoint_interval == 0) or (  # Avoid checkpoint at step 0
                    step == self.training_config.training_steps - 1
                )
                run_eval_flag = (step % eval_interval == 0) or (step == self.training_config.training_steps - 1)

                # --- Evaluation (all ranks participate to match collectives) ---
                # In tensor-parallel mode the model forward includes collective ops (all_reduce/all_gather).
                # If only rank 0 performed the forward pass these collectives would block on the other ranks
                # resulting in NCCL timeouts.  Therefore, *every* rank must execute the evaluation forward pass.
                # We still only log / store the resulting metrics on rank 0.
                if run_eval_flag:
                    if self.distributed:
                        dist.barrier()  # Sync before evaluation starts so that all ranks enter together

                    # Compute evaluation metrics on all ranks to keep collective ops aligned
                    current_dead_mask = self.dead_neurons_mask.detach().clone()
                    eval_metrics = self.evaluator.compute_metrics(
                        inputs,
                        targets,
                        dead_neuron_mask=current_dead_mask,
                    )

                    if not self.distributed or self.rank == 0:
                        # Store evaluation metrics (for saving to JSON)
                        self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                        # --- Update Progress Bar Postfix ---
                        l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                        ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"
                        avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                        dens_str = f"Dens: {avg_density_mean:.3f}" if avg_density_mean is not None else "Dens: N/A"
                        eval_dead_str = f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                        eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                        if isinstance(pbar, tqdm):
                            pbar.set_postfix_str(eval_msg)
                            pbar.refresh()

                        # --- Log evaluation metrics to WandB ---
                        self.wandb_logger.log_evaluation(step, eval_metrics)

                        # --- Save metrics JSON after evaluation ---
                        self._save_metrics()

                    # Optionally compute and log sparsity diagnostics (can be slow)
                    if self.training_config.compute_sparsity_diagnostics:
                        # Calculate diagnostics using the same batch data and cached activations/norms
                        sparsity_diag_metrics = self._compute_sparsity_diagnostics(inputs, feature_activations_batch)
                        # Merge diagnostics into the main eval metrics dict
                        if sparsity_diag_metrics:
                            eval_metrics.update(sparsity_diag_metrics)
                            # Log updated metrics to WandB (only rank 0)
                            if not self.distributed or self.rank == 0:
                                self.wandb_logger.log_evaluation(step, eval_metrics)

                    # Ensure all ranks finish evaluation before proceeding
                    if self.distributed:
                        dist.barrier()

                # --- Checkpointing (All ranks participate) ---
                if save_checkpoint_flag:
                    self._save_checkpoint(step)

            # --- Explicitly delete tensors at the very end of the loop iteration --- #
            # Do this on all ranks
            try:
                del inputs
                del targets
                if "loss" in locals() and loss is not None:
                    del loss
                if "feature_activations_batch" in locals():
                    del feature_activations_batch
            except NameError:
                pass

        except KeyboardInterrupt:
            if not self.distributed or self.rank == 0:
                print("\nTraining interrupted by user.")
        finally:
            if isinstance(pbar, tqdm):
                pbar.close()
            if not self.distributed or self.rank == 0:
                print(f"Training loop finished at step {step}.")

        # Sync before final save attempt
        if self.distributed:
            dist.barrier()

        # --- Save final model and metrics --- (Rank 0 handles metrics/store, all ranks save model state)
        final_checkpoint_dir = os.path.join(self.log_dir, "final")
        final_store_path = os.path.join(final_checkpoint_dir, "activation_store_final.pt")  # Store inside final dir

        # All ranks save final model state
        try:
            final_model_state_dict = self.model.state_dict()
            save_state_dict(
                state_dict=final_model_state_dict,
                storage_writer=FileSystemWriter(final_checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=(not self.distributed),  # Disable distributed save if not distributed
            )
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save final distributed model state: {e}")

        # Rank 0 saves store, metrics, logs artifact
        if not self.distributed or self.rank == 0:
            print(f"Saving final activation store state to {final_store_path}...")
            os.makedirs(final_checkpoint_dir, exist_ok=True)  # Ensure dir exists for store save
            try:
                torch.save(self.activation_store.state_dict(), final_store_path)
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save final activation store state: {e}")

            print("Saving final metrics...")
            self._save_metrics()

            # Log final checkpoint directory as artifact
            self.wandb_logger.log_artifact(artifact_path=final_checkpoint_dir, artifact_type="model", name="clt_final")

            # Finish WandB logging
            self.wandb_logger.finish()
            print(f"Training completed! Final checkpoint saved to {final_checkpoint_dir}")

        # Clean up distributed process group
        if self.distributed:
            dist.destroy_process_group()

        return self.model

    # --- Helper method for optional, potentially slow diagnostics --- #
    @torch.no_grad()
    def _compute_sparsity_diagnostics(
        self, inputs: Dict[int, torch.Tensor], feature_activations: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
        """Computes detailed sparsity diagnostics (z-scores, tanh saturation, etc.).

        Args:
            inputs: Dictionary of input activations (used implicitly by get_decoder_norms).
            feature_activations: Dictionary of feature activations (pre-computed).

        Returns:
            Dictionary containing sparsity diagnostic metrics.
        """
        diag_metrics: Dict[str, Any] = {}
        layerwise_z_median: Dict[str, float] = {}
        layerwise_z_p90: Dict[str, float] = {}
        layerwise_mean_tanh: Dict[str, float] = {}
        layerwise_sat_frac: Dict[str, float] = {}
        layerwise_mean_abs_act: Dict[str, float] = {}
        layerwise_mean_dec_norm: Dict[str, float] = {}

        all_layer_medians = []
        all_layer_p90s = []
        all_layer_mean_tanhs = []
        all_layer_sat_fracs = []
        all_layer_abs_act = []
        all_layer_dec_norm = []

        sparsity_c = self.training_config.sparsity_c

        # Norms should be cached from the loss calculation earlier in the step
        diag_dec_norms = self.model.get_decoder_norms()  # [L, F]

        for l_idx, layer_acts in feature_activations.items():
            if layer_acts.numel() == 0:
                layer_key = f"layer_{l_idx}"
                layerwise_z_median[layer_key] = float("nan")
                layerwise_z_p90[layer_key] = float("nan")
                layerwise_mean_tanh[layer_key] = float("nan")
                layerwise_sat_frac[layer_key] = float("nan")
                continue

            # Ensure norms and activations are compatible and on the same device
            norms_l = diag_dec_norms[l_idx].to(layer_acts.device, layer_acts.dtype).unsqueeze(0)  # [1, F]
            layer_acts = layer_acts.to(norms_l.device, norms_l.dtype)

            z = sparsity_c * norms_l * layer_acts  # [tokens, F]
            on_mask = layer_acts > 1e-6  # Use a small threshold > 0
            z_on = z[on_mask]

            if z_on.numel() > 0:
                med = torch.median(z_on).item()
                p90 = torch.quantile(z_on, 0.9).item()
                tanh_z_on = torch.tanh(z_on)
                mean_tanh = tanh_z_on.mean().item()
                sat_frac = (tanh_z_on > 0.99).float().mean().item()
            else:
                # If no features were active, assign NaN or 0
                med, p90, mean_tanh, sat_frac = float("nan"), float("nan"), float("nan"), float("nan")

            layer_key = f"layer_{l_idx}"
            layerwise_z_median[layer_key] = med
            layerwise_z_p90[layer_key] = p90
            layerwise_mean_tanh[layer_key] = mean_tanh
            layerwise_sat_frac[layer_key] = sat_frac

            # --- Additional diagnostics: mean |a| and decoder norm ---
            mean_abs_act_val = layer_acts.abs().mean().item() if layer_acts.numel() > 0 else float("nan")
            mean_dec_norm_val = diag_dec_norms[l_idx].mean().item()
            layerwise_mean_abs_act.setdefault(f"layer_{l_idx}", mean_abs_act_val)
            layerwise_mean_dec_norm.setdefault(f"layer_{l_idx}", mean_dec_norm_val)
            if not torch.isnan(torch.tensor(mean_abs_act_val)):
                all_layer_abs_act.append(mean_abs_act_val)
            if not torch.isnan(torch.tensor(mean_dec_norm_val)):
                all_layer_dec_norm.append(mean_dec_norm_val)

            # Add to lists for aggregation (skip NaNs)
            if not torch.isnan(torch.tensor(med)):
                all_layer_medians.append(med)
            if not torch.isnan(torch.tensor(p90)):
                all_layer_p90s.append(p90)
            if not torch.isnan(torch.tensor(mean_tanh)):
                all_layer_mean_tanhs.append(mean_tanh)
            if not torch.isnan(torch.tensor(sat_frac)):
                all_layer_sat_fracs.append(sat_frac)

        # Calculate aggregate metrics (average of per-layer values)
        agg_z_median = torch.tensor(all_layer_medians).mean().item() if all_layer_medians else float("nan")
        agg_z_p90 = torch.tensor(all_layer_p90s).mean().item() if all_layer_p90s else float("nan")
        agg_mean_tanh = torch.tensor(all_layer_mean_tanhs).mean().item() if all_layer_mean_tanhs else float("nan")
        agg_sat_frac = torch.tensor(all_layer_sat_fracs).mean().item() if all_layer_sat_fracs else float("nan")
        agg_mean_abs_act = torch.tensor(all_layer_abs_act).mean().item() if all_layer_abs_act else float("nan")
        agg_mean_dec_norm = torch.tensor(all_layer_dec_norm).mean().item() if all_layer_dec_norm else float("nan")

        # --- Populate diagnostics dictionary --- #
        # Layerwise dictionaries
        diag_metrics["layerwise/sparsity_z_median"] = layerwise_z_median
        diag_metrics["layerwise/sparsity_z_p90"] = layerwise_z_p90
        diag_metrics["layerwise/sparsity_mean_tanh"] = layerwise_mean_tanh
        diag_metrics["layerwise/sparsity_sat_frac"] = layerwise_sat_frac
        diag_metrics["layerwise/mean_abs_activation"] = layerwise_mean_abs_act
        diag_metrics["layerwise/mean_decoder_norm"] = layerwise_mean_dec_norm
        # Aggregate scalars
        diag_metrics["sparsity/z_median_agg"] = agg_z_median
        diag_metrics["sparsity/z_p90_agg"] = agg_z_p90
        diag_metrics["sparsity/mean_tanh_agg"] = agg_mean_tanh
        diag_metrics["sparsity/sat_frac_agg"] = agg_sat_frac
        diag_metrics["sparsity/mean_abs_activation_agg"] = agg_mean_abs_act
        diag_metrics["sparsity/mean_decoder_norm_agg"] = agg_mean_dec_norm

        # Clean up large tensors (optional, but good practice)
        del diag_dec_norms
        del z
        del z_on
        del on_mask
        del tanh_z_on

        return diag_metrics
