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

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data import ActivationStore
from clt.training.losses import LossManager
from clt.nnsight.extractor import ActivationExtractorCLT
from .evaluator import CLTEvaluator  # Import the new evaluator

# Get logger for this module
logger = logging.getLogger(__name__)


class WandBLogger:
    """Wrapper class for Weights & Biases logging."""

    def __init__(
        self, training_config: TrainingConfig, clt_config: CLTConfig, log_dir: str
    ):
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
    ):
        """Log metrics for a training step under the 'training/' group.

        Args:
            step: Current training step
            loss_dict: Dictionary of loss values (e.g., total, reconstruction, sparsity)
            lr: Current learning rate
            sparsity_lambda: Current sparsity coefficient lambda
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
                                print(
                                    f"Wandb: Error creating histogram for {wandb_key}: {e}"
                                )
                                # Fallback: log mean or placeholder
                                try:
                                    mean_val = (
                                        sum(layer_value) / len(layer_value)
                                        if layer_value
                                        else 0.0
                                    )
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

    def log_artifact(
        self, artifact_path: str, artifact_type: str, name: Optional[str] = None
    ):
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

    def __init__(
        self,
        clt_config: CLTConfig,
        training_config: TrainingConfig,
        log_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the CLT trainer.

        Args:
            clt_config: Configuration for the CLT model
            training_config: Configuration for training
            log_dir: Directory to save logs and checkpoints
            device: Device to use for training
        """
        self.clt_config = clt_config
        self.training_config = training_config

        # Ensure self.device is a torch.device object
        _device_input = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = (
            torch.device(_device_input)
            if isinstance(_device_input, str)
            else _device_input
        )

        # Set up log directory
        self.log_dir = log_dir or f"clt_train_{int(time.time())}"
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model, passing device for direct initialization
        self.model = CrossLayerTranscoder(clt_config, device=self.device)

        # Initialize optimizer
        if training_config.optimizer == "adam":
            self.optimizer: Any = optim.Adam(
                self.model.parameters(), lr=training_config.learning_rate
            )
        else:  # "adamw"
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=training_config.learning_rate
            )

        # Initialize scheduler
        self.scheduler: Optional[Any] = None
        if training_config.lr_scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=training_config.training_steps,
            )
        elif training_config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=training_config.training_steps
            )

        # Initialize activation extractor and store
        self.activation_extractor = self._create_activation_extractor()
        self.activation_store = self._create_activation_store()

        # Initialize loss manager
        self.loss_manager = LossManager(training_config)

        # Initialize Evaluator
        self.evaluator = CLTEvaluator(self.model, self.device)

        # Initialize dead neuron counters
        self.n_forward_passes_since_fired = torch.zeros(
            (clt_config.num_layers, clt_config.num_features),
            device=self.device,
            dtype=torch.long,
        )

        # Training metrics
        self.metrics: Dict[str, list] = {
            "train_losses": [],
            "eval_metrics": [],
        }

        # Initialize WandB logger
        self.wandb_logger = WandBLogger(
            training_config=training_config, clt_config=clt_config, log_dir=self.log_dir
        )

    @property
    def dead_neurons_mask(self) -> torch.Tensor:
        """Boolean mask indicating dead neurons based on inactivity window."""
        # Ensure counter is initialized
        if (
            not hasattr(self, "n_forward_passes_since_fired")
            or self.n_forward_passes_since_fired is None
        ):
            # Return an all-false mask if counter doesn't exist yet
            return torch.zeros(
                (self.clt_config.num_layers, self.clt_config.num_features),
                dtype=torch.bool,
                device=self.device,
            )
        return (
            self.n_forward_passes_since_fired > self.training_config.dead_feature_window
        )

    def _create_activation_extractor(self) -> ActivationExtractorCLT:
        """Create an activation extractor based on training config.

        Returns:
            Configured ActivationExtractorCLT instance
        """
        return ActivationExtractorCLT(
            model_name=self.training_config.model_name,
            device=self.device,
            model_dtype=self.training_config.model_dtype,
            context_size=self.training_config.context_size,
            store_batch_size_prompts=self.training_config.store_batch_size_prompts,
            exclude_special_tokens=self.training_config.exclude_special_tokens,
            prepend_bos=self.training_config.prepend_bos,
        )

    def _create_activation_store(self) -> ActivationStore:
        """Create an activation store based on training config using the new extractor.

        Returns:
            Configured ActivationStore instance
        """
        # Create generator from the extractor
        activation_generator = self.activation_extractor.stream_activations(
            dataset_path=self.training_config.dataset_path,
            dataset_split=self.training_config.dataset_split,
            dataset_text_column=self.training_config.dataset_text_column,
            streaming=self.training_config.streaming,
            dataset_trust_remote_code=self.training_config.dataset_trust_remote_code,
            cache_path=self.training_config.cache_path,
            max_samples=getattr(self.training_config, "max_samples", None),
        )

        # Create the store with the generator
        store = ActivationStore(
            activation_generator=activation_generator,
            n_batches_in_buffer=self.training_config.n_batches_in_buffer,
            train_batch_size_tokens=self.training_config.train_batch_size_tokens,
            normalization_method=self.training_config.normalization_method,
            normalization_estimation_batches=(
                self.training_config.normalization_estimation_batches
            ),
            device=self.device,
        )

        return store

    def _log_metrics(self, step: int, loss_dict: Dict[str, float]):
        """Log training metrics, including current LR and lambda.

        Args:
            step: Current training step
            loss_dict: Dictionary of loss values from LossManager
        """
        # Add step to training loss record (for saving to JSON)
        self.metrics["train_losses"].append({"step": step, **loss_dict})

        # --- Gather additional metrics for logging --- #
        current_lr = None
        if self.scheduler is not None:
            # Assuming one parameter group
            current_lr = self.scheduler.get_last_lr()[0]

        current_lambda = self.loss_manager.get_current_sparsity_lambda()

        # --- Log to WandB --- #
        # Note: Dead features from trainer window are no longer logged here.
        # We use the dead feature count calculated during evaluation step.
        self.wandb_logger.log_step(
            step,
            loss_dict,
            lr=current_lr,
            sparsity_lambda=current_lambda,
        )

        # --- Save metrics periodically --- #
        log_interval = self.training_config.log_interval
        if step % log_interval == 0:
            self._save_metrics()

    def _save_metrics(self):
        """Save training metrics to disk."""
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        try:
            with open(metrics_path, "w") as f:
                # Use default=str to handle potential non-serializable types
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save metrics to {metrics_path}: {e}")

    def _save_checkpoint(self, step: int):
        """Save a checkpoint of the model and activation store state.

        Args:
            step: Current training step
        """
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Save model checkpoint
        model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.pt")
        mem_before_save = 0
        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_before_save = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.debug(
                f"Checkpoint Step {step} - Before Save. Mem: {mem_before_save:.2f} MB"
            )

        try:
            self.model.save(model_checkpoint_path)
            # Log checkpoint as artifact to WandB
            self.wandb_logger.log_artifact(
                artifact_path=model_checkpoint_path,
                artifact_type="model",
                name=f"clt_checkpoint_{step}",
            )
        except Exception as e:
            print(
                f"Warning: Failed to save model checkpoint to "
                f"{model_checkpoint_path}: {e}"
            )

        # Save activation store state
        store_checkpoint_path = os.path.join(
            self.log_dir, f"activation_store_checkpoint_{step}.pt"
        )
        try:
            torch.save(self.activation_store.state_dict(), store_checkpoint_path)
        except Exception as e:
            print(
                f"Warning: Failed to save activation store state to "
                f"{store_checkpoint_path}: {e}"
            )

        # Also save a copy as latest
        latest_model_path = os.path.join(self.log_dir, "clt_checkpoint_latest.pt")
        latest_store_path = os.path.join(
            self.log_dir, "activation_store_checkpoint_latest.pt"
        )
        try:
            self.model.save(latest_model_path)
        except Exception as e:
            print(f"Warning: Failed to save latest model checkpoint: {e}")
        try:
            torch.save(self.activation_store.state_dict(), latest_store_path)
        except Exception as e:
            print(f"Warning: Failed to save latest activation store state: {e}")

        if torch.cuda.is_available() and self.device.type == "cuda":
            mem_after_save = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.debug(
                f"Checkpoint Step {step} - After Save. Mem: {mem_after_save:.2f} MB (+{mem_after_save - mem_before_save:.2f} MB)"
            )

    def load_checkpoint(
        self, checkpoint_path: str, store_checkpoint_path: Optional[str] = None
    ):
        """Load model and activation store checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            store_checkpoint_path: Path to activation store checkpoint
                (if None, derived from checkpoint_path)
        """
        # Load model checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            return
        try:
            self.model.load(checkpoint_path)
            print(f"Loaded model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model checkpoint from {checkpoint_path}: {e}")
            return  # Don't proceed if model load fails

        # Determine store checkpoint path if not provided
        if store_checkpoint_path is None:
            # Try to derive from model checkpoint path
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            if basename.startswith("clt_checkpoint_"):
                # Replace "clt_checkpoint_" with "activation_store_checkpoint_"
                store_basename = basename.replace(
                    "clt_checkpoint_", "activation_store_checkpoint_"
                )
                store_checkpoint_path = os.path.join(dirname, store_basename)
            else:
                # Try using _latest suffix if loading latest model
                if basename == "clt_checkpoint_latest.pt":
                    store_checkpoint_path = os.path.join(
                        dirname, "activation_store_checkpoint_latest.pt"
                    )

        # Load activation store checkpoint if available
        if store_checkpoint_path and os.path.exists(store_checkpoint_path):
            try:
                store_state = torch.load(
                    store_checkpoint_path, map_location=self.device
                )
                # Ensure activation_store is initialized before loading state
                if (
                    not hasattr(self, "activation_store")
                    or self.activation_store is None
                ):
                    print(
                        "Warning: Activation store not initialized. Cannot load state."
                    )
                else:
                    self.activation_store.load_state_dict(store_state)
                    print(f"Loaded activation store state from {store_checkpoint_path}")
            except Exception as e:
                print(
                    f"Warning: Failed to load activation store state from "
                    f"{store_checkpoint_path}: {e}"
                )
        else:
            print(
                f"Warning: Activation store checkpoint path not found or specified: "
                f"{store_checkpoint_path}. Store state not loaded."
            )

    def train(self, eval_every: int = 1000) -> CrossLayerTranscoder:
        """Train the CLT model.

        Args:
            eval_every: Evaluate model every N steps

        Returns:
            Trained CLT model
        """
        print(f"Starting CLT training on {self.device}...")
        # Access num_features and num_layers via clt_config
        print(
            f"Model has {self.clt_config.num_features} features per layer "
            f"and {self.clt_config.num_layers} layers"
        )
        print(f"Training for {self.training_config.training_steps} steps.")
        print(f"Logging to {self.log_dir}")

        # Check if using normalization and notify user
        if self.training_config.normalization_method == "estimated_mean_std":
            print("\n>>> NORMALIZATION PHASE <<<")
            print(
                "Normalization statistics are being estimated from dataset activations."
            )
            print(
                "This may take some time, but happens only once before training begins."
            )
            print(
                f"Using {self.training_config.normalization_estimation_batches} batches for estimation.\n"
            )

        # Make sure we flush stdout to ensure prints appear immediately,
        # especially important in Jupyter/interactive environments
        sys.stdout.flush()

        # Wait for 1 second to ensure output is displayed before training starts
        time.sleep(1)

        # Training loop using ActivationStore as iterator
        print("\n>>> TRAINING PHASE <<<")
        sys.stdout.flush()
        if torch.cuda.is_available() and self.device.type == "cuda":
            logger.info(
                f"Training Start Mem: {torch.cuda.memory_allocated(self.device) / (1024**2):.2f} MB"
            )

        # Use tqdm to create progress bar for training
        pbar = tqdm(
            range(self.training_config.training_steps),
            desc="Training CLT",
            leave=True,  # Keep progress bar after completion
        )

        step = 0
        try:
            for step in pbar:
                # Force display update of progress bar
                pbar.refresh()

                try:
                    # Get batch directly from the iterator
                    # mem_before_get_batch = 0 # REMOVED
                    # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                    #      mem_before_get_batch = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED

                    inputs, targets = next(self.activation_store)

                    # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                    #     mem_after_get_batch = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED
                    #     logger.debug(f"Step {step} - After get_batch: Mem {mem_after_get_batch:.2f} MB (+{mem_after_get_batch - mem_before_get_batch:.2f} MB)") # REMOVED

                except StopIteration:
                    print("Activation store exhausted. Training finished early.")
                    break  # Exit training loop if data runs out
                except Exception as e:
                    print(f"\nError getting batch at step {step}: {e}. Skipping step.")
                    continue  # Skip this step if batch fetching fails

                # --- Check for empty batch --- (Optional but good practice)
                if (
                    not inputs
                    or not targets
                    or not any(v.numel() > 0 for v in inputs.values())
                ):
                    print(f"\nWarning: Received empty batch at step {step}. Skipping.")
                    continue

                # --- Forward pass and compute loss ---
                self.optimizer.zero_grad()
                # Loss manager needs the model, inputs, targets, and step info
                # mem_before_forward = 0 # REMOVED
                # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                #     mem_before_forward = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED

                loss, loss_dict = self.loss_manager.compute_total_loss(
                    self.model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
                )

                # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                #     mem_after_forward = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED
                #     logger.debug(f"Step {step} - After forward/loss: Mem {mem_after_forward:.2f} MB (+{mem_after_forward - mem_before_forward:.2f} MB)") # REMOVED

                # --- Update Dead Neuron Counters ---
                # We need feature activations *after* non-linearity
                if hasattr(self, "n_forward_passes_since_fired"):
                    with torch.no_grad():
                        # Get feature activations (post-nonlinearity) for the current batch
                        # Assuming inputs is the dictionary of layer activations fed to the model
                        feature_activations_batch = self.model.get_feature_activations(
                            inputs
                        )

                        for layer_idx, layer_acts in feature_activations_batch.items():
                            # Ensure layer index is within bounds of the counter tensor
                            if layer_idx < self.n_forward_passes_since_fired.shape[0]:
                                if layer_acts.numel() > 0:
                                    # layer_acts shape: [batch_tokens, num_features]
                                    # Check which features fired (activation > threshold)
                                    fired_mask_per_token = (
                                        layer_acts > 1e-6
                                    )  # Shape: [batch_tokens, num_features]
                                    fired_features_this_layer = (
                                        fired_mask_per_token.any(dim=0)
                                    )  # Shape: [num_features]

                                    # Ensure fired_features mask matches counter dimension
                                    if (
                                        fired_features_this_layer.shape[0]
                                        == self.n_forward_passes_since_fired.shape[1]
                                    ):
                                        # Increment counters for all features in this layer
                                        self.n_forward_passes_since_fired[
                                            layer_idx
                                        ] += 1
                                        # Reset counters for features that fired
                                        self.n_forward_passes_since_fired[layer_idx][
                                            fired_features_this_layer
                                        ] = 0
                                    else:
                                        print(
                                            f"Warning: Shape mismatch for dead neuron update at layer {layer_idx}. "
                                            f"Activations shape: {layer_acts.shape}, Fired mask shape: {fired_features_this_layer.shape}, "
                                            f"Counter shape: {self.n_forward_passes_since_fired.shape}"
                                        )

                # --- Backward pass ---
                if torch.isnan(loss):
                    print(
                        f"\nWarning: NaN loss encountered at step {step}. "
                        f"Skipping backward pass and optimizer step."
                    )
                    # Optionally log more details or raise an error
                else:
                    try:
                        # mem_before_backward = 0 # REMOVED
                        # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                        #     mem_before_backward = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED

                        loss.backward()

                        # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                        #     mem_after_backward = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED
                        #     logger.debug(f"Step {step} - After backward: Mem {mem_after_backward:.2f} MB (+{mem_after_backward - mem_before_backward:.2f} MB)") # REMOVED

                    except RuntimeError as e:
                        print(
                            f"\nError during backward pass at step {step}: {e}. "
                            f"Skipping optimizer step."
                        )
                        # Potentially inspect gradients here if debugging is needed
                        continue  # Skip optimizer step if backward fails

                    # --- Optimizer step ---
                    # mem_before_optimizer = 0 # REMOVED
                    # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                    #     mem_before_optimizer = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED

                    self.optimizer.step()

                    # if torch.cuda.is_available() and self.device.type == "cuda": # REMOVED
                    #     mem_after_optimizer = torch.cuda.memory_allocated(self.device) / (1024**2) # REMOVED
                    #     logger.debug(f"Step {step} - After optimizer: Mem {mem_after_optimizer:.2f} MB (+{mem_after_optimizer - mem_before_optimizer:.2f} MB)") # REMOVED

                # --- Scheduler step ---
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar ---
                # Include total dead features from training step log dict
                # dead_str = f"Dead: {loss_dict.get('sparsity/total_dead_features', 0)}" # Removed
                description = (
                    f"Loss: {loss_dict.get('total', float('nan')):.4f} "
                    f"(R: {loss_dict.get('reconstruction', float('nan')):.4f} "
                    f"S: {loss_dict.get('sparsity', float('nan')):.4f} "
                    f"P: {loss_dict.get('preactivation', float('nan')):.4f})"
                    # f"{dead_str}"  # Removed
                )
                pbar.set_description(description)

                # Force update to display progress
                if step % 1 == 0:  # Update every step
                    pbar.refresh()
                    sys.stdout.flush()

                # --- Log metrics --- # Simplified memory logging here
                self._log_metrics(step, loss_dict)
                if torch.cuda.is_available() and self.device.type == "cuda":
                    mem_end_step = torch.cuda.memory_allocated(self.device) / (1024**2)
                    logger.debug(f"Step {step} - End. Mem: {mem_end_step:.2f} MB")

                # --- Evaluation & Checkpointing ---
                eval_interval = self.training_config.eval_interval
                checkpoint_interval = self.training_config.checkpoint_interval

                save_checkpoint_flag = (step % checkpoint_interval == 0) or (
                    step == self.training_config.training_steps - 1
                )
                run_eval_flag = (step % eval_interval == 0) or (
                    step == self.training_config.training_steps - 1
                )

                if run_eval_flag:
                    # Detach mask for evaluator, which runs with no_grad
                    current_dead_mask = self.dead_neurons_mask.detach().clone()

                    # Use the evaluator to compute metrics, passing the mask
                    eval_metrics = self.evaluator.compute_metrics(
                        inputs,
                        targets,
                        dead_neuron_mask=current_dead_mask,  # Pass the mask
                    )

                    # Store evaluation metrics (for saving to JSON)
                    self.metrics["eval_metrics"].append({"step": step, **eval_metrics})

                    # --- Update Progress Bar Postfix --- #
                    l0_str = f"AvgL0: {eval_metrics.get('sparsity/avg_l0', 0.0):.2f}"
                    ev_str = f"EV: {eval_metrics.get('reconstruction/explained_variance', 0.0):.3f}"

                    # Use the pre-calculated aggregate density if available
                    avg_density_mean = eval_metrics.get("sparsity/feature_density_mean")
                    dens_str = (
                        f"Dens: {avg_density_mean:.3f}"
                        if avg_density_mean is not None
                        else "Dens: N/A"
                    )

                    # Add total dead features from evaluation metrics
                    eval_dead_str = (
                        f"Dead(Eval): {eval_metrics.get('dead_features/total_eval', 0)}"
                    )

                    # Note: Lambda is logged during training step
                    eval_msg = f"{l0_str}, {ev_str}, {dens_str}, {eval_dead_str}"

                    pbar.set_postfix_str(eval_msg)
                    pbar.refresh()  # Force update

                    # --- Log evaluation metrics to WandB --- #
                    # The eval_metrics dict already has the desired structure
                    self.wandb_logger.log_evaluation(step, eval_metrics)

                    # --- Save metrics JSON after evaluation --- #
                    self._save_metrics()

                if save_checkpoint_flag:
                    self._save_checkpoint(step)
                    # Optionally remove older checkpoints here if desired

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            pbar.close()
            print(f"Training loop finished at step {step}.")

        # --- Save final model and metrics ---
        final_path = os.path.join(self.log_dir, "clt_final.pt")
        final_store_path = os.path.join(self.log_dir, "activation_store_final.pt")

        print(f"Saving final model to {final_path}...")
        try:
            self.model.save(final_path)
            # Log final model as artifact to WandB
            self.wandb_logger.log_artifact(
                artifact_path=final_path, artifact_type="model", name="clt_final"
            )
        except Exception as e:
            print(f"Warning: Failed to save final model: {e}")

        print(f"Saving final activation store state to {final_store_path}...")
        try:
            torch.save(self.activation_store.state_dict(), final_store_path)
        except Exception as e:
            print(f"Warning: Failed to save final activation store state: {e}")

        print("Saving final metrics...")
        self._save_metrics()

        # Finish WandB logging
        self.wandb_logger.finish()

        print(f"Training completed! Final model attempted save to {final_path}")
        return self.model
