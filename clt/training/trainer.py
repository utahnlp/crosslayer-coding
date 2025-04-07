import torch
import torch.optim as optim
from typing import Dict, Optional, Union, Any
from tqdm import tqdm  # type: ignore
import os
import json
import time
import importlib.util

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data import ActivationStore
from clt.training.losses import LossManager
from clt.nnsight.extractor import ActivationExtractorCLT


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
        self, step: int, loss_dict: Dict[str, float], lr: Optional[float] = None
    ):
        """Log metrics for a training step.

        Args:
            step: Current training step
            loss_dict: Dictionary of loss values
            lr: Current learning rate
        """
        if not self.enabled:
            return

        import wandb

        # Create metrics dict with 'train/' prefix for losses
        metrics = {f"train/{k}": v for k, v in loss_dict.items()}

        # Add learning rate if provided
        if lr is not None:
            metrics["train/learning_rate"] = lr

        # Log to wandb
        wandb.log(metrics, step=step)

    def log_evaluation(self, step: int, l0_stats: Dict[str, Any]):
        """Log evaluation metrics.

        Args:
            step: Current training step
            l0_stats: Dictionary of L0 statistics
        """
        if not self.enabled:
            return

        import wandb

        # Create metrics dict
        metrics = {
            "eval/avg_l0": l0_stats.get("avg_l0", 0.0),
            "eval/total_l0": l0_stats.get("total_l0", 0.0),
            "eval/sparsity": l0_stats.get("sparsity", 1.0),
        }

        # Add per-layer metrics if available
        per_layer = l0_stats.get("per_layer", {})
        if isinstance(per_layer, dict):
            for layer_name, l0_value in per_layer.items():
                metrics[f"eval/l0/{layer_name}"] = l0_value

        # Log to wandb
        wandb.log(metrics, step=step)

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

        # Initialize model
        self.model = CrossLayerTranscoder(clt_config).to(self.device)

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

        # Training metrics
        self.metrics: Dict[str, list] = {
            "train_losses": [],
            "l0_stats": [],
            "eval_metrics": [],
        }

        # Initialize WandB logger
        self.wandb_logger = WandBLogger(
            training_config=training_config, clt_config=clt_config, log_dir=self.log_dir
        )

    def _create_activation_extractor(self) -> ActivationExtractorCLT:
        """Create an activation extractor based on training config.

        Returns:
            Configured ActivationExtractorCLT instance
        """
        return ActivationExtractorCLT(
            model_name=self.training_config.model_name,
            device=self.device,
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
        """Log training metrics.

        Args:
            step: Current training step
            loss_dict: Dictionary of loss values
        """
        # Add to metrics
        self.metrics["train_losses"].append({"step": step, **loss_dict})

        # Get current learning rate if scheduler is used
        current_lr = None
        if self.scheduler is not None:
            current_lr = self.scheduler.get_last_lr()[0]

        # Log to WandB
        self.wandb_logger.log_step(step, loss_dict, lr=current_lr)

        # Log less frequently for potentially large runs
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

    def _compute_l0(
        self, inputs: Dict[int, torch.Tensor]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute L0 stats: avg active features per token, summed across layers.

        Args:
            inputs: Dictionary mapping layer indices to input activations from the
                    current training batch.

        Returns:
            Dictionary of L0 statistics:
                - total_l0: Average active features per token, summed across layers.
                - avg_l0: Average of the per-layer L0s (total_l0 / num_layers).
                - sparsity: Overall sparsity (1 - total_l0 / total_possible_features).
                - per_layer: Dictionary mapping layer index to its avg active features
                  per token.
        """
        # Ensure inputs are valid
        if not inputs or not any(v.numel() > 0 for v in inputs.values()):
            print(
                "Warning: Received empty or invalid inputs for L0 computation. "
                "Returning zeros."
            )
            return {
                "total_l0": 0.0,
                "avg_l0": 0.0,
                "sparsity": 1.0,
                "per_layer": {
                    f"layer_{i}": 0.0 for i in range(self.clt_config.num_layers)
                },
            }

        # Ensure inputs are on the correct device
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            print(
                f"Error moving inputs to device in L0 computation: {e}. "
                f"Returning zeros."
            )
            return {
                "total_l0": 0.0,
                "avg_l0": 0.0,
                "sparsity": 1.0,
                "per_layer": {
                    f"layer_{i}": 0.0 for i in range(self.clt_config.num_layers)
                },
            }

        # Get activations
        with torch.no_grad():
            activations = self.model.get_feature_activations(inputs)

        per_layer_l0 = {}
        total_l0 = 0.0  # Sum of per-layer average active features per token
        num_valid_layers = 0

        # Ensure activations keys match expected layers if possible
        expected_layers = set(range(self.clt_config.num_layers))
        actual_layers = set(activations.keys())
        if expected_layers != actual_layers:
            print(
                f"Warning: Mismatch between expected layers ({expected_layers}) "
                f"and activation keys ({actual_layers}) during L0 computation."
            )

        # Iterate through layers present in activations
        for layer_idx, layer_activations in activations.items():
            # layer_activations shape: [num_tokens, num_features]
            if layer_activations.numel() == 0 or layer_activations.shape[0] == 0:
                per_layer_l0[f"layer_{layer_idx}"] = 0.0
                print(
                    f"Warning: Empty activations for layer {layer_idx} in L0 "
                    f"computation."
                )
                continue

            # 1. Count active features for EACH token (sum along feature dim)
            # Shape: [num_tokens]
            active_count_per_token = (layer_activations != 0).float().sum(dim=-1)

            # 2. Average this count across tokens for this layer
            avg_active_this_layer = active_count_per_token.mean().item()

            per_layer_l0[f"layer_{layer_idx}"] = avg_active_this_layer
            total_l0 += avg_active_this_layer
            num_valid_layers += 1

        # Calculate average L0 across layers
        avg_l0 = total_l0 / num_valid_layers if num_valid_layers > 0 else 0.0

        # Calculate overall sparsity
        total_possible_features = (
            self.clt_config.num_layers * self.clt_config.num_features
        )
        sparsity = (
            1.0 - (total_l0 / total_possible_features)
            if total_possible_features > 0
            else 1.0
        )
        # Clamp sparsity between 0 and 1
        sparsity = max(0.0, min(1.0, sparsity))

        return {
            "total_l0": total_l0,  # Avg active features/token summed across layers
            "avg_l0": avg_l0,  # Average of per-layer L0s
            "sparsity": sparsity,
            "per_layer": per_layer_l0,  # Avg active features/token per layer
        }

    def _save_checkpoint(self, step: int):
        """Save a checkpoint of the model and activation store state.

        Args:
            step: Current training step
        """
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Save model checkpoint
        model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.pt")
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
            self.model = self.model.to(self.device)
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
        import sys

        sys.stdout.flush()

        # Wait for 1 second to ensure output is displayed before training starts
        time.sleep(1)

        # Training loop using ActivationStore as iterator
        print("\n>>> TRAINING PHASE <<<")
        sys.stdout.flush()

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
                    inputs, targets = next(self.activation_store)
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
                loss, loss_dict = self.loss_manager.compute_total_loss(
                    self.model,
                    inputs,
                    targets,
                    step,
                    self.training_config.training_steps,
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
                        loss.backward()
                    except RuntimeError as e:
                        print(
                            f"\nError during backward pass at step {step}: {e}. "
                            f"Skipping optimizer step."
                        )
                        # Potentially inspect gradients here if debugging is needed
                        continue  # Skip optimizer step if backward fails

                    # --- Optimizer step ---
                    self.optimizer.step()

                # --- Scheduler step ---
                if self.scheduler:
                    self.scheduler.step()

                # --- Update progress bar ---
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

                # --- Log metrics ---
                self._log_metrics(step, loss_dict)

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
                    # Pass the current input batch to _compute_l0
                    l0_stats = self._compute_l0(inputs)
                    self.metrics["l0_stats"].append({"step": step, **l0_stats})
                    l0_msg = (
                        f"Layerwise L0: {l0_stats['avg_l0']:.2f} "
                        f"Total L0: {l0_stats['total_l0']:.2f} "
                        f"(Spar: {l0_stats['sparsity']:.3f})"
                    )
                    pbar.set_postfix_str(l0_msg)
                    pbar.refresh()  # Force update

                    # Log evaluation metrics to WandB
                    self.wandb_logger.log_evaluation(step, l0_stats)

                    # Save metrics after evaluation
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
