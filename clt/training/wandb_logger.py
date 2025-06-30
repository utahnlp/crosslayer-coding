import wandb
import os
from typing import Dict, Optional, Any
import logging
from dataclasses import asdict

from clt.config import CLTConfig, TrainingConfig

logger = logging.getLogger(__name__)


# Define the dummy logger class explicitly for better type checking
class DummyWandBLogger:
    _run_id: Optional[str] = None

    def __init__(self, *args, **kwargs):
        pass

    def log_step(self, *args, **kwargs):
        pass

    def log_evaluation(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def get_current_wandb_run_id(self) -> Optional[str]:
        return None


class WandBLogger:
    """Wrapper class for Weights & Biases logging."""

    _run_id: Optional[str] = None
    wandb_run: Optional[Any]

    def __init__(
        self,
        training_config: TrainingConfig,
        clt_config: CLTConfig,
        log_dir: str,
        resume_wandb_id: Optional[str] = None,
    ):
        """Initialize the WandB logger.

        Args:
            training_config: Training configuration
            clt_config: CLT model configuration
            log_dir: Directory to save logs
            resume_wandb_id: Optional WandB run ID to resume a previous run.
        """
        self.enabled = training_config.enable_wandb
        self.log_dir = log_dir
        self.resume_wandb_id = resume_wandb_id

        if not self.enabled:
            logger.info("WandB logging is disabled by training_config.enable_wandb=False.")
            self.wandb_run = None
            return

        try:
            # Try to get project from env var first, then from config
            project_name = os.environ.get("WANDB_PROJECT", training_config.wandb_project)
            entity_name = os.environ.get("WANDB_ENTITY", training_config.wandb_entity)
            run_name = training_config.wandb_run_name  # Can be None

            # Prepare config for both new and resumed runs
            wandb_config = self._create_wandb_config(training_config, clt_config)
            
            if self.resume_wandb_id:
                logger.info(f"Attempting to resume WandB run with ID: {self.resume_wandb_id}")
                # When resuming, we need to update the config after init
                self.wandb_run = wandb.init(
                    id=self.resume_wandb_id, 
                    resume="allow",
                    project=project_name,
                    entity=entity_name,
                    tags=training_config.wandb_tags,
                )
                # Update config after resuming
                if self.wandb_run:
                    wandb.config.update(wandb_config, allow_val_change=True)
            else:
                self.wandb_run = wandb.init(
                    project=project_name,
                    entity=entity_name,
                    name=run_name,
                    config=wandb_config,
                    tags=training_config.wandb_tags,
                )

            if self.wandb_run:
                self._run_id = self.wandb_run.id
                logger.info(f"WandB logging initialized: {self.wandb_run.name} (ID: {self.wandb_run.id})")
                # Log config keys to verify they were set (only in debug mode)
                if logger.isEnabledFor(logging.DEBUG):
                    if hasattr(wandb, 'config') and wandb.config:
                        config_keys = list(wandb.config.keys())
                        logger.debug(f"WandB config keys: {config_keys}")
                    else:
                        logger.debug("WandB config appears to be empty or not accessible")
            else:
                logger.warning("Warning: WandB run initialization failed but no exception was raised.")

        except ImportError:
            logger.warning("wandb library not installed. Please install with `pip install wandb` to use WandB logging.")
            self.wandb_run = None
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")
            self.wandb_run = None

    def _create_wandb_config(self, training_config: TrainingConfig, clt_config: CLTConfig) -> Dict[str, Any]:
        config_dict = {
            "training_config": asdict(training_config),
            "clt_config": asdict(clt_config),
            "log_dir": self.log_dir,
        }
        return config_dict

    def get_current_wandb_run_id(self) -> Optional[str]:
        """Returns the current WandB run ID, if a run is active."""
        return self._run_id

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
            elif key == "auxiliary":
                metrics["training/auxiliary_loss"] = value
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
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Wandb: Error logging metrics at step {step}: {e}")

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
                                logger.error(f"Wandb: Error creating histogram for {wandb_key}: {e}")
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
                    logger.error(f"Wandb: Error creating aggregate histogram for {key}: {e}")
                    # Optional Fallback: log mean of aggregate data
                    try:
                        mean_val = sum(value) / len(value) if value else 0.0
                        wandb_log_dict[f"{key}_mean"] = mean_val
                    except TypeError:
                        wandb_log_dict[f"{key}_mean"] = -1.0

            elif isinstance(value, (float, int)):  # Handle top-level scalars
                # Log directly, e.g., 'reconstruction/mse', 'sparsity/avg_l0', 'dead_features/total_eval'
                wandb_log_dict[key] = value
            else:
                # Pass through other wandb-compatible objects such as wandb.Histogram directly
                # This ensures histograms prepared upstream (e.g., BatchTopK diagnostics) are logged.
                try:
                    # Attempt to reference wandb.Histogram to avoid circular import issues
                    import wandb as _wb

                    if isinstance(value, _wb.Histogram):
                        wandb_log_dict[key] = value
                    else:
                        # For any other types, attempt to add directly if they are serializable by wandb
                        wandb_log_dict[key] = value
                except Exception:
                    # If wandb import fails or value is not compatible, skip silently to avoid breaking logging
                    pass
            # Add other specific handling if needed (e.g., for specific non-scalar, non-layerwise data)

        # Log the prepared dictionary to wandb
        if wandb_log_dict:
            if self.wandb_run:
                try:
                    self.wandb_run.log(wandb_log_dict, step=step)
                except Exception as e:
                    logger.error(f"Wandb: Error logging metrics at step {step}: {e}")

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
        import os

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
