import time
import importlib.util
from typing import Dict, Optional, Any

from clt.config import CLTConfig, TrainingConfig  # Assuming clt.config is accessible


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
        wandb_init_kwargs = {
            "project": training_config.wandb_project,
            "entity": training_config.wandb_entity,
            "name": run_name,
            "dir": log_dir,
            "tags": training_config.wandb_tags,
            "config": {
                **clt_config.__dict__,
                **training_config.__dict__,
                "log_dir": log_dir,
            },
        }

        if resume_wandb_id:
            wandb_init_kwargs["id"] = resume_wandb_id
            wandb_init_kwargs["resume"] = "must"
            if "name" in wandb_init_kwargs:
                del wandb_init_kwargs["name"]
            print(
                f"Attempting to resume WandB run with ID: {resume_wandb_id} and resume='must'. Name will be sourced from existing run."
            )

        wandb.init(**wandb_init_kwargs)

        if wandb.run is not None:
            print(f"WandB logging initialized: {wandb.run.name} (ID: {wandb.run.id})")
            self._run_id = wandb.run.id
        else:
            if resume_wandb_id:
                print(
                    f"Warning: Failed to resume WandB run {resume_wandb_id}. A new run might have been started or init failed."
                )
            else:
                print("Warning: WandB run initialization failed.")

    def get_current_wandb_run_id(self) -> Optional[str]:
        return self._run_id

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        lr: Optional[float] = None,
        sparsity_lambda: Optional[float] = None,
        total_tokens_processed: Optional[int] = None,
    ):
        if not self.enabled:
            return
        import wandb

        metrics = {}
        for key, value in loss_dict.items():
            if key == "total":
                metrics["training/total_loss"] = value
            elif key == "sparsity":
                metrics["training/sparsity_loss"] = value
            elif key == "reconstruction":
                metrics["training/reconstruction_loss"] = value
            elif key == "preactivation":
                metrics["training/preactivation_loss"] = value
            elif key == "auxiliary":
                metrics["training/auxiliary_loss"] = value
            else:
                metrics[f"training/{key}"] = value
        if lr is not None:
            metrics["training/learning_rate"] = lr
        if sparsity_lambda is not None:
            metrics["training/sparsity_lambda"] = sparsity_lambda
        if total_tokens_processed is not None:
            metrics["training/total_tokens_processed"] = total_tokens_processed
        wandb.log(metrics, step=step)

    def log_evaluation(self, step: int, eval_metrics: Dict[str, Any]):
        if not self.enabled:
            return
        import wandb

        wandb_log_dict: Dict[str, Any] = {}
        for key, value in eval_metrics.items():
            if key.startswith("layerwise/"):
                if isinstance(value, dict):
                    for layer_key, layer_value in value.items():
                        wandb_key = f"{key}/{layer_key}"
                        if isinstance(layer_value, list):
                            try:
                                wandb_log_dict[wandb_key] = wandb.Histogram(layer_value)
                            except Exception as e:
                                print(f"Wandb: Error creating histogram for {wandb_key}: {e}")
                                try:
                                    mean_val = sum(layer_value) / len(layer_value) if layer_value else 0.0
                                    wandb_log_dict[f"{wandb_key}_mean"] = mean_val
                                except TypeError:
                                    wandb_log_dict[f"{wandb_key}_mean"] = -1.0
                        elif isinstance(layer_value, (float, int)):
                            wandb_log_dict[wandb_key] = layer_value
                else:
                    wandb_log_dict[key] = value
            elif key.endswith("_agg_hist") and isinstance(value, list):
                try:
                    wandb_log_dict[key] = wandb.Histogram(value)
                except Exception as e:
                    print(f"Wandb: Error creating aggregate histogram for {key}: {e}")
                    try:
                        mean_val = sum(value) / len(value) if value else 0.0
                        wandb_log_dict[f"{key}_mean"] = mean_val
                    except TypeError:
                        wandb_log_dict[f"{key}_mean"] = -1.0
            elif isinstance(value, (float, int)):
                wandb_log_dict[key] = value
            else:
                try:
                    import wandb as _wb

                    if isinstance(value, _wb.Histogram):
                        wandb_log_dict[key] = value
                    else:
                        wandb_log_dict[key] = value
                except Exception:
                    pass
        if wandb_log_dict:
            wandb.log(wandb_log_dict, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        if not self.enabled:
            return
        import wandb
        import os

        if name is None:
            name = os.path.basename(artifact_path)
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if os.path.isdir(artifact_path):
            artifact.add_dir(artifact_path)
        else:
            artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def finish(self):
        if not self.enabled:
            return
        import wandb

        wandb.finish()
