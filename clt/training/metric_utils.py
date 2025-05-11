import os
import json
import logging
from typing import Dict, Optional, Any, Union  # Added Union

# Forward declarations for type hinting
if False:  # TYPE_CHECKING
    from clt.training.wandb_logger import WandBLogger, DummyWandBLogger
    from clt.config import TrainingConfig

    # from clt.training.losses import LossManager # Not directly needed if lambda is passed

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(
        self,
        distributed: bool,
        rank: int,
        log_dir: str,
        wandb_logger: Union["WandBLogger", "DummyWandBLogger"],
        training_config: "TrainingConfig",
        world_size: int,
    ):
        self.distributed = distributed
        self.rank = rank
        self.log_dir = log_dir
        self.wandb_logger = wandb_logger
        self.training_config = training_config
        self.world_size = world_size
        # self.loss_manager = loss_manager # Not storing loss_manager

        self.metrics: Dict[str, list] = {
            "train_losses": [],
            "eval_metrics": [],  # This will be populated by the trainer calling a method here
        }

    def log_training_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        current_lr: Optional[float],
        current_sparsity_lambda: Optional[float],  # Pass lambda directly
    ):
        """Log training metrics for a step, including LR and lambda."""
        # All ranks might update their local copy of train_losses for potential future needs,
        # but only rank 0 saves/logs to WandB.
        self.metrics["train_losses"].append({"step": step, **loss_dict})

        if not self.distributed or self.rank == 0:
            total_tokens_processed = self.training_config.train_batch_size_tokens * self.world_size * (step + 1)

            self.wandb_logger.log_step(
                step,
                loss_dict,
                lr=current_lr,
                sparsity_lambda=current_sparsity_lambda,  # Use passed lambda
                total_tokens_processed=total_tokens_processed,
            )

            log_interval = self.training_config.log_interval
            if step % log_interval == 0:
                self._save_metrics_to_disk()  # Renamed for clarity

    def log_evaluation_metrics(self, step: int, eval_metrics_dict: Dict[str, Any]):
        """Logs evaluation metrics. Assumes only called on rank 0 if distributed."""
        if not self.distributed or self.rank == 0:
            self.metrics["eval_metrics"].append({"step": step, **eval_metrics_dict})
            self.wandb_logger.log_evaluation(step, eval_metrics_dict)
            self._save_metrics_to_disk()  # Save after eval as well

    def _save_metrics_to_disk(self):
        """Save all tracked metrics to disk. Assumes only called on rank 0 if distributed."""
        if self.distributed and self.rank != 0:
            # This check is a safeguard, but typically this method should only be called by rank 0 logic.
            return

        metrics_path = os.path.join(self.log_dir, "metrics.json")
        try:
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to save metrics to {metrics_path}: {e}")  # Use logger

    def get_metrics_history(self) -> Dict[str, list]:
        """Returns the history of all metrics."""
        return self.metrics
