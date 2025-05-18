import logging
from typing import Dict, Optional, Any, Union, TYPE_CHECKING

# Imports for the loggers it will manage
from clt.logging.wandb_logger import WandBLogger, DummyWandBLogger
from clt.logging.json_file_logger import JsonFileLogger

if TYPE_CHECKING:
    from clt.config import TrainingConfig, CLTConfig  # Keep for type hints

logger = logging.getLogger(__name__)


class MetricLogger:
    """Manages different logging handlers (WandB, JSON file)."""

    wandb_handler: Union[WandBLogger, DummyWandBLogger]
    json_file_handler: JsonFileLogger

    def __init__(
        self,
        training_config: "TrainingConfig",
        clt_config: "CLTConfig",  # Added clt_config for WandB init
        log_dir: str,
        rank: int,
        distributed: bool,
        world_size: int,  # Added world_size for consistency with old MetricLogger
        enable_wandb: bool,  # Specific flag to control WandB
        wandb_project: Optional[str],
        wandb_entity: Optional[str],
        wandb_run_name: Optional[str],
        wandb_tags: Optional[list[str]],
        resume_wandb_id: Optional[str] = None,
    ):
        self.training_config = training_config  # Store for general use if needed
        self.log_dir = log_dir
        self.rank = rank
        self.distributed = distributed
        self.world_size = world_size  # Store world_size

        # Instantiate WandB handler (real or dummy)
        if enable_wandb and (not self.distributed or self.rank == 0):
            self.wandb_handler = WandBLogger(
                training_config=training_config,
                clt_config=clt_config,
                log_dir=log_dir,
                resume_wandb_id=resume_wandb_id,
            )
        else:
            # Dummy for non-rank-0 or if WandB disabled
            self.wandb_handler = DummyWandBLogger(
                training_config=training_config,  # Pass configs even to dummy for interface consistency
                clt_config=clt_config,
                log_dir=log_dir,
                resume_wandb_id=None,
            )

        # Instantiate JSON file logger (always active, but saves only on rank 0)
        self.json_file_handler = JsonFileLogger(log_dir=self.log_dir, rank=self.rank, distributed=self.distributed)

    def log_training_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        current_lr: Optional[float],
        current_sparsity_lambda: Optional[float],
    ):
        # Log to WandB (handled internally by wandb_handler if it's rank 0 / enabled)
        total_tokens_processed = self.training_config.train_batch_size_tokens * self.world_size * (step + 1)
        self.wandb_handler.log_step(
            step,
            loss_dict,
            lr=current_lr,
            sparsity_lambda=current_sparsity_lambda,
            total_tokens_processed=total_tokens_processed,
        )

        # Log data for JSON file (all ranks can accumulate, save is rank 0)
        # Combine all relevant info for JSON log
        train_step_data = {**loss_dict}
        if current_lr is not None:
            train_step_data["learning_rate"] = current_lr
        if current_sparsity_lambda is not None:
            train_step_data["sparsity_lambda"] = current_sparsity_lambda
        if total_tokens_processed is not None:
            train_step_data["total_tokens_processed"] = total_tokens_processed
        self.json_file_handler.log_training_step_data(step, train_step_data)

        # Conditional save to disk for JSON (rank 0 only)
        log_interval = self.training_config.log_interval
        if step % log_interval == 0 and (not self.distributed or self.rank == 0):
            self.json_file_handler.save_metrics_to_disk()

    def log_evaluation_metrics(self, step: int, eval_metrics_dict: Dict[str, Any]):
        # Log to WandB (handled internally by wandb_handler if it's rank 0 / enabled)
        self.wandb_handler.log_evaluation(step, eval_metrics_dict)

        # Log data for JSON file (all ranks can accumulate, save is rank 0)
        self.json_file_handler.log_evaluation_step_data(step, eval_metrics_dict)

        # Conditional save to disk for JSON (rank 0 only)
        if not self.distributed or self.rank == 0:
            self.json_file_handler.save_metrics_to_disk()

    def _save_metrics_to_disk(self):  # Keep this method name for CLTTrainer compatibility during transition
        """Public method to trigger saving JSON metrics, typically called by trainer at end."""
        if not self.distributed or self.rank == 0:
            self.json_file_handler.save_metrics_to_disk()

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        # Delegate to WandB handler (which checks for enabled and rank 0)
        self.wandb_handler.log_artifact(artifact_path, artifact_type, name)

    def finish(self):
        # Delegate to WandB handler
        self.wandb_handler.finish()
        # Ensure final JSON save
        if not self.distributed or self.rank == 0:
            self.json_file_handler.save_metrics_to_disk()

    def get_current_wandb_run_id(self) -> Optional[str]:
        """Delegates to the WandB handler to get the current run ID."""
        return self.wandb_handler.get_current_wandb_run_id()

    def get_metrics_history(self) -> Dict[str, list]:
        """Returns the history from the JSON file logger."""
        return self.json_file_handler.get_metrics_history()
