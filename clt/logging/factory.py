from typing import Optional, Union, Tuple, TYPE_CHECKING

# Import the logger classes this factory will create and manage
from clt.logging.wandb_logger import WandBLogger, DummyWandBLogger  # wandb_logger lives here now
from clt.logging.metric_logger import MetricLogger  # The new MetricLogger

# JsonFileLogger is instantiated within the new MetricLogger, so not directly needed by factory user

if TYPE_CHECKING:
    from clt.config import TrainingConfig, CLTConfig


def setup_loggers(
    training_config: "TrainingConfig",
    clt_config: "CLTConfig",
    log_dir: str,
    rank: int,
    distributed: bool,
    world_size: int,
    resume_wandb_id: Optional[str] = None,
) -> Tuple[Union[WandBLogger, DummyWandBLogger], MetricLogger]:  # Return type updated
    """
    Initializes WandBLogger (or Dummy) and the main MetricLogger which manages all logging.

    Args:
        training_config: Training configuration.
        clt_config: CLT model configuration.
        log_dir: Directory for logs.
        rank: Process rank.
        distributed: Boolean indicating if in distributed mode.
        world_size: Total number of processes.
        resume_wandb_id: Optional WandB run ID to resume.

    Returns:
        A tuple containing (wandb_logger_instance_for_direct_access, metric_logger_instance).
        The metric_logger_instance itself manages its own wandb_handler and json_file_handler.
    """

    # The new MetricLogger needs parameters from training_config for WandB initialization
    # These were previously passed directly to WandBLogger constructor.
    metric_logger_instance = MetricLogger(
        training_config=training_config,
        clt_config=clt_config,
        log_dir=log_dir,
        rank=rank,
        distributed=distributed,
        world_size=world_size,
        enable_wandb=training_config.enable_wandb,
        wandb_project=training_config.wandb_project,
        wandb_entity=training_config.wandb_entity,
        wandb_run_name=training_config.wandb_run_name,
        wandb_tags=training_config.wandb_tags,
        resume_wandb_id=resume_wandb_id,
    )

    # CLTTrainer needs direct access to the wandb_handler for get_current_wandb_run_id() and log_artifact()
    # which are now exposed via MetricLogger. So we can return metric_logger_instance.wandb_handler.
    # However, the original CLTTrainer directly assigned the first element of the tuple to self.wandb_logger
    # and second to self.metric_logger.
    # The new MetricLogger *contains* the wandb_handler.
    # For minimal changes to CLTTrainer, we can return metric_logger_instance.wandb_handler and metric_logger_instance

    return metric_logger_instance.wandb_handler, metric_logger_instance
