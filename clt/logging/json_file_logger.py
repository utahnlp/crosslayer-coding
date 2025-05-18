import os
import json
import logging
from typing import Dict, Any, List  # Python's list for type hinting

logger = logging.getLogger(__name__)


class JsonFileLogger:
    def __init__(self, log_dir: str, rank: int = 0, distributed: bool = False):
        self.log_dir = log_dir
        self.rank = rank
        self.distributed = distributed
        self.metrics: Dict[str, List[Dict[str, Any]]] = {  # Ensure correct type for metrics
            "train_steps": [],
            "eval_steps": [],
        }
        # Ensure log_dir exists, especially for rank 0
        if not self.distributed or self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

    def log_training_step_data(self, step: int, data_dict: Dict[str, Any]):
        """Stores training data for a step. Data will be written to disk by save_metrics_to_disk."""
        # All ranks can store, but only rank 0 should save to prevent race conditions/multiple files.
        self.metrics["train_steps"].append({"step": step, **data_dict})

    def log_evaluation_step_data(self, step: int, data_dict: Dict[str, Any]):
        """Stores evaluation data for a step. Data will be written to disk by save_metrics_to_disk."""
        self.metrics["eval_steps"].append({"step": step, **data_dict})

    def save_metrics_to_disk(self):
        """Save all tracked metrics to disk (metrics.json). Should only be called on rank 0 if distributed."""
        if self.distributed and self.rank != 0:
            return

        metrics_path = os.path.join(self.log_dir, "metrics.json")
        try:
            # Ensure data is serializable (e.g. convert tensors to list/float)
            # This basic implementation assumes data_dict items are already JSON-serializable.
            # More robust handling might be needed if tensors or complex objects are passed.
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)  # default=str for things like np.int64
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to save metrics to {metrics_path}: {e}")

    def get_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.metrics
