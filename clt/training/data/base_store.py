import torch
from typing import Dict, List, Tuple, Any
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ActivationBatchCLT = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


class BaseActivationStore(ABC):
    layer_indices: List[int]
    d_model: int
    dtype: torch.dtype
    device: torch.device
    train_batch_size_tokens: int
    total_tokens: int

    @abstractmethod
    def get_batch(self) -> ActivationBatchCLT:
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def close(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.get_batch()
        except StopIteration:
            raise
        except Exception as e:
            logger.error(f"Error during iteration: {e}", exc_info=True)
            raise

    def __len__(self):
        if (
            not hasattr(self, "total_tokens")
            or self.total_tokens <= 0
            or not hasattr(self, "train_batch_size_tokens")
            or self.train_batch_size_tokens <= 0
        ):
            return 0
        return (self.total_tokens + self.train_batch_size_tokens - 1) // self.train_batch_size_tokens
