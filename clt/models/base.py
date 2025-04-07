from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, TypeVar, Type

from clt.config import CLTConfig


# Generic type for Transcoder subclasses
T = TypeVar("T", bound="BaseTranscoder")


class BaseTranscoder(nn.Module, ABC):
    """Abstract base class for all transcoders."""

    config: CLTConfig

    def __init__(self, config: CLTConfig):
        """Initialize the transcoder with the given configuration.

        Args:
            config: Configuration for the transcoder.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.

        Args:
            x: Input activations [batch_size, seq_len, d_model]
            layer_idx: Index of the layer

        Returns:
            Encoded activations
        """
        pass

    @abstractmethod
    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Decode the feature activations to reconstruct outputs at the specified layer.

        Args:
            a: Dictionary mapping layer indices to feature activations
            layer_idx: Index of the layer to reconstruct outputs for

        Returns:
            Reconstructed outputs
        """
        pass

    @abstractmethod
    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Process inputs through the transcoder model.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to reconstructed outputs
        """
        pass

    def save(self, path: str) -> None:
        """Save the transcoder model to the specified path.

        Args:
            path: Path to save the model
        """
        # Ensure config is serializable (e.g., if it's a dataclass)
        # If config is a dataclass, convert to dict
        config_dict = (
            self.config.__dict__ if hasattr(self.config, "__dict__") else self.config
        )

        checkpoint = {"config": config_dict, "state_dict": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls: Type[T], path: str, device: Optional[torch.device] = None) -> T:
        """Load a transcoder model from the specified path.

        Args:
            path: Path to load the model from
            device: Device to load the model to

        Returns:
            Loaded transcoder model
        """
        checkpoint = torch.load(path, map_location=device)

        # Instantiate the config object from the dictionary
        config_dict = checkpoint["config"]
        # Assuming the config class is CLTConfig for now
        # A more robust solution might store the config class name
        config = CLTConfig(**config_dict)

        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
