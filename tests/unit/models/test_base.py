import pytest
import torch
import torch.nn as nn
from typing import Dict
import os

# Import the actual config
from clt.config import CLTConfig

from clt.models.base import BaseTranscoder


# Assume BaseTranscoder exists in clt.models.base
# Need a concrete implementation for testing save/load


class DummyTranscoder(BaseTranscoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.Linear(config.d_model, config.d_model)

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.layer(x)  # Simplified encode

    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        # Simplified decode - just returns the activation from layer 0 if present
        # Assumes key 0 is present based on simplified forward logic/test setup
        return a[0]

    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs = {}
        activations = {}
        for layer_idx, x in inputs.items():
            activations[layer_idx] = self.encode(x, layer_idx)

        act_0 = activations.get(0)
        # Only proceed with decode if layer 0 activation exists
        if act_0 is not None:
            decode_input = {0: act_0}
            for layer_idx in inputs.keys():
                # Call decode with the dictionary containing only layer 0 activation
                outputs[layer_idx] = self.decode(decode_input, layer_idx)
        # If act_0 is None, outputs dict remains empty or partially filled
        # depending on previous loops, which is acceptable for this dummy class.
        return outputs


@pytest.fixture
def dummy_config():
    # Use the actual CLTConfig
    return CLTConfig(d_model=16, num_layers=2, num_features=32, activation_fn="relu")


@pytest.fixture
def dummy_transcoder(dummy_config):
    return DummyTranscoder(dummy_config)


def test_base_transcoder_init(dummy_transcoder, dummy_config):
    """Test BaseTranscoder initialization through a dummy implementation."""
    assert dummy_transcoder.config == dummy_config
    assert isinstance(dummy_transcoder, nn.Module)


def test_base_transcoder_save_load(dummy_transcoder, dummy_config, tmp_path):
    """Test saving and loading a BaseTranscoder."""
    save_path = tmp_path / "dummy_transcoder.pt"

    # Save the model
    dummy_transcoder.save(str(save_path))
    assert os.path.exists(save_path)

    # Load the model (safe_globals not needed for dataclass config)
    loaded_transcoder = DummyTranscoder.load(str(save_path))

    # Check loaded model
    assert isinstance(loaded_transcoder, DummyTranscoder)
    assert isinstance(loaded_transcoder.config, CLTConfig)
    assert loaded_transcoder.config.d_model == dummy_config.d_model
    assert loaded_transcoder.config.num_layers == dummy_config.num_layers
    # Removed check for mock-specific param

    # Check state dicts match
    assert dummy_transcoder.state_dict().keys() == loaded_transcoder.state_dict().keys()
    for key in dummy_transcoder.state_dict():
        assert torch.equal(
            dummy_transcoder.state_dict()[key], loaded_transcoder.state_dict()[key]
        )


def test_base_transcoder_load_device(dummy_transcoder, tmp_path):
    """Test loading a BaseTranscoder to a specific device."""
    save_path = tmp_path / "dummy_transcoder_device.pt"
    dummy_transcoder.save(str(save_path))

    # Try loading to CPU (assuming tests run on CPU by default)
    loaded_transcoder_cpu = DummyTranscoder.load(
        str(save_path), device=torch.device("cpu")
    )
    assert next(loaded_transcoder_cpu.parameters()).device.type == "cpu"

    # If CUDA is available, test loading to CUDA
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        loaded_transcoder_cuda = DummyTranscoder.load(
            path=str(save_path), device=cuda_device
        )
        assert next(loaded_transcoder_cuda.parameters()).device.type == "cuda"


# Test abstract methods raise NotImplementedError if called on BaseTranscoder directly
# This requires a bit of setup, maybe skip for now unless explicitly needed.
# Trying to instantiate BaseTranscoder directly should fail anyway.

# Note: Linter errors in base.py (unused Any, Tuple, unknown CLTConfig) should be addressed
# separately. This test file uses a MockCLTConfig.
