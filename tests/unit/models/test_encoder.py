import pytest
import torch
import logging

from clt.config import CLTConfig
from clt.models.encoder import Encoder


def get_available_devices():
    """Returns available devices, including cpu, mps, and cuda if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = get_available_devices()
GPU_DEVICES = [d for d in DEVICES if d != "cpu"]


def require_gpu(func):
    """Decorator to skip tests if no GPU is available."""
    return pytest.mark.skipif(not GPU_DEVICES, reason="Test requires a GPU (CUDA or MPS)")(func)


@pytest.fixture
def clt_config():
    """Provides a basic CLTConfig for testing."""
    return CLTConfig(
        num_layers=2,
        d_model=8,
        num_features=16,
    )


@pytest.fixture
def encoder(clt_config, device):
    """Provides an Encoder instance."""
    return Encoder(
        config=clt_config,
        process_group=None,  # Non-distributed for unit tests
        device=device,
        dtype=torch.float32,
    ).to(device)


class TestEncoder:
    def test_get_preactivations_2d_input(self, encoder, clt_config, device):
        """Test get_preactivations with a 2D tensor."""
        batch_size = 4
        x = torch.randn(batch_size, clt_config.d_model, device=device)
        layer_idx = 0

        preacts = encoder.get_preactivations(x, layer_idx)

        assert preacts.shape == (batch_size, clt_config.num_features)
        assert preacts.device.type == device.type
        assert preacts.dtype == torch.float32

    def test_get_preactivations_3d_input(self, encoder, clt_config, device):
        """Test get_preactivations with a 3D tensor."""
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, clt_config.d_model, device=device)
        layer_idx = 1

        preacts = encoder.get_preactivations(x, layer_idx)

        assert preacts.shape == (batch_size * seq_len, clt_config.num_features)
        assert preacts.device.type == device.type
        assert preacts.dtype == torch.float32

    @require_gpu
    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_get_preactivations_3d_input_gpu(self, clt_config, device):
        """Dedicated GPU test for 3D input to ensure it runs on accelerator."""
        device_obj = torch.device(device)
        # We need to recreate the encoder on the correct GPU device for this specific test
        gpu_encoder = Encoder(
            config=clt_config,
            process_group=None,
            device=device_obj,
            dtype=torch.float32,
        ).to(device_obj)
        self.test_get_preactivations_3d_input(gpu_encoder, clt_config, device_obj)

    def test_get_preactivations_mismatched_d_model(self, encoder, clt_config, device, caplog):
        """Test get_preactivations with mismatched d_model, expecting a warning and zero tensor."""
        batch_size = 2
        seq_len = 5
        wrong_d_model = clt_config.d_model + 4
        x = torch.randn(batch_size, seq_len, wrong_d_model, device=device)
        layer_idx = 0

        with caplog.at_level(logging.WARNING):
            preacts = encoder.get_preactivations(x, layer_idx)

        assert preacts.shape == (batch_size * seq_len, clt_config.num_features)
        assert torch.all(preacts == 0)
        assert "Input d_model" in caplog.text
        assert str(wrong_d_model) in caplog.text

    def test_get_preactivations_invalid_layer_idx(self, encoder, clt_config, device, caplog):
        """Test get_preactivations with an out-of-bounds layer_idx."""
        batch_size = 4
        x = torch.randn(batch_size, clt_config.d_model, device=device)
        invalid_layer_idx = clt_config.num_layers  # num_layers is OOB

        with caplog.at_level(logging.ERROR):
            preacts = encoder.get_preactivations(x, invalid_layer_idx)

        assert preacts.shape == (batch_size, clt_config.num_features)
        assert torch.all(preacts == 0)
        assert "Invalid layer index" in caplog.text
        assert str(invalid_layer_idx) in caplog.text

    def test_encode_all_layers(self, encoder, clt_config, device):
        """Test the encode_all_layers method."""
        inputs = {
            0: torch.randn(4, clt_config.d_model, device=device),
            1: torch.randn(2, 5, clt_config.d_model, device=device),
        }

        preacts_dict, shapes_info = encoder.encode_all_layers(inputs)

        # Check keys
        assert sorted(preacts_dict.keys()) == sorted(inputs.keys())

        # Check shapes and values
        assert preacts_dict[0].shape == (4, clt_config.num_features)
        assert preacts_dict[1].shape == (10, clt_config.num_features)

        # Verify against direct calls
        torch.testing.assert_close(preacts_dict[0], encoder.get_preactivations(inputs[0], 0))
        torch.testing.assert_close(preacts_dict[1], encoder.get_preactivations(inputs[1], 1))

        # Check shapes_info
        assert len(shapes_info) == 2
        # Order should be sorted by layer_idx
        assert shapes_info[0] == (0, 4, 1)  # (layer_idx, batch_size, seq_len)
        assert shapes_info[1] == (1, 2, 5)

    def test_get_preactivations_unusual_input_dims(self, encoder, clt_config, device, caplog):
        """Test get_preactivations with unexpected input dimensions (1D, 4D)."""
        # Test with 1D input
        x_1d = torch.randn(clt_config.d_model, device=device)
        with caplog.at_level(logging.WARNING):
            preacts_1d = encoder.get_preactivations(x_1d, 0)
        assert "Cannot handle input shape" in caplog.text
        # The fallback logic uses shape[0] as the batch dim.
        assert preacts_1d.shape == (x_1d.shape[0], clt_config.num_features)

        caplog.clear()

        # Test with 4D input
        x_4d = torch.randn(2, 3, 4, clt_config.d_model, device=device)
        with caplog.at_level(logging.WARNING):
            preacts_4d = encoder.get_preactivations(x_4d, 0)
        assert "Cannot handle input shape" in caplog.text
        assert preacts_4d.shape == (x_4d.shape[0], clt_config.num_features)
