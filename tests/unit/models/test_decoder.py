import pytest
import torch
import logging

from clt.config import CLTConfig
from clt.models.decoder import Decoder


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
def decoder(clt_config, device):
    """Provides a Decoder instance."""
    return Decoder(
        config=clt_config,
        process_group=None,  # Non-distributed for unit tests
        device=device,
        dtype=torch.float32,
    ).to(device)


class TestDecoder:
    def test_decode_single_layer(self, decoder, clt_config, device):
        """Test decoding from a single source layer."""
        batch_size = 4
        activations = {0: torch.randn(batch_size, clt_config.num_features, device=device)}
        target_layer = 0

        reconstruction = decoder.decode(activations, target_layer)

        assert reconstruction.shape == (batch_size, clt_config.d_model)
        assert reconstruction.device.type == device.type
        assert reconstruction.dtype == torch.float32

    def test_decode_multi_layer_sum(self, decoder, clt_config, device):
        """Test that reconstructions are summed across multiple source layers."""
        batch_size = 3
        activations = {
            0: torch.ones(batch_size, clt_config.num_features, device=device),
            1: torch.ones(batch_size, clt_config.num_features, device=device) * 2,
        }
        target_layer = 1

        # Decode with both layers
        reconstruction_both = decoder.decode(activations, target_layer)

        # Decode with only the first layer
        reconstruction_first = decoder.decode({0: activations[0]}, target_layer)

        # Decode with only the second layer
        reconstruction_second = decoder.decode({1: activations[1]}, target_layer)

        # The sum should be close to the combined reconstruction
        torch.testing.assert_close(reconstruction_both, reconstruction_first + reconstruction_second)

    def test_decode_empty_activations_dict(self, decoder, clt_config, device):
        """Test decode with an empty activation dictionary."""
        reconstruction = decoder.decode({}, layer_idx=0)
        assert reconstruction.shape == (0, clt_config.d_model)

    def test_decode_activations_with_zero_elements(self, decoder, clt_config, device):
        """Test decode with a tensor that has a zero dimension."""
        activations = {0: torch.randn(0, clt_config.num_features, device=device)}
        reconstruction = decoder.decode(activations, layer_idx=0)
        assert reconstruction.shape == (0, clt_config.d_model)

    def test_decode_mismatched_feature_dim(self, decoder, clt_config, device, caplog):
        """Test that activations with wrong feature dimensions are skipped with a warning."""
        batch_size = 4
        wrong_features = clt_config.num_features - 4
        activations = {
            0: torch.randn(batch_size, wrong_features, device=device),
            1: torch.randn(batch_size, clt_config.num_features, device=device),
        }
        target_layer = 1

        with caplog.at_level(logging.WARNING):
            reconstruction = decoder.decode(activations, target_layer)

        # The reconstruction should only be from layer 1
        reconstruction_only_l1 = decoder.decode({1: activations[1]}, target_layer)
        torch.testing.assert_close(reconstruction, reconstruction_only_l1)

        assert "incorrect feature dimension" in caplog.text
        assert str(wrong_features) in caplog.text

    def test_get_decoder_norms_shape_and_device(self, decoder, clt_config, device):
        """Test the shape and device of the decoder norms tensor."""
        norms = decoder.get_decoder_norms()
        assert norms.shape == (clt_config.num_layers, clt_config.num_features)
        assert norms.device.type == device.type

    def test_get_decoder_norms_caching(self, decoder):
        """Test that get_decoder_norms caches the result."""
        norms1 = decoder.get_decoder_norms()
        norms2 = decoder.get_decoder_norms()
        # The exact same object should be returned
        assert id(norms1) == id(norms2)
        # Invalidate cache and check again
        decoder._cached_decoder_norms = None
        norms3 = decoder.get_decoder_norms()
        assert id(norms1) != id(norms3)

    @require_gpu
    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_decode_on_gpu(self, clt_config, device):
        """Dedicated test to ensure decode runs on GPU."""
        device_obj = torch.device(device)
        decoder = Decoder(
            config=clt_config,
            process_group=None,
            device=device_obj,
            dtype=torch.float32,
        ).to(device_obj)
        self.test_decode_single_layer(decoder, clt_config, device_obj)
