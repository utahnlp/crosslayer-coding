import pytest
import torch

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def get_available_devices():
    """Returns available devices, including cpu, mps, and cuda if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = get_available_devices()


@pytest.fixture(params=DEVICES)
def device(request):
    """Fixture to iterate over all available devices."""
    return torch.device(request.param)


@pytest.fixture
def clt_config():
    """Provides a basic CLTConfig for end-to-end testing."""
    return CLTConfig(
        num_layers=2,
        d_model=8,
        num_features=16,
        activation_fn="relu",  # Use simple ReLU for gradient checking
    )


@pytest.fixture
def clt_model(clt_config, device):
    """Provides a CrossLayerTranscoder instance for integration tests."""
    model = CrossLayerTranscoder(
        config=clt_config,
        process_group=None,
        device=device,
    )
    # Ensure all parameters have requires_grad=True for the backward pass test
    for param in model.parameters():
        param.requires_grad = True
    return model.to(device)


@pytest.fixture
def sample_inputs(clt_config, device):
    """Provides a sample input dictionary with consistent token counts."""
    total_tokens = 20
    return {
        0: torch.randn(total_tokens, clt_config.d_model, device=device),
        1: torch.randn(total_tokens, clt_config.d_model, device=device),
    }


class TestCLTEndToEnd:
    def test_forward_backward_pass(self, clt_model, sample_inputs):
        """
        Tests a full forward and backward pass to ensure gradients are computed.
        """
        # --- Forward Pass ---
        reconstructions = clt_model.forward(sample_inputs)

        # --- Loss Calculation ---
        # A simple MSE loss between the reconstructions and the original inputs
        loss = torch.tensor(0.0, device=clt_model.device, dtype=torch.float32)
        for layer_idx, recon_tensor in reconstructions.items():
            original_tensor = sample_inputs[layer_idx]
            loss += torch.mean((recon_tensor - original_tensor) ** 2)

        # --- Backward Pass ---
        try:
            loss.backward()
        except Exception as e:
            pytest.fail(f"Backward pass failed with exception: {e}")

        # --- Gradient Check ---
        # Check that some gradients have been computed. We check a few key parameters.
        # Encoder weights for layer 0
        assert clt_model.encoder_module.encoders[0].weight.grad is not None
        assert torch.all(torch.isfinite(clt_model.encoder_module.encoders[0].weight.grad))
        assert not torch.all(clt_model.encoder_module.encoders[0].weight.grad == 0)

        # Decoder weights for 0->1
        decoder_key = "0->1"
        assert clt_model.decoder_module.decoders[decoder_key].weight.grad is not None
        assert torch.all(torch.isfinite(clt_model.decoder_module.decoders[decoder_key].weight.grad))
        assert not torch.all(clt_model.decoder_module.decoders[decoder_key].weight.grad == 0)

        # Decoder bias for 1->1
        decoder_key = "1->1"
        if clt_model.decoder_module.decoders[decoder_key].bias_param is not None:
            assert clt_model.decoder_module.decoders[decoder_key].bias_param.grad is not None
            assert torch.all(torch.isfinite(clt_model.decoder_module.decoders[decoder_key].bias_param.grad))
            # Note: Bias gradients can sometimes be zero in simple cases, so we don't assert non-zero
