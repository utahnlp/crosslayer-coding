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


@pytest.fixture(params=["relu", "jumprelu", "batchtopk"])
def activation_fn(request):
    return request.param


@pytest.fixture
def clt_config(activation_fn):
    """Provides a basic CLTConfig for testing, parameterized by activation function."""
    return CLTConfig(
        num_layers=2,
        d_model=8,
        num_features=16,
        activation_fn=activation_fn,
        jumprelu_threshold=0.5,
        batchtopk_k=4,
    )


@pytest.fixture
def clt_model(clt_config, device):
    """Provides a CrossLayerTranscoder instance."""
    return CrossLayerTranscoder(
        config=clt_config,
        process_group=None,
        device=device,
    ).to(device)


@pytest.fixture
def sample_inputs(clt_config, device):
    """
    Provides a sample input dictionary for the CLT.
    Ensures that the total number of tokens is the same for all layers,
    which is a key assumption for batchtopk.
    """
    total_tokens = 30
    return {
        0: torch.randn(total_tokens, clt_config.d_model, device=device),
        1: torch.randn(total_tokens, clt_config.d_model, device=device),
    }


class TestCLTEncodeDecode:
    def test_get_feature_activations_shapes(self, clt_model, sample_inputs, clt_config):
        """Test that get_feature_activations returns activations of the correct shape."""
        activations = clt_model.get_feature_activations(sample_inputs)

        assert isinstance(activations, dict)
        assert sorted(activations.keys()) == sorted(sample_inputs.keys())

        # Check shapes (note that 3D input is flattened in the fixture now)
        assert activations[0].shape == (30, clt_config.num_features)
        assert activations[1].shape == (30, clt_config.num_features)

    def test_relu_activations_are_non_negative(self, clt_model, sample_inputs):
        """Test that ReLU activations are always >= 0."""
        if clt_model.config.activation_fn != "relu":
            pytest.skip("Test only for ReLU activation")

        activations = clt_model.get_feature_activations(sample_inputs)
        for layer_idx in activations:
            assert torch.all(activations[layer_idx] >= 0)

    def test_decode_shapes(self, clt_model, sample_inputs, clt_config):
        """Test that decoding from feature activations produces the correct output shape."""
        activations = clt_model.get_feature_activations(sample_inputs)

        # Decode for layer 1, which can see activations from layers 0 and 1
        reconstruction = clt_model.decode(activations, layer_idx=1)

        # The output batch dimension should match the input batch dimension for that layer
        expected_batch_dim = sample_inputs[1].shape[0]
        assert reconstruction.shape == (expected_batch_dim, clt_config.d_model)

    def test_forward_pass_shapes(self, clt_model, sample_inputs, clt_config):
        """Test the full forward() method returns a dictionary of reconstructions with correct shapes."""
        reconstructions = clt_model.forward(sample_inputs)

        assert isinstance(reconstructions, dict)
        assert sorted(reconstructions.keys()) == sorted(sample_inputs.keys())

        # Check shapes
        assert reconstructions[0].shape == (
            sample_inputs[0].shape[0],
            clt_config.d_model,
        )
        assert reconstructions[1].shape == (
            sample_inputs[1].shape[0],
            clt_config.d_model,
        )
