import pytest
import torch
import torch.nn as nn
import logging
from torch.utils.data import IterableDataset

from clt.config import CLTConfig
from clt.models.theta import ThetaManager


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
def clt_config_jumprelu():
    """Provides a CLTConfig for testing JumpReLU."""
    return CLTConfig(
        num_layers=2,
        d_model=8,
        num_features=16,
        activation_fn="jumprelu",
        jumprelu_threshold=0.5,
    )


@pytest.fixture
def clt_config_batchtopk():
    """Provides a CLTConfig for testing BatchTopK conversion."""
    return CLTConfig(
        num_layers=2,
        d_model=8,
        num_features=4,
        activation_fn="batchtopk",
        batchtopk_k=2,
    )


class MockIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class TestThetaManager:
    def test_initialization_jumprelu(self, clt_config_jumprelu, device):
        """Test ThetaManager initializes correctly for JumpReLU."""
        tm = ThetaManager(clt_config_jumprelu, None, device, torch.float32)
        assert isinstance(tm.log_threshold, nn.Parameter)
        assert tm.log_threshold.shape == (
            clt_config_jumprelu.num_layers,
            clt_config_jumprelu.num_features,
        )
        expected_val = torch.log(torch.tensor(0.5, device=device))
        torch.testing.assert_close(tm.log_threshold.mean(), expected_val)

    def test_initialization_other_activation(self, clt_config_batchtopk, device):
        """Test ThetaManager initializes correctly for other activations."""
        tm = ThetaManager(clt_config_batchtopk, None, device, torch.float32)
        assert tm.log_threshold is None

    def test_jumprelu_activation(self, clt_config_jumprelu, device):
        """Test the JumpReLU activation logic."""
        tm = ThetaManager(clt_config_jumprelu, None, device, torch.float32)
        # Input must match the number of features
        preacts = torch.linspace(-1.0, 1.0, clt_config_jumprelu.num_features, device=device).view(1, -1)
        # threshold is 0.5
        layer_idx = 0
        activated = tm.jumprelu(preacts, layer_idx)

        # Manually compute expected output
        threshold_val = clt_config_jumprelu.jumprelu_threshold
        expected = torch.where(preacts >= threshold_val, preacts, torch.zeros_like(preacts))

        torch.testing.assert_close(activated, expected)

    def test_jumprelu_invalid_layer_idx(self, clt_config_jumprelu, device, caplog):
        """Test JumpReLU with an out-of-bounds layer index."""
        tm = ThetaManager(clt_config_jumprelu, None, device, torch.float32)
        preacts = torch.randn(2, 16, device=device)
        with caplog.at_level(logging.ERROR):
            activated = tm.jumprelu(preacts, layer_idx=clt_config_jumprelu.num_layers)
        assert "Invalid layer_idx" in caplog.text
        # Should return the original tensor
        torch.testing.assert_close(activated, preacts)

    def test_convert_to_jumprelu_raises_error_if_stats_missing(self, clt_config_batchtopk, device):
        """Test that conversion fails if estimation has not been run."""
        tm = ThetaManager(clt_config_batchtopk, None, device, torch.float32)
        with pytest.raises(RuntimeError, match="Required buffer .* not found"):
            tm.convert_to_jumprelu_inplace()

    def test_estimate_and_convert_posthoc(self, clt_config_batchtopk, device):
        """
        Test the full estimate_theta_posthoc and convert_to_jumprelu_inplace flow.
        """
        tm = ThetaManager(clt_config_batchtopk, None, device, torch.float32)
        num_features = clt_config_batchtopk.num_features
        num_layers = clt_config_batchtopk.num_layers

        def mock_encode_all_layers(inputs):
            # This mock should return preactivations that look like they came from a real model
            # i.e., they have some mean and std dev. The estimation process will normalize them.
            # Layer 0: High values for first k features
            preacts_l0 = torch.cat(
                [
                    torch.randn(4, 2, device=device) + 5,  # High values for top-k
                    torch.randn(4, 2, device=device),  # Low values for others
                ],
                dim=1,
            )
            # Layer 1: High values for last k features
            preacts_l1 = torch.cat(
                [torch.randn(4, 2, device=device), torch.randn(4, 2, device=device) + 5],  # Low values  # High values
                dim=1,
            )
            return {0: preacts_l0, 1: preacts_l1}, []

        # Mock data iterator to yield one batch
        mock_data = [({0: torch.randn(1, 8), 1: torch.randn(1, 8)}, None)]
        mock_data_iter = MockIterableDataset(mock_data)

        # --- Run estimation ---
        tm.estimate_theta_posthoc(
            encode_all_layers_fn=mock_encode_all_layers,
            data_iter=mock_data_iter,
            num_batches=1,
        )

        # --- Check results of conversion ---
        assert tm.config.activation_fn == "jumprelu"
        assert tm.log_threshold is not None
        assert tm.log_threshold.shape == (num_layers, num_features)

        # The post-hoc estimation logic is complex. Instead of asserting exact values,
        # which are sensitive to the mock, we check for reasonable behavior:
        # 1. Thetas should be positive and finite.
        # 2. Thetas for the activated features should be higher than for non-activated ones.
        final_thetas = torch.exp(tm.log_threshold)
        assert torch.all(torch.isfinite(final_thetas))
        assert torch.all(final_thetas > 0)

        # For layer 0, the first 2 features were consistently active and high.
        # Their thresholds should be higher than the last 2 features.
        assert torch.all(final_thetas[0, :2] > final_thetas[0, 2:])

        # For layer 1, the last 2 features were active.
        assert torch.all(final_thetas[1, 2:] > final_thetas[1, :2])
