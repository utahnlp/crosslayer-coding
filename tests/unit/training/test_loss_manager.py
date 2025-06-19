import pytest
import torch

from clt.training.losses import LossManager
from clt.models.clt import CrossLayerTranscoder
from tests.helpers.tiny_configs import create_tiny_clt_config, create_tiny_training_config


@pytest.fixture
def tiny_model() -> CrossLayerTranscoder:
    """Provides a tiny CLT model for testing."""
    clt_config = create_tiny_clt_config(num_features=8, d_model=4)
    return CrossLayerTranscoder(clt_config, process_group=None)


@pytest.fixture
def sample_data():
    """Provides sample predictions and targets."""
    device = torch.device("cpu")
    preds = {0: torch.randn(10, 4, device=device), 1: torch.randn(10, 4, device=device)}
    targets = {0: torch.randn(10, 4, device=device), 1: torch.randn(10, 4, device=device)}
    inputs = {0: torch.randn(10, 4, device=device), 1: torch.randn(10, 4, device=device)}
    activations = {0: torch.rand(10, 8, device=device), 1: torch.rand(10, 8, device=device)}
    return preds, targets, inputs, activations


class TestLossManager:
    def test_reconstruction_loss_no_denorm(self, sample_data):
        """Test reconstruction loss without de-normalization."""
        preds, targets, _, _ = sample_data
        config = create_tiny_training_config(activation_path="dummy")
        loss_manager = LossManager(config)

        loss = loss_manager.compute_reconstruction_loss(preds, targets)

        expected_loss = torch.nn.functional.mse_loss(preds[0], targets[0]) + torch.nn.functional.mse_loss(
            preds[1], targets[1]
        )

        assert torch.isclose(loss, expected_loss)

    def test_reconstruction_loss_with_denorm(self, sample_data):
        """Test that de-normalization is applied correctly."""
        preds, targets, _, _ = sample_data
        config = create_tiny_training_config(activation_path="dummy")

        mean_tg = {0: torch.tensor([[10.0]]), 1: torch.tensor([[-5.0]])}
        std_tg = {0: torch.tensor([[2.0]]), 1: torch.tensor([[0.5]])}

        loss_manager = LossManager(config, mean_tg=mean_tg, std_tg=std_tg)
        loss = loss_manager.compute_reconstruction_loss(preds, targets)

        pred0_denorm = preds[0] * std_tg[0] + mean_tg[0]
        target0_denorm = targets[0] * std_tg[0] + mean_tg[0]
        pred1_denorm = preds[1] * std_tg[1] + mean_tg[1]
        target1_denorm = targets[1] * std_tg[1] + mean_tg[1]

        expected_loss = torch.nn.functional.mse_loss(pred0_denorm, target0_denorm) + torch.nn.functional.mse_loss(
            pred1_denorm, target1_denorm
        )

        assert torch.isclose(loss, expected_loss)

    @pytest.mark.parametrize("schedule", ["linear", "delayed_linear"])
    def test_sparsity_penalty_schedule(self, tiny_model, sample_data, schedule):
        """Test sparsity penalty follows the lambda schedule."""
        _, _, _, activations = sample_data
        config = create_tiny_training_config(
            sparsity_lambda=1.0,
            sparsity_lambda_schedule=schedule,
            sparsity_lambda_delay_frac=0.5,
            training_steps=100,
            activation_path="dummy",
        )
        loss_manager = LossManager(config)

        # At the beginning (step 0), lambda should be 0
        loss_at_start, lambda_at_start = loss_manager.compute_sparsity_penalty(tiny_model, activations, 0, 100)
        assert torch.isclose(loss_at_start, torch.tensor(0.0))
        assert lambda_at_start == 0.0

        # Halfway through
        loss_mid, lambda_mid = loss_manager.compute_sparsity_penalty(tiny_model, activations, 50, 100)
        if schedule == "linear":
            assert lambda_mid == pytest.approx(0.5)
            assert loss_mid > 0
        else:  # delayed_linear with 0.5 delay
            assert lambda_mid == pytest.approx(0.0)
            assert torch.isclose(loss_mid, torch.tensor(0.0))

        # At the end
        loss_end, lambda_end = loss_manager.compute_sparsity_penalty(tiny_model, activations, 100, 100)
        assert lambda_end == pytest.approx(1.0)
        assert loss_end > loss_mid

    def test_preactivation_loss(self, tiny_model, sample_data):
        """Test pre-activation loss penalizes negative pre-activations."""
        _, _, inputs, _ = sample_data
        config = create_tiny_training_config(preactivation_coef=1.0, activation_path="dummy")
        loss_manager = LossManager(config)

        # Mock get_preactivations to return controlled values
        def mock_get_preactivations(x, layer_idx):
            if layer_idx == 0:
                return torch.tensor([[-0.5, 0.5], [-0.2, 0.8]])  # Has negative values
            return torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # All positive

        tiny_model.get_preactivations = mock_get_preactivations

        loss = loss_manager.compute_preactivation_loss(tiny_model, inputs)

        # Expected penalty is from layer 0: (0.5 + 0.2) / 8 elements total
        expected_loss = (0.5 + 0.2) / 8
        assert torch.isclose(loss, torch.tensor(expected_loss))

    def test_total_loss_computation(self, tiny_model, sample_data, mocker):
        """Test that the total loss is the sum of its components."""
        preds, targets, inputs, activations = sample_data
        config = create_tiny_training_config(sparsity_lambda=0.1, preactivation_coef=0.1, activation_path="dummy")
        loss_manager = LossManager(config)

        mocker.patch.object(tiny_model, "get_feature_activations", return_value=activations)
        mocker.patch.object(tiny_model, "__call__", return_value=preds)

        total_loss, loss_dict = loss_manager.compute_total_loss(
            model=tiny_model, inputs=inputs, targets=targets, current_step=50, total_steps=100
        )

        expected_total = (
            loss_dict["reconstruction"] + loss_dict["sparsity"] + loss_dict["preactivation"] + loss_dict["auxiliary"]
        )

        assert total_loss.item() == pytest.approx(expected_total)
        assert loss_dict["total"] == pytest.approx(expected_total)
        assert "reconstruction" in loss_dict
        assert "sparsity" in loss_dict
        assert "preactivation" in loss_dict
        assert "auxiliary" in loss_dict
