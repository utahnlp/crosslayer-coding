# tests/unit/training/test_losses.py
import pytest
import torch
from unittest.mock import MagicMock, call

# Move import to top
from clt.training.losses import LossManager


# Mock the dependencies before importing LossManager
# Usually, you'd structure your project so these are actual importable classes
class MockTrainingConfig:
    def __init__(self, sparsity_lambda=0.1, sparsity_c=1.0, preactivation_coef=0.01):
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_c = sparsity_c
        self.preactivation_coef = preactivation_coef


class MockCrossLayerTranscoder(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mock methods that LossManager calls
        self.get_decoder_norms = MagicMock()
        self.get_preactivations = MagicMock()
        self.get_feature_activations = MagicMock()
        # __call__ is handled by setting self.side_effect on the instance


# Now import the class under test


@pytest.fixture
def mock_config():
    return MockTrainingConfig(
        sparsity_lambda=0.1, sparsity_c=2.0, preactivation_coef=0.05
    )


@pytest.fixture
def mock_model(mock_config):
    model = MockCrossLayerTranscoder()
    device = torch.device("cpu")  # Assuming CPU for tests

    # Setup default mock return values
    decoder_norms = {
        0: torch.tensor([0.5, 1.0, 1.5], device=device),
        1: torch.tensor([1.0, 0.8], device=device),
        2: torch.tensor([2.0], device=device),  # For 1D test
    }
    model.get_decoder_norms.return_value = decoder_norms

    # Mock preactivations based on input and layer_idx
    def mock_get_preactivations(x, layer_idx):
        # Ensure input is on the correct device
        x = x.to(device)
        # Simple mock: return input shifted and scaled
        if layer_idx == 0:  # Used for 3D/2D input tests
            # Expects [B, S, D] or [B*S, D] -> returns [B*S, Feat=3]
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x
            # Ensure output dim matches feature dim for the layer
            return (x_flat[:, :3] * 0.8 - 0.1).to(device)
        elif layer_idx == 1:  # Used for 3D/2D input tests
            # Expects [B, S, D] or [B*S, D] -> returns [B*S, Feat=2]
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.shape[-1])
            else:
                x_flat = x
            # Ensure output dim matches feature dim for the layer
            return (x_flat[:, :2] * 1.2 + 0.2).to(device)
        elif layer_idx == 2:  # Used for 1D input test
            # Expects [D] -> reshaped to [1, D] -> returns [1, Feat=1]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            # Ensure output dim matches feature dim for the layer
            return (x[:, :1] * 0.5 - 0.05).to(device)  # Example preact for 1D
        return torch.zeros(x.shape[0], 1, device=device)  # Default fallback

    model.get_preactivations.side_effect = mock_get_preactivations

    # Mock feature activations based on input
    def mock_get_feature_activations(inputs):
        activations = {}
        for layer_idx, x_in in inputs.items():
            x_in = x_in.to(device)
            preacts = mock_get_preactivations(
                x_in, layer_idx
            )  # Use the same mock logic
            # Apply ReLU - actual model uses JumpReLU, but ReLU is simpler for testing activation shapes
            acts = torch.relu(preacts).to(device)
            # Reshape back if original input was 3D? No, feature acts are usually 2D/3D
            # The loss function handles reshaping internally. Let's return consistent shapes.
            # For testing, let's assume get_feature_activations returns [Batch*Seq, Features] or [Batch, Seq, Features]
            # Let's return 3D for layer 0, 2D for layer 1, 1D (reshaped to 2D) for layer 2 for testing robustness
            if layer_idx == 0:
                # Find original batch/seq shape if possible (crude heuristic)
                original_shape = inputs[layer_idx].shape
                if len(original_shape) == 3:
                    activations[layer_idx] = acts.reshape(
                        original_shape[0], original_shape[1], -1
                    )
                else:  # Assume original was 2D
                    activations[layer_idx] = acts  # Return as [Batch*Seq, Feat]
            elif layer_idx == 1:
                activations[layer_idx] = acts  # Return as [Batch*Seq, Feat]
            elif layer_idx == 2:
                activations[layer_idx] = acts.squeeze(
                    0
                )  # Return as [Feat] -> Loss handles unsqueeze

        return activations

    model.get_feature_activations.side_effect = mock_get_feature_activations

    # Mock predictions based on input - make the *instance* callable
    def mock_call(inputs):
        predictions = {}
        for layer_idx, x_in in inputs.items():
            x_in = x_in.to(device)
            # Simple mock prediction logic
            if layer_idx == 0:
                predictions[layer_idx] = (x_in * 0.9).to(device)
            elif layer_idx == 1:
                predictions[layer_idx] = (x_in * 1.1).to(device)
            else:
                predictions[layer_idx] = x_in.to(device)
        return predictions

    # Use model.side_effect for the instance __call__
    model.side_effect = mock_call
    model.device = device  # Add device attribute if needed

    return model


@pytest.fixture
def loss_manager(mock_config):
    return LossManager(mock_config)


# --- Test Cases ---


def test_loss_manager_init(loss_manager, mock_config):
    assert loss_manager.config == mock_config
    assert isinstance(loss_manager.reconstruction_loss_fn, torch.nn.MSELoss)
    assert loss_manager.current_sparsity_lambda == 0.0  # Check initial value


def test_compute_reconstruction_loss_basic(loss_manager):
    device = torch.device("cpu")
    predicted = {
        0: torch.tensor([[1.0, 2.0]], device=device),
        1: torch.tensor([[3.0, 4.0]], device=device),
    }
    target = {
        0: torch.tensor([[1.1, 1.9]], device=device),
        1: torch.tensor([[3.2, 4.1]], device=device),
    }

    expected_loss_0 = torch.mean(
        torch.tensor([(1.0 - 1.1) ** 2, (2.0 - 1.9) ** 2], device=device)
    )
    expected_loss_1 = torch.mean(
        torch.tensor([(3.0 - 3.2) ** 2, (4.0 - 4.1) ** 2], device=device)
    )
    expected_total_loss = (expected_loss_0 + expected_loss_1) / 2

    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.isclose(loss, expected_total_loss)
    assert loss.device == device


def test_compute_reconstruction_loss_mismatched_keys(loss_manager):
    device = torch.device("cpu")
    predicted = {
        0: torch.tensor([[1.0]], device=device),
        1: torch.tensor([[3.0]], device=device),
    }
    target = {
        0: torch.tensor([[1.1]], device=device),
        2: torch.tensor([[5.0]], device=device),
    }  # Layer 1 missing in target, layer 2 missing in predicted

    expected_loss_0 = torch.mean(torch.tensor([(1.0 - 1.1) ** 2], device=device))
    # Layer 1 loss is not calculated as it's not in target
    expected_total_loss = expected_loss_0 / 1  # Only one layer (layer 0) contributes

    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.isclose(loss, expected_total_loss)
    assert loss.device == device


def test_compute_reconstruction_loss_empty(loss_manager):
    device = torch.device("cpu")  # Assume default device if empty
    predicted = {}
    target = {}
    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.equal(loss, torch.tensor(0.0, device=device))
    assert loss.device == device

    predicted = {0: torch.tensor([[1.0]], device=device)}
    target = {}
    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.equal(loss, torch.tensor(0.0, device=device))  # No matching keys
    assert loss.device == device

    # Case with target but no predicted
    predicted = {}
    target = {0: torch.tensor([[1.0]], device=device)}
    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.equal(loss, torch.tensor(0.0, device=device))
    assert loss.device == device


def test_compute_sparsity_penalty_basic_3d(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Batch=1, Seq=2, Feat=3
    activations = {0: torch.tensor([[[0.1, 0.0, 0.5], [0.2, 0.3, 0.0]]], device=device)}
    current_step = 50
    total_steps = 100

    # Get expected norms from mock
    local_decoder_norms = mock_model.get_decoder_norms()

    acts_flat = activations[0].reshape(-1, 3)  # Shape (2, 3)
    weights = local_decoder_norms[0].unsqueeze(0).to(device)  # Shape (1, 3)
    weighted_acts = acts_flat * weights  # Shape (2, 3)

    # tanh penalty computation
    penalty_tensor = torch.tanh(mock_config.sparsity_c * weighted_acts)
    expected_penalty_sum = penalty_tensor.sum()

    lambda_factor = mock_config.sparsity_lambda * (current_step / total_steps)
    expected_total_penalty = lambda_factor * expected_penalty_sum
    # Note: The implementation sums the penalty, it doesn't average per element.

    mock_model.get_decoder_norms.reset_mock()  # Reset before the call under test
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, current_step, total_steps
    )

    mock_model.get_decoder_norms.assert_called_once()
    assert torch.isclose(penalty, expected_total_penalty)
    assert isinstance(current_lambda, float)
    assert abs(current_lambda - lambda_factor) < 1e-9
    assert penalty.device == device


def test_compute_sparsity_penalty_basic_2d(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Batch*Seq=2, Feat=3
    activations = {0: torch.tensor([[0.1, 0.0, 0.5], [0.2, 0.3, 0.0]], device=device)}
    current_step = 50
    total_steps = 100
    # mock_model.get_decoder_norms.reset_mock() # Remove reset from here

    # Get expected norms from mock
    local_decoder_norms = mock_model.get_decoder_norms()

    acts_flat = activations[0]  # Already flat
    weights = local_decoder_norms[0].unsqueeze(0).to(device)  # Shape (1, 3)
    weighted_acts = acts_flat * weights  # Shape (2, 3)

    # tanh penalty computation
    penalty_tensor = torch.tanh(mock_config.sparsity_c * weighted_acts)
    expected_penalty_sum = penalty_tensor.sum()

    lambda_factor = mock_config.sparsity_lambda * (current_step / total_steps)
    expected_total_penalty = lambda_factor * expected_penalty_sum

    mock_model.get_decoder_norms.reset_mock()  # Reset before the call under test
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, current_step, total_steps
    )

    mock_model.get_decoder_norms.assert_called_once()
    assert torch.isclose(penalty, expected_total_penalty)
    assert isinstance(current_lambda, float)
    assert abs(current_lambda - lambda_factor) < 1e-9
    assert penalty.device == device


def test_compute_sparsity_penalty_basic_1d(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Feat=1 (for layer 2 as per mock setup)
    # The loss function expects at least 2D, but handles 1D by unsqueezing
    activations = {2: torch.tensor([0.5], device=device)}
    current_step = 50
    total_steps = 100
    # mock_model.get_decoder_norms.reset_mock() # Remove reset from here

    # Get expected norms from mock
    local_decoder_norms = mock_model.get_decoder_norms()  # Norms for layer 2

    acts_flat = activations[2].unsqueeze(0)  # Shape [1, 1]
    weights = local_decoder_norms[2].unsqueeze(0).to(device)  # Shape [1, 1]
    weighted_acts = acts_flat * weights

    # tanh penalty computation
    penalty_tensor = torch.tanh(mock_config.sparsity_c * weighted_acts)
    expected_penalty_sum = penalty_tensor.sum()

    lambda_factor = mock_config.sparsity_lambda * (current_step / total_steps)
    expected_total_penalty = lambda_factor * expected_penalty_sum

    mock_model.get_decoder_norms.reset_mock()  # Reset before the call under test
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, current_step, total_steps
    )

    mock_model.get_decoder_norms.assert_called_once()
    assert torch.isclose(penalty, expected_total_penalty)
    assert isinstance(current_lambda, float)
    assert abs(current_lambda - lambda_factor) < 1e-9
    assert penalty.device == device


def test_compute_sparsity_penalty_schedule(loss_manager, mock_model, mock_config):
    device = mock_model.device
    activations = {0: torch.tensor([[[0.1, 0.0, 0.5]]], device=device)}  # B=1, S=1, F=3
    total_steps = 100
    mock_model.get_decoder_norms.reset_mock()
    mock_model.get_decoder_norms.return_value = {
        0: torch.tensor([0.5, 1.0, 1.5], device=device)
    }

    # Step 0
    penalty_0, lambda_0 = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 0, total_steps
    )
    assert torch.isclose(penalty_0, torch.tensor(0.0, device=device))
    assert lambda_0 == 0.0

    # Step 50
    mock_model.get_decoder_norms.reset_mock()  # Reset call count for next call
    penalty_50, lambda_50 = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 50, total_steps
    )
    expected_lambda_50 = mock_config.sparsity_lambda * (50 / total_steps)
    assert penalty_50 > 0
    assert abs(lambda_50 - expected_lambda_50) < 1e-9

    # Step 100
    mock_model.get_decoder_norms.reset_mock()
    penalty_100, lambda_100 = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 100, total_steps
    )
    expected_lambda_100 = mock_config.sparsity_lambda * (100 / total_steps)
    assert abs(lambda_100 - expected_lambda_100) < 1e-9

    # Penalty should scale linearly with lambda
    assert torch.isclose(penalty_100, penalty_50 * 2.0)


def test_compute_sparsity_penalty_empty(loss_manager, mock_model):
    device = mock_model.device
    activations = {}
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 50, 100
    )
    assert torch.equal(penalty, torch.tensor(0.0, device=device))
    assert current_lambda == 0.0
    mock_model.get_decoder_norms.assert_not_called()


def test_compute_sparsity_penalty_empty_tensor(loss_manager, mock_model):
    device = mock_model.device
    activations = {0: torch.empty((0, 3), device=device)}  # Empty tensor
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 50, 100
    )
    assert torch.equal(penalty, torch.tensor(0.0, device=device))
    # Lambda calculation still happens, but penalty is 0
    expected_lambda = loss_manager.config.sparsity_lambda * 0.5
    assert abs(current_lambda - expected_lambda) < 1e-9
    # get_decoder_norms might be called depending on implementation details before empty check
    # The current implementation checks numel() after getting norms, so it might be called.


def test_compute_sparsity_penalty_missing_norms(loss_manager, mock_model, mock_config):
    device = mock_model.device
    activations = {
        0: torch.tensor([[[0.1, 0.0, 0.5]]], device=device),
        99: torch.tensor([[[0.1, 0.2]]], device=device),
    }  # Layer 99 norms not in mock
    current_step = 50
    total_steps = 100
    # mock_model.get_decoder_norms.reset_mock() # Remove reset from here
    # Norms only available for layer 0
    mock_model.get_decoder_norms.return_value = {
        0: torch.tensor([0.5, 1.0, 1.5], device=device)
    }

    # Calculate expected penalty only for layer 0
    acts_flat_0 = activations[0].reshape(-1, 3)
    weights_0 = mock_model.get_decoder_norms()[0].unsqueeze(0).to(device)
    weighted_acts_0 = acts_flat_0 * weights_0
    penalty_tensor_0 = torch.tanh(mock_config.sparsity_c * weighted_acts_0)
    expected_penalty_sum = penalty_tensor_0.sum()
    lambda_factor = mock_config.sparsity_lambda * (current_step / total_steps)
    expected_total_penalty = lambda_factor * expected_penalty_sum

    mock_model.get_decoder_norms.reset_mock()  # Reset before the call under test
    penalty, current_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, activations, current_step, total_steps
    )

    # Norms should be fetched once
    mock_model.get_decoder_norms.assert_called_once()
    # Penalty should only include layer 0
    assert torch.isclose(penalty, expected_total_penalty)
    assert abs(current_lambda - lambda_factor) < 1e-9
    assert penalty.device == device


# --- Preactivation Loss Tests ---


def test_compute_preactivation_loss_basic(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Use 2D input: Batch*Seq=1, Dim=3
    inputs = {0: torch.tensor([[-0.5, 0.2, -0.1]], device=device)}

    mock_model.get_preactivations.reset_mock()

    # Calculate expected preacts using the mock's logic
    # Input [-0.5, 0.2, -0.1] -> Preacts [-0.5, 0.06, -0.18] (using layer 0 logic)
    expected_preacts = mock_model.get_preactivations.side_effect(inputs[0], 0)

    relu_neg_preacts = torch.relu(-expected_preacts)
    expected_penalty_sum = relu_neg_preacts.sum()
    num_elements = expected_preacts.numel()

    expected_total_loss = (
        mock_config.preactivation_coef * expected_penalty_sum / num_elements
        if num_elements > 0
        else 0.0
    )

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    mock_model.get_preactivations.assert_called_once()
    # Check call arguments carefully
    call_args = mock_model.get_preactivations.call_args
    assert torch.equal(call_args[0][0], inputs[0])
    assert call_args[0][1] == 0

    assert abs(loss.item() - expected_total_loss) < 1e-6
    assert loss.device == device


def test_compute_preactivation_loss_1d_input(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Use 1D input: Dim=1 (using layer 2 logic)
    inputs = {2: torch.tensor([-0.5], device=device)}
    mock_model.get_preactivations.reset_mock()

    # Calculate expected preacts using the mock's logic for layer 2
    # Input [-0.5] -> unsqueezed to [1,1] -> preacts [-0.3] (0.5*-0.5 - 0.05)
    # The mock returns shape [1, 1]
    expected_preacts = mock_model.get_preactivations.side_effect(
        inputs[2], 2
    )  # Shape [1, 1]

    relu_neg_preacts = torch.relu(-expected_preacts)
    expected_penalty_sum = relu_neg_preacts.sum()
    num_elements = expected_preacts.numel()  # Should be 1

    expected_total_loss = (
        mock_config.preactivation_coef * expected_penalty_sum / num_elements
        if num_elements > 0
        else 0.0
    )

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    mock_model.get_preactivations.assert_called_once()
    call_args = mock_model.get_preactivations.call_args
    # The loss function unsqueezes the 1D input before passing to get_preactivations
    assert torch.equal(call_args[0][0], inputs[2].unsqueeze(0))
    assert call_args[0][1] == 2

    assert abs(loss.item() - expected_total_loss) < 1e-6
    assert loss.device == device


def test_compute_preactivation_loss_all_positive(loss_manager, mock_model):
    device = mock_model.device
    # Input that results in positive preactivations for layer 0
    inputs = {0: torch.tensor([[0.5, 0.2, 0.15]], device=device)}  # Preacts > 0
    mock_model.get_preactivations.reset_mock()

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    mock_model.get_preactivations.assert_called_once()
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-8)
    assert loss.device == device


def test_compute_preactivation_loss_empty(loss_manager, mock_model):
    device = mock_model.device
    inputs = {}
    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)
    assert torch.equal(loss, torch.tensor(0.0, device=device))
    mock_model.get_preactivations.assert_not_called()
    assert loss.device == device


def test_compute_preactivation_loss_empty_tensor(loss_manager, mock_model):
    device = mock_model.device
    inputs = {0: torch.empty((0, 3), device=device)}
    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)
    assert torch.equal(loss, torch.tensor(0.0, device=device))
    # get_preactivations should not be called if numel is 0 before call
    # Current implementation checks numel() before calling get_preactivations
    mock_model.get_preactivations.assert_not_called()
    assert loss.device == device


def test_compute_preactivation_loss_exception(loss_manager, mock_model):
    device = mock_model.device
    inputs = {
        0: torch.tensor([[1.0, 2.0, 3.0]], device=device),
        1: torch.tensor([[4.0, 5.0]], device=device),
    }  # Use layer 1 input that will have preacts calculated
    mock_model.get_preactivations.reset_mock()
    # Make get_preactivations raise an exception for layer 0, but work for layer 1
    original_side_effect = mock_model.get_preactivations.side_effect

    def side_effect_with_exception(x, layer_idx):
        if layer_idx == 0:
            raise ValueError("Test Exception")
        else:
            return original_side_effect(x, layer_idx)

    mock_model.get_preactivations.side_effect = side_effect_with_exception

    # Calculate expected loss only for layer 1
    expected_preacts_1 = original_side_effect(inputs[1], 1)
    relu_neg_preacts_1 = torch.relu(-expected_preacts_1)
    expected_penalty_sum = relu_neg_preacts_1.sum()
    num_elements = expected_preacts_1.numel()
    expected_total_loss = (
        loss_manager.config.preactivation_coef * expected_penalty_sum / num_elements
        if num_elements > 0
        else 0.0
    )

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    # Should have been called for both layers, but failed on layer 0
    assert mock_model.get_preactivations.call_count == 2
    assert abs(loss.item() - expected_total_loss) < 1e-6
    assert loss.device == device

    # Restore original side effect if fixture is used elsewhere
    mock_model.get_preactivations.side_effect = original_side_effect


# --- Total Loss Tests ---


def test_compute_total_loss(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # B=1, S=1, Dim=4 for input/output
    inputs = {
        0: torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device),
        1: torch.tensor([[[5.0, 6.0, 7.0, 8.0]]], device=device),
    }
    # Target dimensions should match model output dimensions based on mock_call
    targets = {
        0: torch.tensor([[[1.1, 1.9, 3.1, 3.9]]], device=device),
        1: torch.tensor([[[4.9, 6.1, 7.0, 8.1]]], device=device),
    }
    current_step = 75
    total_steps = 150

    # Reset mocks before the call
    mock_model.reset_mock()
    mock_model.get_feature_activations.reset_mock()
    mock_model.get_decoder_norms.reset_mock()
    mock_model.get_preactivations.reset_mock()

    # Call the method under test
    total_loss, loss_dict = loss_manager.compute_total_loss(
        mock_model, inputs, targets, current_step, total_steps
    )

    # --- Assertions ---
    # 1. Check mock calls made *within* compute_total_loss
    mock_model.assert_called_once_with(inputs)  # Checks the instance call (__call__)
    mock_model.get_feature_activations.assert_called_once_with(inputs)
    # Sparsity penalty calls get_decoder_norms once inside compute_total_loss
    mock_model.get_decoder_norms.assert_called_once()
    # Preactivation loss calls get_preactivations once per layer in inputs
    assert mock_model.get_preactivations.call_count == len(inputs)
    expected_preact_calls = [call(inputs[0], 0), call(inputs[1], 1)]
    # Use assert_has_calls with any_order=True for flexibility if order isn't guaranteed
    mock_model.get_preactivations.assert_has_calls(
        expected_preact_calls, any_order=True
    )

    # 2. Manually calculate expected values *after* the call for verification
    #    Use the mocks configured in the fixture
    expected_predictions = mock_model.side_effect(inputs)
    expected_activations = mock_model.get_feature_activations.side_effect(inputs)

    # Create temporary LossManager instances to avoid state pollution if needed,
    # or ensure mocks are appropriately configured/reset. Here we reuse the fixture one.
    # Re-call individual components to get expected values
    expected_recon_loss = loss_manager.compute_reconstruction_loss(
        expected_predictions, targets
    )
    # Reset mocks that might be called again during manual calculation
    mock_model.get_decoder_norms.reset_mock()
    expected_sparsity_loss, expected_lambda = loss_manager.compute_sparsity_penalty(
        mock_model, expected_activations, current_step, total_steps
    )
    mock_model.get_preactivations.reset_mock()
    expected_preactivation_loss = loss_manager.compute_preactivation_loss(
        mock_model, inputs
    )
    expected_total_loss_val = (
        expected_recon_loss + expected_sparsity_loss + expected_preactivation_loss
    )

    # 3. Compare results
    assert torch.isclose(total_loss, expected_total_loss_val)
    assert total_loss.device == device
    assert isinstance(loss_dict, dict)
    assert "total" in loss_dict
    assert "reconstruction" in loss_dict
    assert "sparsity" in loss_dict
    assert "preactivation" in loss_dict

    # Check if the components roughly match (allow for float precision)
    assert abs(loss_dict["total"] - total_loss.item()) < 1e-6
    assert abs(loss_dict["reconstruction"] - expected_recon_loss.item()) < 1e-6
    assert abs(loss_dict["sparsity"] - expected_sparsity_loss.item()) < 1e-6
    assert abs(loss_dict["preactivation"] - expected_preactivation_loss.item()) < 1e-6

    # 4. Check if current_sparsity_lambda was updated
    assert abs(loss_manager.current_sparsity_lambda - expected_lambda) < 1e-9


def test_get_current_sparsity_lambda(loss_manager, mock_model, mock_config):
    device = mock_model.device
    # Initial value
    assert loss_manager.get_current_sparsity_lambda() == 0.0

    # Run total loss calculation to update lambda
    inputs = {0: torch.tensor([[[1.0, 2.0, 3.0]]], device=device)}
    targets = {0: torch.tensor([[[1.1, 1.9, 3.1]]], device=device)}
    current_step = 50
    total_steps = 100

    _, loss_dict = loss_manager.compute_total_loss(
        mock_model, inputs, targets, current_step, total_steps
    )

    expected_lambda = mock_config.sparsity_lambda * (current_step / total_steps)
    # The lambda stored should be the one calculated during the last total_loss call
    assert abs(loss_manager.get_current_sparsity_lambda() - expected_lambda) < 1e-9

    # Check that get just returns the value without recalculating
    mock_model.get_decoder_norms.reset_mock()
    retrieved_lambda = loss_manager.get_current_sparsity_lambda()
    assert abs(retrieved_lambda - expected_lambda) < 1e-9
    mock_model.get_decoder_norms.assert_not_called()  # Ensure get doesn't trigger calcs
