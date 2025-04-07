# tests/unit/training/test_losses.py
import pytest
import torch
from unittest.mock import MagicMock

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
def mock_model():
    model = MockCrossLayerTranscoder()

    # Setup default mock return values
    decoder_norms = {
        0: torch.tensor([0.5, 1.0, 1.5]),  # Example norms for layer 0, feature dim 3
        1: torch.tensor([1.0, 0.8]),  # Example norms for layer 1, feature dim 2
    }
    model.get_decoder_norms.return_value = decoder_norms

    # Mock preactivations based on input and layer_idx
    def mock_get_preactivations(x, layer_idx):
        # Simple mock: return input shifted and scaled
        if layer_idx == 0:
            # Batch=1, Seq=1, Feat=3
            return x * 0.8 - 0.1
        elif layer_idx == 1:
            # Batch=1, Seq=1, Feat=2
            return x * 1.2 + 0.2
        return torch.zeros_like(x)

    model.get_preactivations.side_effect = mock_get_preactivations

    # Mock feature activations based on input
    def mock_get_feature_activations(inputs):
        activations = {}
        if 0 in inputs:
            # Batch=1, Seq=1, Feat=3 - Should return this shape
            # Simulate feature extraction by slicing the result to the correct dimension
            raw_acts = torch.relu(inputs[0] * 0.8 - 0.1)
            activations[0] = raw_acts[:, :, :3]  # Slice to feature dimension 3
        if 1 in inputs:
            # Batch=1, Seq=1, Feat=2 - Should return this shape
            raw_acts = torch.relu(inputs[1] * 1.2 + 0.2)
            activations[1] = raw_acts[:, :, :2]  # Slice to feature dimension 2
        return activations

    model.get_feature_activations.side_effect = mock_get_feature_activations

    # Mock predictions based on input - make the *instance* callable
    def mock_call(inputs):
        predictions = {}
        if 0 in inputs:
            # Batch=1, Seq=1, Dim=4 (example output dim)
            predictions[0] = inputs[0][:, :, :4] * 0.9  # Mock transformation
        if 1 in inputs:
            # Batch=1, Seq=1, Dim=4
            predictions[1] = inputs[1][:, :, :4] * 1.1  # Mock transformation
        return predictions

    # Use model.side_effect for the instance __call__
    model.side_effect = mock_call

    return model


@pytest.fixture
def loss_manager(mock_config):
    return LossManager(mock_config)


# --- Test Cases ---


def test_loss_manager_init(loss_manager, mock_config):
    assert loss_manager.config == mock_config
    assert isinstance(loss_manager.mse_loss, torch.nn.MSELoss)


def test_compute_reconstruction_loss_basic(loss_manager):
    predicted = {0: torch.tensor([[1.0, 2.0]]), 1: torch.tensor([[3.0, 4.0]])}
    target = {0: torch.tensor([[1.1, 1.9]]), 1: torch.tensor([[3.2, 4.1]])}

    expected_loss_0 = torch.mean(torch.tensor([(1.0 - 1.1) ** 2, (2.0 - 1.9) ** 2]))
    expected_loss_1 = torch.mean(torch.tensor([(3.0 - 3.2) ** 2, (4.0 - 4.1) ** 2]))
    expected_total_loss = (expected_loss_0 + expected_loss_1) / 2

    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.isclose(loss, expected_total_loss)


def test_compute_reconstruction_loss_mismatched_keys(loss_manager):
    predicted = {0: torch.tensor([[1.0]]), 1: torch.tensor([[3.0]])}
    target = {
        0: torch.tensor([[1.1]]),
        2: torch.tensor([[5.0]]),
    }  # Layer 1 missing in target, layer 2 missing in predicted

    expected_loss_0 = torch.mean(torch.tensor([(1.0 - 1.1) ** 2]))
    # Layer 1 loss is not calculated as it's not in target
    expected_total_loss = expected_loss_0 / 1  # Only one layer (layer 0) contributes

    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.isclose(loss, expected_total_loss)


def test_compute_reconstruction_loss_empty(loss_manager):
    predicted = {}
    target = {}
    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.equal(loss, torch.tensor(0.0))

    predicted = {0: torch.tensor([[1.0]])}
    target = {}
    loss = loss_manager.compute_reconstruction_loss(predicted, target)
    assert torch.equal(loss, torch.tensor(0.0))  # No matching keys


def test_compute_sparsity_penalty_basic(loss_manager, mock_model, mock_config):
    # Batch=1, Seq=2, Feat=3
    activations = {0: torch.tensor([[[0.1, 0.0, 0.5], [0.2, 0.3, 0.0]]])}
    current_step = 50
    total_steps = 100

    # Define the norms locally for expectation calculation, don't call the mock here
    local_decoder_norms = {0: torch.tensor([0.5, 1.0, 1.5])}
    mock_model.get_decoder_norms.return_value = (
        local_decoder_norms  # Set the return value
    )

    acts_flat = activations[0].reshape(-1, 3)  # Shape (2, 3)
    weights = local_decoder_norms[0].unsqueeze(0)  # Shape (1, 3) - Use local definition
    weighted_acts = acts_flat * weights  # Shape (2, 3)

    penalty_tensor = torch.tanh(mock_config.sparsity_c * weighted_acts)
    expected_penalty_sum = penalty_tensor.sum()

    lambda_factor = mock_config.sparsity_lambda * (current_step / total_steps)
    total_elements = activations[0].numel()
    expected_total_penalty = lambda_factor * expected_penalty_sum / total_elements

    penalty = loss_manager.compute_sparsity_penalty(
        mock_model, activations, current_step, total_steps
    )

    mock_model.get_decoder_norms.assert_called_once()
    assert torch.isclose(penalty, expected_total_penalty)


def test_compute_sparsity_penalty_schedule(loss_manager, mock_model, mock_config):
    activations = {0: torch.tensor([[[0.1, 0.0, 0.5]]])}  # Batch=1, Seq=1, Feat=3
    mock_model.get_decoder_norms.return_value = {0: torch.tensor([0.5, 1.0, 1.5])}

    # Step 0
    penalty_0 = loss_manager.compute_sparsity_penalty(mock_model, activations, 0, 100)
    assert torch.isclose(penalty_0, torch.tensor(0.0))

    # Step 50
    penalty_50 = loss_manager.compute_sparsity_penalty(mock_model, activations, 50, 100)

    # Step 100
    penalty_100 = loss_manager.compute_sparsity_penalty(
        mock_model, activations, 100, 100
    )

    assert penalty_50 > 0
    assert torch.isclose(penalty_100, penalty_50 * 2.0)  # Linear ramp-up


def test_compute_sparsity_penalty_empty(loss_manager, mock_model):
    activations = {}
    penalty = loss_manager.compute_sparsity_penalty(mock_model, activations, 50, 100)
    assert torch.equal(penalty, torch.tensor(0.0))
    mock_model.get_decoder_norms.assert_not_called()


def test_compute_preactivation_loss_basic(loss_manager, mock_model, mock_config):
    # Batch=1, Seq=1, Dim=3
    inputs = {0: torch.tensor([[[-0.5, 0.2, -0.1]]])}

    # Mock get_preactivations for layer 0
    preacts = torch.tensor([[[-0.5 * 0.8 - 0.1, 0.2 * 0.8 - 0.1, -0.1 * 0.8 - 0.1]]])
    preacts = preacts.to(inputs[0].device)  # Ensure device match
    mock_model.get_preactivations.return_value = preacts

    relu_neg_preacts = torch.relu(-preacts)
    expected_penalty_sum = relu_neg_preacts.sum()
    num_elements = preacts.numel()

    expected_total_loss = (
        mock_config.preactivation_coef * expected_penalty_sum / num_elements
    )

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    mock_model.get_preactivations.assert_called_once_with(inputs[0], 0)
    assert torch.isclose(loss, expected_total_loss)


def test_compute_preactivation_loss_all_positive(loss_manager, mock_model):
    # Adjusted input slightly so all preactivations are positive with the mock logic
    inputs = {0: torch.tensor([[[0.5, 0.2, 0.15]]])}
    # 0.15 * 0.8 - 0.1 = 0.12 - 0.1 = 0.02 > 0

    # Mock get_preactivations to return positive values
    preacts = torch.tensor([[[0.1, 0.2, 0.3]]])
    preacts = preacts.to(inputs[0].device)
    # We rely on the side_effect configured in the fixture for the actual
    # calculation logic. But for *this specific test*, we can override the
    # return value if needed, or just adjust the input as done above. Let's
    # stick with adjusting input.
    # mock_model.get_preactivations.return_value = preacts
    # Overriding the return value is also an option

    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-8)  # Added tolerance


def test_compute_preactivation_loss_empty(loss_manager, mock_model):
    inputs = {}
    loss = loss_manager.compute_preactivation_loss(mock_model, inputs)
    assert torch.equal(loss, torch.tensor(0.0))
    mock_model.get_preactivations.assert_not_called()


def test_compute_total_loss(loss_manager, mock_model, mock_config):
    # B=1, S=1, Dim=4 for input/output, Feat=3 for layer 0, Feat=2 for layer 1
    inputs = {
        0: torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        1: torch.tensor([[[5.0, 6.0, 7.0, 8.0]]]),
    }
    targets = {
        0: torch.tensor([[[1.1, 1.9, 3.1, 3.9]]]),
        1: torch.tensor([[[4.9, 6.1, 7.0, 8.1]]]),
    }
    current_step = 75
    total_steps = 150

    # Reset mocks before the call to ensure counts are only from this execution
    mock_model.reset_mock()
    mock_model.get_feature_activations.reset_mock()
    mock_model.get_decoder_norms.reset_mock()
    mock_model.get_preactivations.reset_mock()

    # Call the method under test
    total_loss, loss_dict = loss_manager.compute_total_loss(
        mock_model, inputs, targets, current_step, total_steps
    )

    # Assertions
    # Check that the mocks were called correctly *within* compute_total_loss
    mock_model.assert_called_once_with(inputs)  # Checks the instance call
    mock_model.get_feature_activations.assert_called_once_with(inputs)
    mock_model.get_decoder_norms.assert_called_once()
    # get_preactivations is called once per layer in inputs
    assert mock_model.get_preactivations.call_count == len(inputs)
    # Check get_preactivations args manually due to tensor comparison issues
    calls = mock_model.get_preactivations.call_args_list
    expected_calls = [(inputs[0], 0), (inputs[1], 1)]
    # Check if the actual calls match the expected calls, ignoring order
    matched_calls = 0
    actual_call_args = [(call.args[0], call.args[1]) for call in calls]
    for expected_arg_tuple in expected_calls:
        for actual_arg_tuple in actual_call_args:
            if (
                torch.equal(expected_arg_tuple[0], actual_arg_tuple[0])
                and expected_arg_tuple[1] == actual_arg_tuple[1]
            ):
                matched_calls += 1
                # Avoid matching the same actual call twice
                actual_call_args.remove(actual_arg_tuple)
                break
    assert matched_calls == len(
        expected_calls
    ), f"Expected calls with args {expected_calls}, but got {calls}"

    # Manually calculate expected values *after* the call for verification
    # (using the same logic as the tested function and mocks)
    expected_predictions = mock_model.side_effect(inputs)  # Use the mock's side effect
    expected_activations = mock_model.get_feature_activations.side_effect(inputs)
    expected_recon_loss = loss_manager.compute_reconstruction_loss(
        expected_predictions, targets
    )
    expected_sparsity_loss = loss_manager.compute_sparsity_penalty(
        mock_model, expected_activations, current_step, total_steps
    )
    # Need to reset mocks again as manual calc will call them
    mock_model.get_preactivations.reset_mock()
    expected_preactivation_loss = loss_manager.compute_preactivation_loss(
        mock_model, inputs
    )
    expected_total_loss_val = (
        expected_recon_loss + expected_sparsity_loss + expected_preactivation_loss
    )

    assert torch.isclose(total_loss, expected_total_loss_val)
    assert isinstance(loss_dict, dict)
    assert "total" in loss_dict
    assert "reconstruction" in loss_dict
    assert "sparsity" in loss_dict
    assert "preactivation" in loss_dict

    # Check if the components roughly match (allow for float precision)
    assert abs(loss_dict["total"] - total_loss.item()) < 1e-6
    # Compare dict values to manually calculated expected values
    assert abs(loss_dict["reconstruction"] - expected_recon_loss.item()) < 1e-6
    assert abs(loss_dict["sparsity"] - expected_sparsity_loss.item()) < 1e-6
    assert abs(loss_dict["preactivation"] - expected_preactivation_loss.item()) < 1e-6
