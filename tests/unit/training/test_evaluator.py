import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
import time
from typing import Dict

# Imports from the module under test and dependencies
from clt.training.evaluator import CLTEvaluator, _format_elapsed_time
from clt.models.clt import CrossLayerTranscoder
from clt.config import CLTConfig

# Constants for test configuration
NUM_LAYERS = 2
NUM_FEATURES = 4
D_MODEL = 8
BATCH_TOKENS = 10


# --- Fixtures ---


@pytest.fixture
def device():
    """Provides the device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def mock_clt_config():
    """Provides a mock CLTConfig."""
    config = MagicMock(spec=CLTConfig)
    config.num_layers = NUM_LAYERS
    config.num_features = NUM_FEATURES
    config.d_model = D_MODEL
    return config


@pytest.fixture
def mock_clt_model(mock_clt_config, device):
    """Provides a mock CrossLayerTranscoder model."""
    model = MagicMock(spec=CrossLayerTranscoder)
    model.config = mock_clt_config
    model.device = device

    # Mock the __call__ method (reconstruction)
    def mock_call(inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        reconstructions = {}
        for layer_idx, inp in inputs.items():
            # Simple identity reconstruction for testing
            reconstructions[layer_idx] = inp.clone().detach()
        return reconstructions

    # Corrected assignment for side_effect
    model.__call__ = MagicMock(side_effect=mock_call)

    # Mock the get_feature_activations method
    def mock_get_feature_activations(
        inputs: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        activations = {}
        for layer_idx, inp in inputs.items():
            # Generate dummy activations (batch_tokens, num_features)
            # Let's make layer 0 sparse, layer 1 dense for testing
            if layer_idx == 0:
                # 50% active features on average
                acts = torch.rand(inp.shape[0], NUM_FEATURES, device=device) * 2 - 0.5
                acts = torch.relu(acts)
            else:
                # Mostly active features
                acts = torch.rand(inp.shape[0], NUM_FEATURES, device=device) + 0.1
            activations[layer_idx] = acts
        return activations

    model.get_feature_activations.side_effect = mock_get_feature_activations

    return model


@pytest.fixture
def evaluator(mock_clt_model, device):
    """Provides an instance of CLTEvaluator."""
    return CLTEvaluator(model=mock_clt_model, device=device, start_time=time.time())


@pytest.fixture
def sample_inputs(device):
    """Provides sample input activations."""
    inputs = {}
    for i in range(NUM_LAYERS):
        # Shape: [batch_tokens, d_model]
        inputs[i] = torch.randn(BATCH_TOKENS, D_MODEL, device=device)
    return inputs


@pytest.fixture
def sample_targets(device):
    """Provides sample target activations (same as inputs for simple test)."""
    targets = {}
    for i in range(NUM_LAYERS):
        # Shape: [batch_tokens, d_model]
        targets[i] = torch.randn(BATCH_TOKENS, D_MODEL, device=device)
    return targets


@pytest.fixture
def sample_activations(device):
    """Provides sample feature activations."""
    activations = {}
    # Layer 0: sparse
    acts0 = torch.zeros(BATCH_TOKENS, NUM_FEATURES, device=device)
    acts0[0, 0] = 1.0
    acts0[1, 1] = 1.0
    activations[0] = acts0
    # Layer 1: dense
    acts1 = torch.ones(BATCH_TOKENS, NUM_FEATURES, device=device)
    activations[1] = acts1
    return activations


# --- Test Helper Functions ---


def test_format_elapsed_time():
    """Tests the _format_elapsed_time helper function."""
    assert _format_elapsed_time(50) == "00:50"
    assert _format_elapsed_time(125.5) == "02:05"
    assert _format_elapsed_time(3600) == "01:00:00"
    assert _format_elapsed_time(3725) == "01:02:05"
    assert _format_elapsed_time(86400 + 3600 + 120 + 5) == "25:02:05"


# --- Test Static Methods ---


def test_log_density(device):
    """Tests the _log_density static method."""
    density = torch.tensor([0.0, 0.1, 1.0, 1e-12], device=device, dtype=torch.float32)
    log_density = CLTEvaluator._log_density(density, eps=1e-10)
    # Ensure expected tensor has matching dtype (float32)
    # Corrected expected value for the 1e-12 case
    expected = torch.tensor(
        [-10.0, np.log10(0.1 + 1e-10), 0.0, np.log10(1e-12 + 1e-10)],
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(log_density, expected, atol=1e-6)
    # Test with zero epsilon
    log_density_no_eps = CLTEvaluator._log_density(density, eps=0)
    # Ensure expected tensor has matching dtype (float32)
    # Corrected expected value for log10(1e-12) which is -12.0
    expected_no_eps = torch.tensor(
        [float("-inf"), np.log10(0.1), 0.0, -12.0],  # Changed last element from -inf
        device=device,
        dtype=torch.float32,
    )
    assert torch.allclose(log_density_no_eps, expected_no_eps, equal_nan=True)


def test_calculate_aggregate_metric():
    """Tests the _calculate_aggregate_metric static method."""
    # Empty input
    assert CLTEvaluator._calculate_aggregate_metric({}) is None
    # Single layer
    data1 = {"layer_0": [1.0, 2.0, 3.0]}
    assert CLTEvaluator._calculate_aggregate_metric(data1) == pytest.approx(2.0)
    # Multiple layers
    data2 = {"layer_0": [1.0, 2.0], "layer_1": [3.0, 4.0]}
    assert CLTEvaluator._calculate_aggregate_metric(data2) == pytest.approx(2.5)
    # Layer with empty list
    data3 = {"layer_0": [], "layer_1": [1.0, 3.0]}
    assert CLTEvaluator._calculate_aggregate_metric(data3) == pytest.approx(2.0)
    # All empty lists
    data4 = {"layer_0": [], "layer_1": []}
    assert CLTEvaluator._calculate_aggregate_metric(data4) is None


def test_calculate_aggregate_histogram_data():
    """Tests the _calculate_aggregate_histogram_data static method."""
    # Empty input
    assert CLTEvaluator._calculate_aggregate_histogram_data({}) == []
    # Single layer
    data1 = {"layer_0": [1.0, 2.0, 3.0]}
    assert CLTEvaluator._calculate_aggregate_histogram_data(data1) == [1.0, 2.0, 3.0]
    # Multiple layers
    data2 = {"layer_0": [1.0, 2.0], "layer_1": [3.0, 4.0]}
    assert CLTEvaluator._calculate_aggregate_histogram_data(data2) == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    # Layer with empty list
    data3 = {"layer_0": [], "layer_1": [1.0, 3.0]}
    assert CLTEvaluator._calculate_aggregate_histogram_data(data3) == [1.0, 3.0]
    # All empty lists
    data4 = {"layer_0": [], "layer_1": []}
    assert CLTEvaluator._calculate_aggregate_histogram_data(data4) == []


# --- Test Private Calculation Methods ---


def test_compute_sparsity(evaluator, sample_activations, device):
    """Tests the _compute_sparsity method."""
    metrics = evaluator._compute_sparsity(sample_activations)

    # Expected values based on sample_activations
    # Layer 0: 2 activations out of BATCH_TOKENS * NUM_FEATURES = 10 * 4 = 40
    #          L0 per token: (1 activation/token) for 2 tokens,
    #                        (0 activation/token) for 8 tokens. Avg = 2/10 = 0.2
    # Layer 1: All active. BATCH_TOKENS * NUM_FEATURES = 40 activations.
    #          Avg L0 per token = 4.0
    # Total L0 = 0.2 + 4.0 = 4.2
    # Avg L0 = 4.2 / 2 = 2.1
    # Sparsity Fraction = 1 - (Avg L0 / Total Features)
    #                   = 1 - (2.1 / 4) = 1 - 0.525 = 0.475

    assert metrics["sparsity/total_l0"] == pytest.approx(4.2)
    assert metrics["sparsity/avg_l0"] == pytest.approx(2.1)
    assert metrics["sparsity/sparsity_fraction"] == pytest.approx(
        1.0 - (2.1 / NUM_FEATURES)
    )
    # Avg L0 for layer 0
    assert metrics["layerwise/l0"]["layer_0"] == pytest.approx(2 / BATCH_TOKENS)
    # Avg L0 for layer 1
    assert metrics["layerwise/l0"]["layer_1"] == pytest.approx(NUM_FEATURES)

    # Test with empty activations
    empty_metrics = evaluator._compute_sparsity({})
    assert empty_metrics["sparsity/total_l0"] == 0.0
    assert empty_metrics["sparsity/avg_l0"] == 0.0
    assert empty_metrics["sparsity/sparsity_fraction"] == 1.0
    assert empty_metrics["layerwise/l0"] == {
        f"layer_{i}": 0.0 for i in range(NUM_LAYERS)
    }

    # Test with activations containing empty tensors
    activations_with_empty = {
        0: torch.randn(BATCH_TOKENS, NUM_FEATURES, device=device),
        1: torch.empty((0, NUM_FEATURES), device=device),  # Empty tensor
    }
    metrics_with_empty = evaluator._compute_sparsity(activations_with_empty)
    assert "layer_1" in metrics_with_empty["layerwise/l0"]
    assert metrics_with_empty["layerwise/l0"]["layer_1"] == 0.0
    assert metrics_with_empty["sparsity/avg_l0"] > 0  # Only layer 0 contributes


def test_compute_reconstruction_metrics(evaluator, device):
    """Tests the _compute_reconstruction_metrics method."""
    targets = {
        0: torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        1: torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device),
    }
    # Perfect reconstruction
    recons_perfect = {k: v.clone() for k, v in targets.items()}
    metrics_perfect = evaluator._compute_reconstruction_metrics(targets, recons_perfect)
    assert metrics_perfect["reconstruction/total_mse"] == pytest.approx(0.0)
    assert metrics_perfect["reconstruction/explained_variance"] == pytest.approx(1.0)

    # Zero reconstruction
    recons_zero = {k: torch.zeros_like(v) for k, v in targets.items()}
    metrics_zero = evaluator._compute_reconstruction_metrics(targets, recons_zero)
    expected_mse = 25.5
    assert metrics_zero["reconstruction/total_mse"] == pytest.approx(expected_mse)
    # EV = 1 - Var(Target - 0) / Var(Target) = 1 - Var(Target) / Var(Target) = 0
    assert metrics_zero["reconstruction/explained_variance"] == pytest.approx(
        0.0
    )  # Should be approx 0

    # Partial reconstruction
    recons_partial = {
        0: torch.tensor([[1.1, 1.9], [3.1, 3.9]], device=device),
        1: torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device),  # Perfect for layer 1
    }
    metrics_partial = evaluator._compute_reconstruction_metrics(targets, recons_partial)
    expected_mse_partial = (
        F.mse_loss(targets[0], recons_partial[0]).item()
        + F.mse_loss(targets[1], recons_partial[1]).item()
    ) / 2
    assert metrics_partial["reconstruction/total_mse"] == pytest.approx(
        expected_mse_partial
    )
    assert 0.0 < metrics_partial["reconstruction/explained_variance"] < 1.0

    # Test with zero variance target
    targets_zero_var = {0: torch.ones((2, 2), device=device) * 3}
    recons_zero_var = {0: torch.ones((2, 2), device=device) * 3.1}
    metrics_zero_var = evaluator._compute_reconstruction_metrics(
        targets_zero_var, recons_zero_var
    )
    assert metrics_zero_var["reconstruction/total_mse"] == pytest.approx(
        0.1**2, abs=1e-5
    )
    # EV = 1 - Var(Err)/Var(Target) -> 1 - 0/0. If error var is 0, EV should be 1.
    # Corrected assertion
    assert metrics_zero_var["reconstruction/explained_variance"] == pytest.approx(1.0)

    # Test with zero variance target and perfect recon
    targets_zero_var_perf = {0: torch.ones((2, 2), device=device) * 3}
    recons_zero_var_perf = {0: torch.ones((2, 2), device=device) * 3}
    metrics_zero_var_perf = evaluator._compute_reconstruction_metrics(
        targets_zero_var_perf, recons_zero_var_perf
    )
    assert metrics_zero_var_perf["reconstruction/total_mse"] == pytest.approx(0.0)
    # EV = 1 - Var(0)/0 -> Should be 1
    assert metrics_zero_var_perf["reconstruction/explained_variance"] == pytest.approx(
        1.0
    )

    # Test with missing layer in reconstruction
    targets_missing = {0: torch.randn(2, 2), 1: torch.randn(2, 2)}
    recons_missing = {0: torch.randn(2, 2)}  # Missing layer 1
    metrics_missing = evaluator._compute_reconstruction_metrics(
        targets_missing, recons_missing
    )
    assert metrics_missing["reconstruction/total_mse"] > 0  # Only layer 0 contributes
    # Corrected assertion key
    assert "reconstruction/explained_variance" in metrics_missing


def test_compute_feature_density(evaluator, sample_activations, device):
    """Tests the _compute_feature_density method."""
    metrics = evaluator._compute_feature_density(sample_activations)

    assert "layerwise/log_feature_density" in metrics
    assert "layerwise/consistent_activation_heuristic" in metrics

    # --- Layer 0 (Sparse) ---
    # Density: Feature 0 active in 1/10 tokens, Feature 1 active in 1/10 tokens,
    #          others 0/10
    expected_density0 = torch.tensor([0.1, 0.1, 0.0, 0.0], device=device)
    expected_log_density0 = CLTEvaluator._log_density(expected_density0).tolist()
    assert metrics["layerwise/log_feature_density"]["layer_0"] == pytest.approx(
        expected_log_density0
    )

    # Heuristic:
    # Feature 0: 1 total activation / 1 prompt active = 1
    # Feature 1: 1 total activation / 1 prompt active = 1
    # Feature 2: 0 total activations / 0 prompts active = 0 / eps -> ~0
    # Feature 3: 0 total activations / 0 prompts active = 0 / eps -> ~0
    expected_heuristic0 = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)
    assert metrics["layerwise/consistent_activation_heuristic"][
        "layer_0"
    ] == pytest.approx(expected_heuristic0.tolist())

    # --- Layer 1 (Dense) ---
    # Density: All features active in 10/10 tokens = 1.0
    expected_density1 = torch.ones(NUM_FEATURES, device=device)
    expected_log_density1 = CLTEvaluator._log_density(expected_density1).tolist()
    # Should be list of 0.0
    assert metrics["layerwise/log_feature_density"]["layer_1"] == pytest.approx(
        expected_log_density1
    )

    # Heuristic:
    # Each feature: BATCH_TOKENS total activations / 1 prompt active = 10 / 1 = 10
    expected_heuristic1 = torch.ones(NUM_FEATURES, device=device) * BATCH_TOKENS
    assert metrics["layerwise/consistent_activation_heuristic"][
        "layer_1"
    ] == pytest.approx(expected_heuristic1.tolist())

    # Test with empty activations
    empty_metrics = evaluator._compute_feature_density({})
    assert empty_metrics["layerwise/log_feature_density"] == {}
    assert empty_metrics["layerwise/consistent_activation_heuristic"] == {}


def test_compute_dead_neuron_metrics(evaluator, device, mock_clt_config):
    """Tests the _compute_dead_neuron_metrics method."""
    # --- Test with valid mask ---
    # Mask: layer 0 has 1 dead, layer 1 has 2 dead
    dead_mask = torch.zeros(NUM_LAYERS, NUM_FEATURES, dtype=torch.bool, device=device)
    dead_mask[0, 1] = True
    dead_mask[1, 0] = True
    dead_mask[1, 2] = True

    metrics = evaluator._compute_dead_neuron_metrics(dead_mask)
    assert "layerwise/dead_features" in metrics
    assert metrics["layerwise/dead_features"]["layer_0"] == 1
    assert metrics["layerwise/dead_features"]["layer_1"] == 2
    # Total is calculated in compute_metrics, not here

    # --- Test with all dead ---
    all_dead_mask = torch.ones(
        NUM_LAYERS, NUM_FEATURES, dtype=torch.bool, device=device
    )
    metrics_all_dead = evaluator._compute_dead_neuron_metrics(all_dead_mask)
    assert metrics_all_dead["layerwise/dead_features"]["layer_0"] == NUM_FEATURES
    assert metrics_all_dead["layerwise/dead_features"]["layer_1"] == NUM_FEATURES

    # --- Test with all alive ---
    all_alive_mask = torch.zeros(
        NUM_LAYERS, NUM_FEATURES, dtype=torch.bool, device=device
    )
    metrics_all_alive = evaluator._compute_dead_neuron_metrics(all_alive_mask)
    assert metrics_all_alive["layerwise/dead_features"]["layer_0"] == 0
    assert metrics_all_alive["layerwise/dead_features"]["layer_1"] == 0

    # --- Test with None mask ---
    metrics_none = evaluator._compute_dead_neuron_metrics(None)
    assert "layerwise/dead_features" in metrics_none
    assert metrics_none["layerwise/dead_features"] == {}

    # --- Test with incorrect shape mask ---
    wrong_shape_mask = torch.zeros(NUM_LAYERS + 1, NUM_FEATURES, device=device)
    # Should print a warning, but return default empty dict structure
    with patch("builtins.print") as mock_print:
        metrics_wrong_shape = evaluator._compute_dead_neuron_metrics(wrong_shape_mask)
        assert "layerwise/dead_features" in metrics_wrong_shape
        assert metrics_wrong_shape["layerwise/dead_features"] == {}
        mock_print.assert_called_once()
        assert (
            "Warning: Received dead_neuron_mask with unexpected shape"
            in mock_print.call_args[0][0]
        )


# --- Test Main Method ---


def test_compute_metrics_integration(evaluator, sample_inputs, sample_targets, device):
    """Tests the compute_metrics method integration."""
    # --- Mock internal methods to control their output ---
    # We want to check if compute_metrics correctly aggregates results
    mock_sparsity_result = {
        "sparsity/total_l0": 4.2,
        "sparsity/avg_l0": 2.1,
        "sparsity/sparsity_fraction": 0.475,
        "layerwise/l0": {"layer_0": 0.2, "layer_1": 4.0},
    }
    mock_recon_result = {
        "reconstruction/explained_variance": 0.95,
        "reconstruction/total_mse": 0.1,
    }
    mock_density_result = {
        "layerwise/log_feature_density": {
            "layer_0": [
                -1.0,
                -1.0,
                -10.0,
                -10.0,
            ],  # Derived from density [0.1, 0.1, 0, 0]
            "layer_1": [0.0, 0.0, 0.0, 0.0],  # Derived from density [1, 1, 1, 1]
        },
        "layerwise/consistent_activation_heuristic": {
            "layer_0": [1.0, 1.0, 0.0, 0.0],
            "layer_1": [10.0, 10.0, 10.0, 10.0],
        },
    }
    mock_dead_result = {
        "layerwise/dead_features": {"layer_0": 1, "layer_1": 0},
    }
    # Dead mask to produce the mock_dead_result
    dead_mask = torch.zeros(NUM_LAYERS, NUM_FEATURES, dtype=torch.bool, device=device)
    dead_mask[0, 0] = True  # One dead feature in layer 0

    with patch.object(
        evaluator, "_compute_sparsity", return_value=mock_sparsity_result
    ) as mock_sparsity, patch.object(
        evaluator, "_compute_reconstruction_metrics", return_value=mock_recon_result
    ) as mock_recon, patch.object(
        evaluator, "_compute_feature_density", return_value=mock_density_result
    ) as mock_density, patch.object(
        evaluator, "_compute_dead_neuron_metrics", return_value=mock_dead_result
    ) as mock_dead:

        # Call the main method
        all_metrics = evaluator.compute_metrics(
            sample_inputs, sample_targets, dead_mask
        )

        # --- Assertions ---
        # 1. Check if internal methods were called (mocks can verify this implicitly)
        mock_sparsity.assert_called_once()
        mock_recon.assert_called_once()
        mock_density.assert_called_once()
        mock_dead.assert_called_once_with(dead_mask)

        # 2. Check if the output dict contains keys from all mocked results
        assert "sparsity/total_l0" in all_metrics
        assert "reconstruction/explained_variance" in all_metrics
        assert "layerwise/log_feature_density" in all_metrics
        assert "layerwise/dead_features" in all_metrics

        # 3. Check aggregate calculations performed by compute_metrics
        # Aggregate dead features
        assert "dead_features/total_eval" in all_metrics
        assert (
            all_metrics["dead_features/total_eval"] == 1
        )  # Sum of layerwise dead features

        # Aggregate density mean (mean of log densities)
        expected_log_density_mean = np.mean(
            [-1.0, -1.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0]
        )  # (-22) / 8 = -2.75
        assert "sparsity/feature_density_mean" in all_metrics
        assert all_metrics["sparsity/feature_density_mean"] == pytest.approx(
            expected_log_density_mean
        )

        # Aggregate heuristic mean
        expected_heuristic_mean = np.mean(
            [1.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]
        )  # (42) / 8 = 5.25
        assert "sparsity/consistent_activation_heuristic_mean" in all_metrics
        assert all_metrics[
            "sparsity/consistent_activation_heuristic_mean"
        ] == pytest.approx(expected_heuristic_mean)

        # Aggregate histogram data
        expected_log_density_hist = [-1.0, -1.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0]
        assert "sparsity/log_feature_density_agg_hist" in all_metrics
        assert (
            all_metrics["sparsity/log_feature_density_agg_hist"]
            == expected_log_density_hist
        )

        expected_heuristic_hist = [1.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0]
        assert "sparsity/consistent_activation_heuristic_agg_hist" in all_metrics
        assert (
            all_metrics["sparsity/consistent_activation_heuristic_agg_hist"]
            == expected_heuristic_hist
        )

        # 4. Check if values are copied correctly
        assert all_metrics["sparsity/avg_l0"] == mock_sparsity_result["sparsity/avg_l0"]
        assert (
            all_metrics["reconstruction/total_mse"]
            == mock_recon_result["reconstruction/total_mse"]
        )
        assert (
            all_metrics["layerwise/dead_features"]
            == mock_dead_result["layerwise/dead_features"]
        )


def test_compute_metrics_integration_no_dead_mask(
    evaluator, sample_inputs, sample_targets
):
    """Tests compute_metrics without providing a dead neuron mask."""
    # Configure mock return value for when mask is None
    expected_return_when_none = {"layerwise/dead_features": {}}
    with patch.object(
        evaluator,
        "_compute_dead_neuron_metrics",
        return_value=expected_return_when_none,
    ) as mock_dead:
        # Call without dead_neuron_mask
        all_metrics = evaluator.compute_metrics(
            sample_inputs, sample_targets, dead_neuron_mask=None
        )

        # Check that _compute_dead_neuron_metrics was called with None
        mock_dead.assert_called_once_with(None)
        # Check that the resulting dead feature counts are zero or empty
        assert "layerwise/dead_features" in all_metrics
        assert all_metrics["layerwise/dead_features"] == {}
        assert "dead_features/total_eval" in all_metrics
        assert all_metrics["dead_features/total_eval"] == 0


# Add tests for edge cases like empty inputs/targets if needed
def test_compute_metrics_empty_input(evaluator):
    """Tests compute_metrics with empty input dictionaries."""
    empty_inputs: Dict[int, torch.Tensor] = {}
    empty_targets: Dict[int, torch.Tensor] = {}

    # Mock model behavior for empty inputs
    evaluator.model.__call__.return_value = {}
    evaluator.model.get_feature_activations.return_value = {}

    all_metrics = evaluator.compute_metrics(empty_inputs, empty_targets)

    # Check for sensible default/zero values
    assert all_metrics.get("reconstruction/explained_variance") == 0.0
    assert all_metrics.get("reconstruction/total_mse") == 0.0
    assert all_metrics.get("sparsity/avg_l0") == 0.0
    assert all_metrics.get("sparsity/sparsity_fraction") == 1.0
    assert all_metrics.get("dead_features/total_eval") == 0
    assert (
        "sparsity/feature_density_mean" not in all_metrics
    )  # Should be None internally, so key omitted
    assert "sparsity/consistent_activation_heuristic_mean" not in all_metrics
    assert all_metrics.get("layerwise/l0") == {
        f"layer_{i}": 0.0 for i in range(NUM_LAYERS)
    }
    assert all_metrics.get("layerwise/dead_features") == {}
    assert all_metrics.get("layerwise/log_feature_density") == {}
    assert all_metrics.get("layerwise/consistent_activation_heuristic") == {}
