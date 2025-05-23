import torch
import pytest
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
import logging

from clt.activations.registry import (
    ACTIVATION_REGISTRY,
    register_activation_fn,
    get_activation_fn,
    relu_activation,
    jumprelu_activation,
    batchtopk_per_layer_activation,
    topk_per_layer_activation,
)
from clt.config import CLTConfig


# Helper to clear registry for isolated tests
@pytest.fixture(autouse=True)
def clear_registry_for_test():
    original_registry_items = list(ACTIVATION_REGISTRY.items())
    ACTIVATION_REGISTRY.clear()
    # Re-register default ones that are imported directly in the test module
    # This ensures they are available for tests that might use get_activation_fn implicitly
    ACTIVATION_REGISTRY["relu"] = relu_activation
    ACTIVATION_REGISTRY["jumprelu"] = jumprelu_activation
    ACTIVATION_REGISTRY["batchtopk"] = batchtopk_per_layer_activation
    ACTIVATION_REGISTRY["topk"] = topk_per_layer_activation
    yield
    ACTIVATION_REGISTRY.clear()
    for name, fn in original_registry_items:
        ACTIVATION_REGISTRY[name] = fn


def test_register_activation_fn(caplog):
    @register_activation_fn("test_act")
    def dummy_activation(model, preact, layer_idx):
        return preact * 2

    assert "test_act" in ACTIVATION_REGISTRY
    assert ACTIVATION_REGISTRY["test_act"] == dummy_activation

    # Test overwriting (check for log message)
    # Clear previous logs if any for this test
    caplog.clear()
    with caplog.at_level(logging.WARNING):

        @register_activation_fn("test_act")
        def dummy_activation_overwrite(model, preact, layer_idx):
            return preact * 3

    assert ACTIVATION_REGISTRY["test_act"] == dummy_activation_overwrite

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "Activation function 'test_act' is already registered. Overwriting." in record.message


def test_get_activation_fn_success():
    @register_activation_fn("my_retrieved_act")
    def another_dummy(model, preact, layer_idx):
        return preact

    retrieved_fn = get_activation_fn("my_retrieved_act")
    assert retrieved_fn == another_dummy


def test_get_activation_fn_failure():
    with pytest.raises(ValueError, match="Activation function 'non_existent_act' not found in registry."):
        get_activation_fn("non_existent_act")


# --- Test Individual Registered Functions ---


@pytest.fixture
def mock_model() -> MagicMock:
    mock = MagicMock()
    mock.config = CLTConfig(d_model=16, num_features=32, num_layers=2, activation_fn="relu")  # Basic config
    mock.rank = 0
    mock.device = torch.device("cpu")
    mock.dtype = torch.float32
    return mock


def test_relu_activation_registered(mock_model):
    preact = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    expected_output = F.relu(preact)

    fn = get_activation_fn("relu")
    output = fn(mock_model, preact, 0)
    assert torch.equal(output, expected_output)


def test_jumprelu_activation_registered(mock_model):
    preact = torch.tensor([-1.0, 1.0, 2.0])
    layer_idx = 0
    mock_model.jumprelu = MagicMock(return_value=torch.tensor([0.0, 1.0, 2.0]))  # Simulate model's jumprelu

    fn = get_activation_fn("jumprelu")
    output = fn(mock_model, preact, layer_idx)

    mock_model.jumprelu.assert_called_once_with(preact, layer_idx)
    assert torch.equal(output, mock_model.jumprelu.return_value)


@patch("clt.models.activations.BatchTopK")
def test_batchtopk_per_layer_activation_registered(MockBatchTopK, mock_model, caplog):
    preact = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    layer_idx = 0
    mock_model.config.activation_fn = "batchtopk"
    mock_model.config.batchtopk_k = 2
    mock_model.config.batchtopk_straight_through = True

    # Mock the .apply method of the BatchTopK class
    mock_batchtopk_apply_return = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    MockBatchTopK.apply = MagicMock(return_value=mock_batchtopk_apply_return)

    fn = get_activation_fn("batchtopk")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        output = fn(mock_model, preact, layer_idx)

    MockBatchTopK.apply.assert_called_once_with(
        preact, float(mock_model.config.batchtopk_k), mock_model.config.batchtopk_straight_through, preact
    )
    assert torch.equal(output, mock_batchtopk_apply_return)
    assert any("This applies TopK per-layer, not globally." in record.message for record in caplog.records)


@patch("clt.models.activations.BatchTopK")
def test_batchtopk_per_layer_activation_k_none(MockBatchTopK, mock_model, caplog):
    preact = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    layer_idx = 0
    mock_model.config.activation_fn = "batchtopk"
    mock_model.config.batchtopk_k = None  # Test k=None case
    mock_model.config.batchtopk_straight_through = False

    mock_batchtopk_apply_return = preact.clone()  # Should return all if k is effectively num_features
    MockBatchTopK.apply = MagicMock(return_value=mock_batchtopk_apply_return)

    fn = get_activation_fn("batchtopk")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        output = fn(mock_model, preact, layer_idx)

    assert any("batchtopk_k not set in config" in record.message for record in caplog.records)
    assert any("This applies TopK per-layer, not globally." in record.message for record in caplog.records)
    # k should default to preact.size(1)
    MockBatchTopK.apply.assert_called_once_with(
        preact, float(preact.size(1)), mock_model.config.batchtopk_straight_through, preact
    )
    assert torch.equal(output, mock_batchtopk_apply_return)


@patch("clt.models.activations.TokenTopK")
def test_topk_per_layer_activation_registered(MockTokenTopK, mock_model, caplog):
    preact = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    layer_idx = 0
    mock_model.config.activation_fn = "topk"
    mock_model.config.topk_k = 0.5  # Example fraction
    mock_model.config.topk_straight_through = True  # Example

    mock_tokentopk_apply_return = torch.tensor([[0.0, 0.0, 3.0, 4.0]])  # Dummy return
    MockTokenTopK.apply = MagicMock(return_value=mock_tokentopk_apply_return)

    fn = get_activation_fn("topk")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        output = fn(mock_model, preact, layer_idx)

    MockTokenTopK.apply.assert_called_once_with(
        preact, float(mock_model.config.topk_k), mock_model.config.topk_straight_through, preact
    )
    assert torch.equal(output, mock_tokentopk_apply_return)
    assert any("This applies TopK per-layer, not globally." in record.message for record in caplog.records)


@patch("clt.models.activations.TokenTopK")
def test_topk_per_layer_activation_k_none(MockTokenTopK, mock_model, caplog):
    preact = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    layer_idx = 0
    mock_model.config.activation_fn = "topk"
    # Simulate topk_k not being in config
    if hasattr(mock_model.config, "topk_k"):
        delattr(mock_model.config, "topk_k")
    mock_model.config.topk_straight_through = False

    mock_tokentopk_apply_return = preact.clone()
    MockTokenTopK.apply = MagicMock(return_value=mock_tokentopk_apply_return)

    fn = get_activation_fn("topk")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        output = fn(mock_model, preact, layer_idx)

    assert any("topk_k not set in config" in record.message for record in caplog.records)
    assert any("This applies TopK per-layer, not globally." in record.message for record in caplog.records)
    MockTokenTopK.apply.assert_called_once_with(
        preact, float(preact.size(1)), mock_model.config.topk_straight_through, preact
    )
    assert torch.equal(output, mock_tokentopk_apply_return)
