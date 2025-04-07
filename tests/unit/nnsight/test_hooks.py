import pytest
from unittest.mock import MagicMock

from clt.nnsight.hooks import get_mlp_paths


@pytest.mark.parametrize(
    "model_name", ["llama-7b", "meta-llama/Meta-Llama-3.1-8B", "LLaMA"]
)
def test_get_mlp_paths_llama_model(model_name):
    """Test get_mlp_paths returns correct paths for LLaMA-style models."""
    paths = get_mlp_paths(model_name)

    # Check return structure
    assert isinstance(paths, dict)
    assert "input_path" in paths
    assert "output_path" in paths
    assert callable(paths["input_path"])
    assert callable(paths["output_path"])

    # Mock a LLaMA-style model
    mock_model = MagicMock()
    mock_model.model.layers = MagicMock()
    layer_idx = 3

    # Set up nested mocks
    mock_layer = MagicMock()
    mock_model.model.layers.__getitem__.return_value = mock_layer

    # Test input path function
    input_path_fn = paths["input_path"]
    input_path_fn(mock_model, layer_idx)

    # Check that the correct path was accessed
    mock_model.model.layers.__getitem__.assert_called_with(layer_idx)

    # Test output path function
    output_path_fn = paths["output_path"]
    output_path_fn(mock_model, layer_idx)

    # Check that the correct path was accessed
    mock_model.model.layers.__getitem__.assert_called_with(layer_idx)


@pytest.mark.parametrize(
    "model_name", ["gpt2", "gpt2-medium", "gpt2-xl", "other-model"]
)
def test_get_mlp_paths_gpt2_model(model_name):
    """Test get_mlp_paths returns correct paths for GPT-2-style models."""
    paths = get_mlp_paths(model_name)

    # Check return structure
    assert isinstance(paths, dict)
    assert "input_path" in paths
    assert "output_path" in paths
    assert callable(paths["input_path"])
    assert callable(paths["output_path"])

    # Mock a GPT-2-style model
    mock_model = MagicMock()
    mock_model.transformer.h = MagicMock()
    layer_idx = 5

    # Set up nested mocks
    mock_layer = MagicMock()
    mock_model.transformer.h.__getitem__.return_value = mock_layer

    # Test input path function
    input_path_fn = paths["input_path"]
    input_path_fn(mock_model, layer_idx)

    # Check that the correct path was accessed
    mock_model.transformer.h.__getitem__.assert_called_with(layer_idx)

    # Test output path function
    output_path_fn = paths["output_path"]
    output_path_fn(mock_model, layer_idx)

    # Check that the correct path was accessed
    mock_model.transformer.h.__getitem__.assert_called_with(layer_idx)
