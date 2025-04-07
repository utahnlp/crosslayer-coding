import pytest
import torch
import os

# import tempfile  # Unused
# import shutil    # Unused
# import numpy as np # Unused

# from clt.config import CLTConfig  # Now unused
from clt.models.clt import CrossLayerTranscoder
from clt.training.data import ActivationStore


@pytest.fixture
def fixture_path():
    """Path to the fixtures directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "data")


@pytest.fixture
def pretrained_model_path(fixture_path):
    """Path to the pretrained model fixture."""
    return os.path.join(fixture_path, "pretrained_models", "pretrained_clt.pt")


@pytest.fixture
def dummy_activations_path(fixture_path):
    """Path to the dummy activations fixtures."""
    return os.path.join(fixture_path, "dummy_activations")


@pytest.mark.integration
def test_load_pretrained_model(pretrained_model_path):
    """Test loading a pretrained model from a file."""
    # Skip if the fixture doesn't exist yet
    if not os.path.exists(pretrained_model_path):
        pytest.skip(f"Pretrained model fixture not found at {pretrained_model_path}")

    # Load the model
    model = CrossLayerTranscoder.load(pretrained_model_path)

    # Check model attributes
    assert model.config.num_layers == 2
    assert model.config.num_features == 8
    assert model.config.d_model == 16
    assert model.config.activation_fn == "jumprelu"
    assert model.config.jumprelu_threshold == 0.03

    # Check model structure
    assert len(model.encoders) == 2
    assert len(model.decoders) == 3  # 0->0, 0->1, 1->1
    assert model.encoders[0].weight.shape == (8, 16)
    assert model.decoders["0->0"].weight.shape == (16, 8)

    # Verify parameters are loaded
    assert torch.any(model.encoders[0].weight != 0)  # Not all zeros
    assert torch.any(model.decoders["0->1"].weight != 0)
    assert torch.allclose(model.threshold, torch.ones(8) * 0.03)


@pytest.mark.integration
def test_pretrained_model_inference(pretrained_model_path, dummy_activations_path):
    """Test running inference with a pretrained model on dummy activations."""
    # Skip if the fixtures don't exist yet
    if not os.path.exists(pretrained_model_path):
        pytest.skip(f"Pretrained model fixture not found at {pretrained_model_path}")
    if not os.path.exists(dummy_activations_path):
        pytest.skip(f"Dummy activations not found at {dummy_activations_path}")

    # Load the model
    model = CrossLayerTranscoder.load(pretrained_model_path)

    # Load some dummy activations
    standard_path = os.path.join(dummy_activations_path, "standard")
    inputs_path = os.path.join(standard_path, "mlp_inputs.pt")

    if not os.path.exists(inputs_path):
        pytest.skip(f"Input activations not found at {inputs_path}")

    # Load inputs
    mlp_inputs = torch.load(inputs_path)

    # Take a small batch for testing
    batch_size = 4
    test_inputs = {
        layer: tensor[:batch_size].unsqueeze(1)  # Add sequence dimension
        for layer, tensor in mlp_inputs.items()
    }

    # Run the model
    with torch.no_grad():
        outputs = model(test_inputs)

    # Check outputs
    assert isinstance(outputs, dict)
    assert len(outputs) == len(test_inputs)

    for layer in test_inputs:
        assert layer in outputs
        # Check shape: [batch_size, seq_len, d_model]
        assert outputs[layer].shape == (batch_size, 1, model.config.d_model)


@pytest.mark.integration
def test_pretrained_model_with_activation_store(
    pretrained_model_path, dummy_activations_path
):
    """Test using a pretrained model with ActivationStore."""
    # Skip if the fixtures don't exist yet
    if not os.path.exists(pretrained_model_path):
        pytest.skip(f"Pretrained model fixture not found at {pretrained_model_path}")

    standard_path = os.path.join(dummy_activations_path, "standard")
    inputs_path = os.path.join(standard_path, "mlp_inputs.pt")
    outputs_path = os.path.join(standard_path, "mlp_outputs.pt")

    if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
        pytest.skip("Activation fixtures not found")

    # Load the model
    model = CrossLayerTranscoder.load(pretrained_model_path)

    # Load inputs and outputs
    mlp_inputs = torch.load(inputs_path)
    mlp_outputs = torch.load(outputs_path)

    # Create an activation store
    store = ActivationStore(
        mlp_inputs=mlp_inputs, mlp_outputs=mlp_outputs, batch_size=8, normalize=True
    )

    # Get a batch from the store
    batch_inputs, batch_targets = store.get_batch()

    # Required shape for model input: [batch_size, seq_len, d_model]
    # Current shape from ActivationStore: [batch_size, d_model]
    model_inputs = {
        layer: tensor.unsqueeze(1)  # Add sequence dimension
        for layer, tensor in batch_inputs.items()
    }

    # Run the model
    with torch.no_grad():
        outputs = model(model_inputs)

    # Check outputs
    assert isinstance(outputs, dict)
    for layer in batch_inputs:
        assert layer in outputs
        assert outputs[layer].shape[0] == batch_inputs[layer].shape[0]  # Batch size
        assert outputs[layer].shape[2] == model.config.d_model  # d_model

    # Flatten outputs back to activation store format for comparison
    flattened_outputs = {
        layer: tensor.squeeze(1)  # Remove sequence dimension
        for layer, tensor in outputs.items()
    }

    # Denormalize both for fair comparison
    denorm_targets = store.denormalize_outputs(batch_targets)
    denorm_outputs = store.denormalize_outputs(flattened_outputs)

    # Calculate MSE difference
    mse_diff = 0.0
    for layer in denorm_outputs:
        layer_mse = torch.mean(
            (denorm_outputs[layer] - denorm_targets[layer]) ** 2
        ).item()
        mse_diff += layer_mse

    # Not checking for a specific error value, just that computation works
    assert isinstance(mse_diff, float)
