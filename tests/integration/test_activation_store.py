import pytest
import torch
import os
import tempfile
import shutil
import numpy as np

from clt.training.data import ActivationStore


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp(prefix="clt_integration_test_data_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_nnsight_activations():
    """Create dummy NNsight-format activations for testing."""
    return {
        "model.layers.0.mlp_in_0": [torch.randn(50, 32), torch.randn(50, 32)],
        "model.layers.1.mlp_in_1": [torch.randn(50, 32), torch.randn(50, 32)],
        "model.layers.0.mlp_out_0": [torch.randn(50, 32), torch.randn(50, 32)],
        "model.layers.1.mlp_out_1": [torch.randn(50, 32), torch.randn(50, 32)],
    }


@pytest.fixture
def saved_activation_files(temp_data_dir, dummy_nnsight_activations):
    """Save dummy activations to disk and return the path."""
    # Save the activation tensor lists as separate files
    file_paths = {}

    for key, tensor_list in dummy_nnsight_activations.items():
        # Combine the tensors into one tensor for simplicity
        combined = torch.cat(tensor_list, dim=0)

        # Create a path for this activation
        file_name = f"{key.replace('.', '_')}.pt"
        file_path = os.path.join(temp_data_dir, file_name)

        # Save the tensor
        torch.save(combined, file_path)
        file_paths[key] = file_path

    # Save a metadata file with the structure
    metadata = {"layer_indices": [0, 1], "d_model": 32, "num_tokens": 100}  # 50 + 50
    metadata_path = os.path.join(temp_data_dir, "metadata.pt")
    torch.save(metadata, metadata_path)

    return {"files": file_paths, "metadata": metadata_path, "dir": temp_data_dir}


@pytest.mark.integration
def test_activation_store_from_nnsight(dummy_nnsight_activations):
    """Test creating an ActivationStore from NNsight-formatted activations."""
    batch_size = 8

    # Create store using the class method
    store = ActivationStore.from_nnsight_activations(
        dummy_nnsight_activations, batch_size=batch_size, normalize=True
    )

    # Verify the store was created correctly
    assert store.num_layers == 2
    assert store.num_tokens == 100  # 50 + 50 from two batches
    assert store.batch_size == batch_size

    # Check layer indices
    assert set(store.layer_indices) == {0, 1}

    # Check shapes of stored activations
    assert store.mlp_inputs[0].shape == (100, 32)
    assert store.mlp_outputs[1].shape == (100, 32)

    # Check normalization happened
    assert 0 in store.input_means
    assert 1 in store.input_stds
    assert torch.allclose(store.mlp_inputs[0].mean(dim=0), torch.zeros(32), atol=1e-6)
    assert torch.allclose(store.mlp_inputs[0].std(dim=0), torch.ones(32), atol=1e-6)

    # Test retrieving batches
    inputs, outputs = store.get_batch()

    # Check batch shapes
    assert inputs[0].shape == (batch_size, 32)
    assert outputs[1].shape == (batch_size, 32)

    # Check we can get all batches
    seen_tokens = 0
    all_batches = []
    original_token_indices = store.token_indices.copy()

    # Collect all batches and track how many tokens we've seen
    while seen_tokens < store.num_tokens:
        batch_inputs, _ = store.get_batch()
        batch_size_actual = batch_inputs[0].shape[0]
        all_batches.append(batch_inputs)
        seen_tokens += batch_size_actual

    # Check we saw all tokens
    assert seen_tokens == store.num_tokens

    # Check that token_indices was shuffled when we exhausted all tokens
    assert not np.array_equal(original_token_indices, store.token_indices)


@pytest.mark.integration
def test_denormalize_outputs(dummy_nnsight_activations):
    """Test that denormalization correctly restores original scale."""
    # Create an unnormalized store first to get the original data
    original_store = ActivationStore.from_nnsight_activations(
        dummy_nnsight_activations, batch_size=16, normalize=False
    )
    original_outputs_all = original_store.mlp_outputs

    # Create the store with normalization for the test
    store = ActivationStore.from_nnsight_activations(
        dummy_nnsight_activations, batch_size=16, normalize=True
    )

    # Get a batch of normalized outputs and its indices
    store.shuffle_tokens()  # Shuffle once to get a specific order
    batch_indices = store.token_indices[: store.batch_size]
    _, normalized_outputs = store.get_batch()  # This updates the pointer

    # Denormalize the outputs
    denormalized_batch = store.denormalize_outputs(normalized_outputs)

    # Check denormalized shape matches normalized shape
    for layer_idx in normalized_outputs:
        assert (
            denormalized_batch[layer_idx].shape == normalized_outputs[layer_idx].shape
        )

    # Compare the denormalized batch to the original data for the same indices
    for layer_idx in denormalized_batch:
        # Retrieve the original (unnormalized) data for the specific batch indices
        original_data_batch = original_outputs_all[layer_idx][batch_indices]

        # Compare denormalized batch data to original data slice
        assert torch.allclose(
            denormalized_batch[layer_idx], original_data_batch, atol=1e-6
        ), f"Denormalization failed for layer {layer_idx}"


# This test simulates a more realistic scenario where we might save activations from
# a model run, then load them back for training
@pytest.mark.integration
def test_save_load_activation_workflow(saved_activation_files):
    """Test a workflow of loading activations from saved files."""
    # In a real scenario, these files would come from an activation extraction run
    file_dir = saved_activation_files["dir"]

    # Load the activations manually to simulate what might happen in a real workflow
    mlp_inputs = {}
    mlp_outputs = {}

    # Get all .pt files in the directory
    for filename in os.listdir(file_dir):
        if filename.endswith(".pt") and "metadata" not in filename:
            file_path = os.path.join(file_dir, filename)
            tensor = torch.load(file_path)

            # Parse the filename to determine what this tensor represents
            if "mlp_in" in filename:
                # Handle potential double extensions like .mlp_in_0.pt
                base_name = filename.rsplit(".", 1)[0]
                layer_idx = int(base_name.split("_")[-1])
                mlp_inputs[layer_idx] = tensor
            elif "mlp_out" in filename:
                base_name = filename.rsplit(".", 1)[0]
                layer_idx = int(base_name.split("_")[-1])
                mlp_outputs[layer_idx] = tensor

    # Create an ActivationStore from the loaded files
    store = ActivationStore(
        mlp_inputs=mlp_inputs, mlp_outputs=mlp_outputs, batch_size=10, normalize=True
    )

    # Verify the store looks correct
    assert store.num_layers == 2
    assert store.num_tokens == 100
    assert set(store.layer_indices) == {0, 1}

    # Test batching works
    inputs, outputs = store.get_batch()
    assert inputs[0].shape == (10, 32)  # Batch size 10, d_model 32
    assert outputs[1].shape == (10, 32)

    # Shuffle and get another batch
    store.shuffle_tokens()
    inputs2, outputs2 = store.get_batch()

    # These should be different batches (tiny chance they are the same)
    # Just check shapes are as expected
    assert inputs2[0].shape == (10, 32)
    assert outputs2[1].shape == (10, 32)
