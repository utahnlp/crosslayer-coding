import pytest
import torch
import numpy as np
from clt.training.data import ActivationStore
from typing import Dict, Tuple, Iterator, Optional


@pytest.fixture
def dummy_activation_generator():
    """Provides a dummy activation generator for testing."""

    class ReusableGenerator:
        def __init__(self):
            self.data_generated = False
            self.cache = []
            self.index = 0
            self.num_batches = 10  # More batches for multiple iterations
            self.batch_size = 4
            self.seq_len = 8
            self.dim = 16

        def __iter__(self):
            return self

        def __next__(self):
            # If we have cached data and haven't reached the end, use it
            if self.cache and self.index < len(self.cache):
                result = self.cache[self.index]
                self.index += 1
                return result

            # If we've reached the end of our cache, reset index and raise StopIteration
            if self.cache and self.index >= len(self.cache):
                self.index = 0  # Reset index for next iteration
                raise StopIteration

            # If we haven't generated data yet, do so now
            if not self.data_generated:
                for _ in range(self.num_batches):
                    mlp_inputs = {
                        0: torch.randn(self.batch_size, self.seq_len, self.dim),
                        1: torch.randn(self.batch_size, self.seq_len, self.dim),
                    }

                    mlp_outputs = {
                        0: torch.randn(self.batch_size, self.seq_len, self.dim),
                        1: torch.randn(self.batch_size, self.seq_len, self.dim),
                    }

                    token_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))

                    self.cache.append((mlp_inputs, mlp_outputs, token_ids))

                self.data_generated = True

                # Return the first item and increment index
                if self.cache:
                    result = self.cache[0]
                    self.index = 1
                    return result

            # If we got here, we're out of data
            raise StopIteration

    return ReusableGenerator()


@pytest.fixture
def dummy_activations() -> Dict[int, torch.Tensor]:
    """Provides dummy activation data for testing (for backwards compatibility tests)."""
    return {
        0: torch.randn(100, 10),  # 100 tokens, 10 dimensions for layer 0
        1: torch.randn(100, 20),  # 100 tokens, 20 dimensions for layer 1
    }


@pytest.fixture
def dummy_nnsight_activations() -> Dict[str, list[torch.Tensor]]:
    """Provides dummy nnsight activation data for testing."""
    return {
        "model.layers.0.mlp_in_0": [torch.randn(50, 10), torch.randn(50, 10)],
        "model.layers.1.mlp_in_1": [torch.randn(50, 20), torch.randn(50, 20)],
        "model.layers.0.mlp_out_0": [torch.randn(50, 10), torch.randn(50, 10)],
        "model.layers.1.mlp_out_1": [torch.randn(50, 20), torch.randn(50, 20)],
    }


def test_streaming_activation_store_init(dummy_activation_generator):
    """Test initialization of streaming ActivationStore."""
    # Initialize the streaming activation store
    store = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="none",
        exclude_special_tokens=False,
    )

    # Check initialization
    assert store.n_batches_in_buffer == 2
    assert store.train_batch_size_tokens == 64
    assert store.normalization_method == "none"
    assert store.exclude_special_tokens is False
    assert store._storage_buffer is None
    assert store._dataloader is None
    assert store._buffer_size_tokens == 0
    assert store.num_tokens_processed == 0
    assert store.device.type in ["cpu", "cuda"]
    assert not store.input_means  # Should be empty dict initially


def test_refill_buffer(dummy_activation_generator):
    """Test the _refill_buffer method."""
    # Initialize the store
    store = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="none",
    )

    # Call refill buffer
    samples, token_ids = store._refill_buffer()

    # Check that samples is a tensor of the expected form
    assert isinstance(samples, torch.Tensor)
    assert len(samples.shape) == 2  # [num_tokens, dim_input + dim_output]

    # The dimension should be input_dim + output_dim for each layer
    # Layer 0: 16 + 16 = 32
    assert samples.shape[1] == 32  # Now we know it's exactly 32

    # Since we're not filtering special tokens, token_ids should be None or tensor
    if token_ids is not None:
        assert isinstance(token_ids, torch.Tensor)
        assert token_ids.shape[0] == samples.shape[0]


def test_create_dataloader(dummy_activation_generator):
    """Test the _create_dataloader method."""
    # Initialize the store
    store = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="none",
    )

    # Call create dataloader
    store._create_dataloader()

    # Check that the dataloader and storage buffer are created
    assert store._dataloader is not None
    assert store._storage_buffer is not None

    # Get a batch and check its shape
    batch = next(store._dataloader)
    assert isinstance(batch, torch.Tensor)
    assert len(batch.shape) == 2  # [batch_size, dim_in + dim_out]


def test_get_batch_streaming(dummy_activation_generator):
    """Test the get_batch method with streaming store."""
    # Initialize the store
    store = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="none",
    )

    # Get a batch
    input_batch, output_batch = store.get_batch()

    # Check return types
    assert isinstance(input_batch, dict)
    assert isinstance(output_batch, dict)

    # Check that we have entries for each layer
    assert 0 in input_batch
    assert 1 in input_batch
    assert 0 in output_batch
    assert 1 in output_batch

    # Check tensor shapes
    assert len(input_batch[0].shape) == 2  # [batch_size, dim]
    assert len(output_batch[0].shape) == 2  # [batch_size, dim]

    # Get multiple batches to ensure we can iterate
    for _ in range(3):
        input_batch, output_batch = store.get_batch()
        assert isinstance(input_batch, dict)
        assert isinstance(output_batch, dict)


def test_normalization_estimation(dummy_activation_generator):
    """Test the normalization estimation functionality."""
    # Initialize the store with estimated normalization
    store = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=2,
    )

    # Force estimation
    store._estimate_normalization_stats()

    # Check that statistics were computed
    assert 0 in store.input_means
    assert 0 in store.input_stds
    assert 0 in store.output_means
    assert 0 in store.output_stds
    assert 1 in store.input_means

    # Check shapes - now using dim 16 for all layers
    assert store.input_means[0].shape[1] == 16  # [1, dim]
    assert store.input_means[1].shape[1] == 16  # [1, dim]


def test_state_dict_and_load(dummy_activation_generator):
    """Test the state_dict and load_state_dict methods."""
    # Initialize and use the store
    store1 = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=2,
    )

    # Estimate stats and get a batch to populate attributes
    store1._estimate_normalization_stats()
    store1.get_batch()

    # Save state
    state = store1.state_dict()

    # Check state contents
    assert "input_means" in state
    assert "input_stds" in state
    assert "output_means" in state
    assert "output_stds" in state
    assert "layer_indices" in state
    assert "num_tokens_processed" in state
    assert "_buffer_size_tokens" in state

    # Create a new store and load the state
    store2 = ActivationStore(
        activation_generator_or_inputs=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
    )

    store2.load_state_dict(state)

    # Check that the state was loaded
    assert store2.input_means == store1.input_means
    assert store2.input_stds == store1.input_stds
    assert store2.layer_indices == store1.layer_indices
    assert store2.num_tokens_processed == store1.num_tokens_processed
    assert store2._buffer_size_tokens == store1._buffer_size_tokens


# Keep existing tests for backwards compatibility testing


def test_activation_store_init(dummy_activations):
    """Test ActivationStore initialization (legacy test)."""
    mlp_inputs = dummy_activations
    mlp_outputs = {
        k: v + 1 for k, v in dummy_activations.items()
    }  # Simulate different outputs
    batch_size = 32

    # Test without normalization
    store_no_norm = ActivationStore(
        mlp_inputs.copy(), mlp_outputs.copy(), batch_size, normalize=False
    )
    assert store_no_norm.num_layers == 2
    assert store_no_norm.num_tokens == 100
    assert store_no_norm.batch_size == batch_size
    assert not store_no_norm.input_means  # Should be empty if normalize=False
    assert torch.equal(store_no_norm.mlp_inputs[0], mlp_inputs[0])

    # Test with normalization
    store_norm = ActivationStore(
        mlp_inputs.copy(), mlp_outputs.copy(), batch_size, normalize=True
    )
    assert store_norm.num_layers == 2
    assert store_norm.num_tokens == 100
    assert store_norm.batch_size == batch_size
    assert 0 in store_norm.input_means
    assert 1 in store_norm.input_stds
    assert 0 in store_norm.output_means
    assert 1 in store_norm.output_stds
    # Check if inputs are actually normalized (mean approx 0, std approx 1)
    assert torch.allclose(
        store_norm.mlp_inputs[0].mean(dim=0), torch.zeros(10), atol=1e-6
    )
    assert torch.allclose(
        store_norm.mlp_inputs[0].std(dim=0), torch.ones(10), atol=1e-6
    )


def test_activation_store_normalization(dummy_activations):
    """Test normalization logic (legacy test)."""
    mlp_inputs = dummy_activations
    mlp_outputs = {k: v + 1 for k, v in dummy_activations.items()}
    store = ActivationStore(mlp_inputs.copy(), mlp_outputs.copy(), normalize=True)

    for layer_idx in store.layer_indices:
        # Check inputs
        assert torch.allclose(
            store.mlp_inputs[layer_idx].mean(dim=0),
            torch.zeros_like(store.input_means[layer_idx][0]),
            atol=1e-6,
        )
        assert torch.allclose(
            store.mlp_inputs[layer_idx].std(dim=0),
            torch.ones_like(store.input_stds[layer_idx][0]),
            atol=1e-6,
        )
        # Check outputs
        assert torch.allclose(
            store.mlp_outputs[layer_idx].mean(dim=0),
            torch.zeros_like(store.output_means[layer_idx][0]),
            atol=1e-6,
        )
        assert torch.allclose(
            store.mlp_outputs[layer_idx].std(dim=0),
            torch.ones_like(store.output_stds[layer_idx][0]),
            atol=1e-6,
        )


def test_activation_store_batching(dummy_activations):
    """Test batch retrieval and shuffling (legacy test)."""
    mlp_inputs = {k: v.clone() for k, v in dummy_activations.items()}
    mlp_outputs = {k: v.clone() + 1 for k, v in dummy_activations.items()}
    batch_size = 16
    num_tokens = 100
    store = ActivationStore(mlp_inputs, mlp_outputs, batch_size, normalize=False)

    num_batches = (num_tokens + batch_size - 1) // batch_size
    initial_token_indices = store.token_indices.copy()
    seen_indices = set()
    indices_yielded_order = []

    # --- Test one full pass with immediate verification ---
    print("\nTesting first pass...")  # Debug print
    for i in range(num_batches):
        # 1. Determine expected indices based on current state BEFORE get_batch
        start_idx = store.batch_pointer
        end_idx = min(start_idx + batch_size, num_tokens)
        expected_indices_for_batch = store.token_indices[start_idx:end_idx]
        indices_yielded_order.extend(expected_indices_for_batch.tolist())
        seen_indices.update(expected_indices_for_batch.tolist())
        print(
            f" Batch {i+1}: Pointer={start_idx}, Expected Indices={expected_indices_for_batch[:5]}..."
        )  # Debug print

        # 2. Get expected data slice based on these indices
        expected_input_batch = {}
        expected_output_batch = {}
        for layer_idx in store.layer_indices:
            expected_input_batch[layer_idx] = store.mlp_inputs[layer_idx][
                expected_indices_for_batch
            ]
            expected_output_batch[layer_idx] = store.mlp_outputs[layer_idx][
                expected_indices_for_batch
            ]

        # 3. Get the actual batch
        actual_input_batch, actual_output_batch = store.get_batch()

        # 4. Compare actual batch to expected slice immediately
        assert actual_input_batch[0].shape[0] == len(expected_indices_for_batch)
        for layer_idx in store.layer_indices:
            assert torch.allclose(
                actual_input_batch[layer_idx], expected_input_batch[layer_idx]
            ), f"Input mismatch layer {layer_idx}, batch {i+1}"
            assert torch.allclose(
                actual_output_batch[layer_idx], expected_output_batch[layer_idx]
            ), f"Output mismatch layer {layer_idx}, batch {i+1}"

        # 5. Check last batch size calculation (redundant given shape check, but safe)
        if i == num_batches - 1:
            expected_last_batch_size = num_tokens % batch_size
            if expected_last_batch_size == 0:
                expected_last_batch_size = batch_size
            assert actual_input_batch[0].shape[0] == expected_last_batch_size

    # --- Verification after one full pass ---
    assert len(seen_indices) == num_tokens
    assert store.batch_pointer == 0  # Pointer should have reset
    # Check the full sequence of yielded indices matches the initial shuffle
    assert np.array_equal(np.array(indices_yielded_order), initial_token_indices)

    # --- Test shuffling (second pass) ---
    store.shuffle_tokens()
    second_initial_indices = store.token_indices.copy()
    assert not np.array_equal(
        initial_token_indices, second_initial_indices
    ), "Shuffle didn't change order"

    seen_indices_pass2 = set()
    indices_yielded_pass2 = []
    for _ in range(num_batches):
        start_idx = store.batch_pointer
        end_idx = min(start_idx + batch_size, num_tokens)
        current_batch_indices = store.token_indices[start_idx:end_idx]
        indices_yielded_pass2.extend(current_batch_indices.tolist())
        seen_indices_pass2.update(current_batch_indices.tolist())
        store.get_batch()

    assert len(seen_indices_pass2) == num_tokens
    assert store.batch_pointer == 0
    assert np.array_equal(np.array(indices_yielded_pass2), second_initial_indices)
    # Verify the order is different from the first pass
    assert not np.array_equal(
        np.array(indices_yielded_order), np.array(indices_yielded_pass2)
    )


def test_activation_store_denormalization(dummy_activations):
    """Test output denormalization (legacy test)."""
    mlp_inputs_orig = dummy_activations
    mlp_outputs_orig = {
        k: v + torch.rand_like(v) * 5 for k, v in dummy_activations.items()
    }  # More varied outputs

    # Store with normalization
    store = ActivationStore(
        mlp_inputs_orig.copy(), mlp_outputs_orig.copy(), normalize=True
    )

    # Get a batch of normalized outputs
    _, output_batch_norm = store.get_batch()

    # Denormalize the batch
    output_batch_denorm = store.denormalize_outputs(output_batch_norm)

    # Get the original outputs corresponding to this batch
    # Note: batch indices are shuffled, need to retrieve them
    start_idx = 0  # Batch pointer was reset in get_batch
    batch_indices = store.token_indices[start_idx : store.batch_size]

    for layer_idx in store.layer_indices:
        original_outputs_batch = mlp_outputs_orig[layer_idx][batch_indices]
        assert torch.allclose(
            output_batch_denorm[layer_idx], original_outputs_batch, atol=1e-6
        )

    # Test denormalization when normalization was off
    store_no_norm = ActivationStore(
        mlp_inputs_orig.copy(), mlp_outputs_orig.copy(), normalize=False
    )
    _, output_batch_no_norm = store_no_norm.get_batch()
    output_batch_denorm_no_norm = store_no_norm.denormalize_outputs(
        output_batch_no_norm
    )
    for layer_idx in store_no_norm.layer_indices:
        assert torch.equal(
            output_batch_denorm_no_norm[layer_idx], output_batch_no_norm[layer_idx]
        )


def test_activation_store_from_nnsight(dummy_nnsight_activations):
    """Test creating ActivationStore from nnsight format (legacy test)."""
    batch_size = 32
    store = ActivationStore.from_nnsight_activations(
        dummy_nnsight_activations, batch_size, normalize=True
    )

    assert store.num_layers == 2
    assert store.num_tokens == 100  # 50 + 50 concatenated
    assert store.batch_size == batch_size
    assert 0 in store.layer_indices
    assert 1 in store.layer_indices

    # Check shapes
    assert store.mlp_inputs[0].shape == (100, 10)
    assert store.mlp_inputs[1].shape == (100, 20)
    assert store.mlp_outputs[0].shape == (100, 10)
    assert store.mlp_outputs[1].shape == (100, 20)

    # Check normalization happened
    assert 0 in store.input_means
    assert torch.allclose(store.mlp_inputs[0].mean(dim=0), torch.zeros(10), atol=1e-6)

    # Test without normalization
    store_no_norm = ActivationStore.from_nnsight_activations(
        dummy_nnsight_activations, batch_size, normalize=False
    )
    assert not store_no_norm.input_means
    # Check concatenation happened correctly
    expected_input_0 = torch.cat(
        dummy_nnsight_activations["model.layers.0.mlp_in_0"], dim=0
    )
    assert torch.equal(store_no_norm.mlp_inputs[0], expected_input_0)


def test_cache_path_integration():
    """Test that activation caching works correctly with CLTTrainer."""
    # This is a higher-level integration test checking that the caching parameters
    # are properly passed through the system.

    import unittest.mock as mock
    from clt.training.trainer import CLTTrainer
    from clt.config import CLTConfig, TrainingConfig

    # Create minimal configs for testing
    clt_config = CLTConfig(num_features=16, num_layers=2, d_model=32)

    training_config = TrainingConfig(
        learning_rate=0.001,
        training_steps=10,
        cache_path="/fake/cache/path",  # This is what we're testing
    )

    # Mock the entire activation store creation process
    with mock.patch(
        "clt.training.trainer.ActivationExtractor"
    ) as MockExtractor, mock.patch("clt.training.trainer.ActivationStore") as MockStore:

        # Set up the mocked extractor
        mock_extractor_instance = MockExtractor.return_value
        mock_stream = mock.MagicMock()
        mock_extractor_instance.stream_activations = mock_stream

        # Set up the mocked store
        mock_store_instance = MockStore.return_value

        # Create trainer (which will call our mocks)
        CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir="test_cache_dir",
        )

        # Verify stream_activations was called with the cache_path
        mock_stream.assert_called_once()
        cache_path_arg = mock_stream.call_args.kwargs.get("cache_path")
        assert cache_path_arg == "/fake/cache/path"

        # Verify ActivationStore was created correctly
        MockStore.assert_called_once()
        store_args = MockStore.call_args.kwargs
        assert "activation_generator_or_inputs" in store_args

        # Cleanup the temp directory
        import shutil
        import os

        if os.path.exists("test_cache_dir"):
            shutil.rmtree("test_cache_dir")
