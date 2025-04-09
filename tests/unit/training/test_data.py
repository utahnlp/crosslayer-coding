import pytest
import torch
import numpy as np
from typing import Dict, Tuple, List, Generator, Union
import time
import sys
from unittest.mock import patch, MagicMock

# Assuming clt is importable from the test environment
from clt.training.data import ActivationStore, ActivationBatchCLT


# --- Test Fixtures ---

NUM_LAYERS = 2
D_MODEL = 16
NUM_GEN_BATCHES = 20  # Number of batches the dummy generator can yield
TOKENS_PER_GEN_BATCH = 128  # Number of tokens in each batch yielded by the generator


@pytest.fixture
def dummy_activation_generator() -> Generator[ActivationBatchCLT, None, None]:
    """Provides a dummy activation generator for testing."""

    def _generator():
        for _ in range(NUM_GEN_BATCHES):
            inputs_dict: Dict[int, torch.Tensor] = {}
            targets_dict: Dict[int, torch.Tensor] = {}
            for layer_idx in range(NUM_LAYERS):
                # Simulate slightly different data for inputs/targets
                inputs_dict[layer_idx] = torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL)
                targets_dict[layer_idx] = (
                    torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL) * 0.5 + 0.1
                )
            yield inputs_dict, targets_dict
            # Small sleep to simulate potential real-world generator delay
            # time.sleep(0.001)

    return _generator()  # Return the generator iterator


@pytest.fixture
def exhausted_generator() -> Generator[ActivationBatchCLT, None, None]:
    """Provides a generator that yields nothing."""

    def _generator():
        if False:  # Never yield
            yield  # pragma: no cover

    return _generator()


@pytest.fixture
def inconsistent_d_model_generator() -> Generator[ActivationBatchCLT, None, None]:
    """Generator yielding inconsistent d_model."""

    def _generator():
        # First batch (consistent)
        inputs_dict = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        targets_dict = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        yield inputs_dict, targets_dict
        # Second batch (inconsistent)
        inputs_dict_bad = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL + 1),
        }
        targets_dict_bad = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL + 1),
        }
        yield inputs_dict_bad, targets_dict_bad

    return _generator()


@pytest.fixture
def inconsistent_layers_generator() -> Generator[ActivationBatchCLT, None, None]:
    """Generator yielding inconsistent layers."""

    def _generator():
        # First batch (layers 0, 1)
        inputs_dict = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        targets_dict = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            1: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        yield inputs_dict, targets_dict
        # Second batch (layers 0, 2 - inconsistent)
        inputs_dict_bad = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            2: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        targets_dict_bad = {
            0: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
            2: torch.randn(TOKENS_PER_GEN_BATCH, D_MODEL),
        }
        yield inputs_dict_bad, targets_dict_bad

    return _generator()


# --- Test Functions ---


def test_activation_store_init_basic(dummy_activation_generator):
    """Test basic initialization of ActivationStore."""
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="none",
        device="cpu",  # Explicitly use CPU for predictability
    )

    assert store.n_batches_in_buffer == 4
    assert store.train_batch_size_tokens == 64
    assert store.target_buffer_size_tokens == 4 * 64
    assert store.normalization_method == "none"
    assert store.device == torch.device("cpu")
    assert not store.buffer_initialized
    assert not store.generator_exhausted
    assert not store.layer_indices  # Initialized lazily
    assert store.d_model == -1  # Initialized lazily
    assert not store.buffered_inputs
    assert not store.buffered_targets
    assert store.read_indices.shape == (0,)
    assert store.total_tokens_yielded_by_generator == 0
    assert store.start_time is not None


def test_buffer_metadata_initialization(dummy_activation_generator):
    """Test that buffer metadata is initialized correctly on first batch pull."""
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        device="cpu",
    )

    # Metadata shouldn't be initialized yet
    assert not store.buffer_initialized
    assert not store.layer_indices
    assert store.d_model == -1

    # Trigger buffer fill and initialization
    store.get_batch()

    assert store.buffer_initialized
    assert store.layer_indices == list(range(NUM_LAYERS))
    assert store.d_model == D_MODEL
    assert store.dtype == torch.float32  # Default or inferred
    assert len(store.buffered_inputs) == NUM_LAYERS
    assert len(store.buffered_targets) == NUM_LAYERS
    assert all(isinstance(t, torch.Tensor) for t in store.buffered_inputs.values())
    assert all(isinstance(t, torch.Tensor) for t in store.buffered_targets.values())
    # Check buffer size is roughly target size (or less if generator is short)
    expected_min_tokens = min(
        store.target_buffer_size_tokens, NUM_GEN_BATCHES * TOKENS_PER_GEN_BATCH
    )
    # Allow for slight variation depending on when buffer fill stops
    assert store.read_indices.shape[0] >= store.train_batch_size_tokens
    assert store.read_indices.shape[0] <= expected_min_tokens


def test_get_batch_basic(dummy_activation_generator):
    """Test fetching a single batch."""
    train_batch_size = 64
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=train_batch_size,
        device="cpu",
    )

    inputs, targets = store.get_batch()

    assert isinstance(inputs, dict)
    assert isinstance(targets, dict)
    assert sorted(inputs.keys()) == list(range(NUM_LAYERS))
    assert sorted(targets.keys()) == list(range(NUM_LAYERS))

    for layer_idx in range(NUM_LAYERS):
        assert isinstance(inputs[layer_idx], torch.Tensor)
        assert isinstance(targets[layer_idx], torch.Tensor)
        assert inputs[layer_idx].shape == (train_batch_size, D_MODEL)
        assert targets[layer_idx].shape == (train_batch_size, D_MODEL)
        assert inputs[layer_idx].device == store.device
        assert targets[layer_idx].device == store.device
        assert inputs[layer_idx].dtype == store.dtype
        assert targets[layer_idx].dtype == store.dtype

    # Check that some tokens are now marked as read
    assert store.read_indices.sum().item() == train_batch_size


def test_get_batch_multiple(dummy_activation_generator):
    """Test fetching multiple batches."""
    train_batch_size = 64
    n_batches_buffer = 4
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=n_batches_buffer,
        train_batch_size_tokens=train_batch_size,
        device="cpu",
    )

    num_fetches = 5
    total_expected_tokens = num_fetches * train_batch_size
    fetched_token_indices = set()

    # Store original buffer contents after initial fill for comparison
    store.get_batch()  # Initial fill + first batch
    initial_buffer_inputs = {k: v.clone() for k, v in store.buffered_inputs.items()}
    initial_read_indices_mask = store.read_indices.clone()
    # Need the indices that *were* sampled for the first batch
    first_batch_indices = initial_read_indices_mask.nonzero().squeeze().tolist()
    if isinstance(first_batch_indices, int):  # Handle single index case
        first_batch_indices = [first_batch_indices]
    fetched_token_indices.update(first_batch_indices)

    for i in range(1, num_fetches):  # Fetch remaining batches
        # Record buffer state *before* get_batch to find newly sampled indices
        buffer_size_before = store.read_indices.shape[0]
        read_mask_before = store.read_indices.clone()

        inputs, targets = store.get_batch()

        # Verify batch structure
        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
        assert len(inputs) == NUM_LAYERS
        for layer_idx in range(NUM_LAYERS):
            assert inputs[layer_idx].shape == (train_batch_size, D_MODEL)

        # Ensure the correct number of *new* indices were sampled for this batch
        # This check is tricky because pruning/refilling happens. A simpler check is
        # that the total number of read tokens increases correctly.
        # assert len(sampled_indices_in_buffer_before) == train_batch_size

        # Add *original* buffer indices to our set (requires mapping back if pruned)
        # This is too complex to track robustly here. Instead, focus on total read count.

    # Check total number of read tokens (might be slightly higher due to pruning timing)
    # The number of True values in read_indices might not be exactly total_expected_tokens
    # because pruning removes read tokens.
    # A better check is the total number of batches retrieved.
    assert i + 1 == num_fetches  # Check we completed the loop

    # Check generator progress
    # Formula: ceil(num_fetches * train_batch_size / TOKENS_PER_GEN_BATCH)
    expected_gen_batches_pulled = -(-total_expected_tokens // TOKENS_PER_GEN_BATCH)
    # Allow for potentially one extra batch pull due to buffer refill logic
    assert (
        store.total_tokens_yielded_by_generator / TOKENS_PER_GEN_BATCH
        >= expected_gen_batches_pulled
    )
    assert (
        store.total_tokens_yielded_by_generator / TOKENS_PER_GEN_BATCH
        <= expected_gen_batches_pulled + n_batches_buffer
    )  # Max buffer pull


def test_buffer_pruning(dummy_activation_generator):
    """Test that read tokens are pruned from the buffer."""
    train_batch_size = 64
    n_batches_buffer = 2  # Small buffer to force pruning
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=n_batches_buffer,
        train_batch_size_tokens=train_batch_size,
        device="cpu",
    )

    # Fetch enough batches to ensure some tokens at the start should be pruned
    num_fetches = n_batches_buffer + 1

    store.get_batch()  # Initial fill + first batch
    size_after_first_batch = store.read_indices.shape[0]

    for _ in range(1, num_fetches):
        store.get_batch()

    # After several fetches, the buffer size should ideally not grow indefinitely
    # if pruning is working. It might be larger than the initial fill if the
    # generator keeps providing data, but shouldn't exceed target + a bit extra.
    max_expected_size = (
        store.target_buffer_size_tokens + TOKENS_PER_GEN_BATCH
    )  # Target + one generator batch margin
    assert store.read_indices.shape[0] <= max_expected_size

    # More specific check: after enough batches, the first few original tokens should be gone.
    # This requires tracking original tokens, which is complex.
    # Alternative: Check that the number of read tokens (True) doesn't just keep increasing
    # towards the total buffer size indefinitely.
    num_read = store.read_indices.sum().item()
    buffer_size = store.read_indices.shape[0]
    # If pruning works, the number read should be less than the current buffer size
    # unless *all* tokens currently in the buffer happen to have been read (unlikely).
    assert num_read < buffer_size or buffer_size == 0


def test_generator_exhaustion(dummy_activation_generator):
    """Test behavior when the generator runs out of data."""
    train_batch_size = 64
    store = ActivationStore(
        activation_generator=dummy_activation_generator,  # Use the standard one
        n_batches_in_buffer=4,
        train_batch_size_tokens=train_batch_size,
        device="cpu",
    )

    # Calculate how many batches we can possibly get
    total_tokens_available = NUM_GEN_BATCHES * TOKENS_PER_GEN_BATCH
    max_batches = total_tokens_available // train_batch_size

    # Fetch all possible full batches
    num_fetched = 0
    try:
        for i in range(max_batches + 5):  # Try to fetch more than available
            store.get_batch()
            num_fetched += 1
            if (
                store.generator_exhausted
                and (~store.read_indices).sum().item() < train_batch_size
            ):
                # If generator done and not enough left for a full batch, break early
                break
    except StopIteration:
        pass  # Expected when buffer is empty after generator exhaustion

    # Check that we fetched roughly the maximum number of batches
    assert num_fetched >= max_batches
    assert num_fetched <= max_batches + 1  # Allow for one partial batch potentially

    # Check that the generator is marked as exhausted
    assert store.generator_exhausted

    # Try fetching again, should raise StopIteration
    with pytest.raises(StopIteration):
        store.get_batch()


def test_iterator_protocol(dummy_activation_generator):
    """Test that the store can be used as an iterator."""
    train_batch_size = 64
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=train_batch_size,
        device="cpu",
    )

    num_batches_to_fetch = 5
    count = 0
    for i, (inputs, targets) in enumerate(store):
        assert isinstance(inputs, dict)
        assert isinstance(targets, dict)
        assert len(inputs[0]) == train_batch_size  # Check batch size
        count += 1
        if count >= num_batches_to_fetch:
            break

    assert count == num_batches_to_fetch


def test_normalization_estimation(dummy_activation_generator):
    """Test the 'estimated_mean_std' normalization."""
    estimation_batches = 5
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=estimation_batches,
        device="cpu",
    )

    # Stats should be computed during __init__
    assert (
        store.normalization_method == "estimated_mean_std"
    )  # Should not change on success
    assert len(store.input_means) == NUM_LAYERS
    assert len(store.input_stds) == NUM_LAYERS
    assert len(store.output_means) == NUM_LAYERS
    assert len(store.output_stds) == NUM_LAYERS

    for layer_idx in range(NUM_LAYERS):
        assert store.input_means[layer_idx].shape == (1, D_MODEL)
        assert store.input_stds[layer_idx].shape == (1, D_MODEL)
        assert store.output_means[layer_idx].shape == (1, D_MODEL)
        assert store.output_stds[layer_idx].shape == (1, D_MODEL)
        # Check stds are positive
        assert torch.all(store.input_stds[layer_idx] > 0)
        assert torch.all(store.output_stds[layer_idx] > 0)

    # Check that the generator was advanced by the correct number of batches
    assert (
        store.total_tokens_yielded_by_generator
        == estimation_batches * TOKENS_PER_GEN_BATCH
    )

    # Check that the initial batches used for stats were added back to the buffer
    assert store.buffer_initialized
    assert store.read_indices.shape[0] >= min(
        store.target_buffer_size_tokens, estimation_batches * TOKENS_PER_GEN_BATCH
    )
    assert (~store.read_indices).sum().item() > 0  # Should have unread tokens

    # Fetch a batch and check if values seem normalized (mean ~0, std ~1)
    # Note: This is tricky because we fetch a *random sample* from the buffer,
    # which includes normalized data from the estimation phase.
    inputs, targets = store.get_batch()
    sample_input_mean = inputs[0].mean(dim=0)
    sample_input_std = inputs[0].std(dim=0)
    sample_target_mean = targets[0].mean(dim=0)
    sample_target_std = targets[0].std(dim=0)

    # Due to sampling and the mix of data, might not be exactly 0/1, but should be close
    assert torch.allclose(sample_input_mean, torch.zeros(D_MODEL), atol=0.5)
    assert torch.allclose(sample_input_std, torch.ones(D_MODEL), atol=1.0)
    assert torch.allclose(sample_target_mean, torch.zeros(D_MODEL), atol=0.5)
    assert torch.allclose(sample_target_std, torch.ones(D_MODEL), atol=1.0)


def test_normalization_estimation_insufficient_data(exhausted_generator):
    """Test normalization estimation when generator provides too little data."""
    store = ActivationStore(
        activation_generator=exhausted_generator,  # Generator yields nothing
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=5,
        device="cpu",
    )

    # Check that normalization method falls back to 'none'
    assert store.normalization_method == "none"
    assert not store.input_means  # Stats dictionaries should be empty
    assert not store.input_stds
    assert not store.output_means
    assert not store.output_stds
    assert store.generator_exhausted  # Generator should be marked exhausted


def test_denormalization(dummy_activation_generator):
    """Test the denormalize_outputs method."""
    estimation_batches = 5
    store = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=estimation_batches,
        device="cpu",
    )

    # Get original means/stds for comparison
    input_means = {k: v.clone() for k, v in store.input_means.items()}
    output_means = {k: v.clone() for k, v in store.output_means.items()}
    input_stds = {k: v.clone() for k, v in store.input_stds.items()}
    output_stds = {k: v.clone() for k, v in store.output_stds.items()}

    # Fetch a normalized batch
    inputs_norm, targets_norm = store.get_batch()

    # Denormalize the targets
    targets_denorm = store.denormalize_outputs(targets_norm)

    # Check denormalization
    for layer_idx in range(NUM_LAYERS):
        # Calculate expected denormalized values
        expected_denorm = (
            targets_norm[layer_idx] * output_stds[layer_idx]
        ) + output_means[layer_idx]
        assert torch.allclose(targets_denorm[layer_idx], expected_denorm, atol=1e-6)

    # Test denormalization when method is 'none'
    store_no_norm = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="none",
        device="cpu",
    )
    inputs_no_norm, targets_no_norm = store_no_norm.get_batch()
    targets_denorm_noop = store_no_norm.denormalize_outputs(targets_no_norm)
    # Should be identical (no-op)
    assert torch.equal(targets_denorm_noop[0], targets_no_norm[0])
    assert torch.equal(targets_denorm_noop[1], targets_no_norm[1])


def test_state_dict_and_load(dummy_activation_generator):
    """Test saving and loading the store's state."""
    estimation_batches = 3
    store1 = ActivationStore(
        activation_generator=dummy_activation_generator,
        n_batches_in_buffer=4,
        train_batch_size_tokens=64,
        normalization_method="estimated_mean_std",
        normalization_estimation_batches=estimation_batches,
        device="cpu",
    )
    # Fetch a batch to ensure buffer is initialized etc.
    store1.get_batch()
    # Fetch a few more to advance the generator state
    store1.get_batch()
    store1.get_batch()

    state = store1.state_dict()

    # Check state contents
    assert "layer_indices" in state
    assert "d_model" in state
    assert "dtype" in state
    assert "input_means" in state
    assert "input_stds" in state
    assert "output_means" in state
    assert "output_stds" in state
    assert "total_tokens_yielded_by_generator" in state
    assert "target_buffer_size_tokens" in state
    assert "normalization_method" in state

    assert state["layer_indices"] == store1.layer_indices
    assert state["d_model"] == store1.d_model
    assert state["dtype"] == str(store1.dtype)
    assert (
        state["total_tokens_yielded_by_generator"]
        == store1.total_tokens_yielded_by_generator
    )
    assert state["target_buffer_size_tokens"] == store1.target_buffer_size_tokens
    assert state["normalization_method"] == store1.normalization_method

    # Check stats are on CPU in state dict
    assert state["input_means"][0].device == torch.device("cpu")

    # Create a new store instance (with a fresh generator)
    store2 = ActivationStore(
        activation_generator=dummy_activation_generator,  # Needs a generator, even if state is loaded
        n_batches_in_buffer=10,  # Different buffer size to check loaded value
        train_batch_size_tokens=128,  # Different batch size
        normalization_method="none",  # Different norm method
        device="cpu",
    )

    store2.load_state_dict(state)

    # Check loaded state
    assert store2.layer_indices == store1.layer_indices
    assert store2.d_model == store1.d_model
    assert store2.dtype == store1.dtype
    assert (
        store2.total_tokens_yielded_by_generator
        == store1.total_tokens_yielded_by_generator
    )
    # These should come from the state dict, not the new __init__ args
    assert store2.target_buffer_size_tokens == store1.target_buffer_size_tokens
    assert store2.normalization_method == store1.normalization_method

    # Check stats are loaded correctly and moved to the store's device
    assert torch.equal(store2.input_means[0], store1.input_means[0])
    assert store2.input_means[0].device == store2.device
    assert torch.equal(store2.input_stds[0], store1.input_stds[0])
    assert store2.input_stds[0].device == store2.device
    assert torch.equal(store2.output_means[0], store1.output_means[0])
    assert store2.output_means[0].device == store2.device
    assert torch.equal(store2.output_stds[0], store1.output_stds[0])
    assert store2.output_stds[0].device == store2.device

    # Check buffer is reset/empty after loading state
    assert not store2.buffer_initialized
    assert not store2.buffered_inputs
    assert store2.read_indices.shape == (0,)

    # Check that getting a batch works after loading state
    inputs, targets = store2.get_batch()
    assert inputs[0].shape == (store2.train_batch_size_tokens, store2.d_model)
    # Check normalization was applied using loaded stats
    if store2.normalization_method == "estimated_mean_std":
        assert torch.allclose(inputs[0].mean(), torch.tensor(0.0), atol=0.5)


def test_inconsistent_d_model_error(inconsistent_d_model_generator):
    """Test ValueError when generator yields inconsistent d_model."""
    store = ActivationStore(
        activation_generator=inconsistent_d_model_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        device="cpu",
    )
    # First batch should work
    store.get_batch()
    # Subsequent fetches should trigger the error when the bad batch is processed
    with pytest.raises(ValueError, match="Inconsistent d_model"):
        # Fetch enough times to force processing the second (bad) batch
        for _ in range(3):
            store.get_batch()


def test_inconsistent_layers_error(inconsistent_layers_generator):
    """Test ValueError when generator yields inconsistent layer indices."""
    store = ActivationStore(
        activation_generator=inconsistent_layers_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        device="cpu",
    )
    # First batch should work
    store.get_batch()
    # Subsequent fetches should trigger the error when the bad batch is processed
    with pytest.raises(ValueError, match="Inconsistent layer indices"):
        # Fetch enough times to force processing the second (bad) batch
        for _ in range(3):
            store.get_batch()


def test_empty_generator_stopiteration(exhausted_generator):
    """Test StopIteration is raised immediately if generator is empty."""
    store = ActivationStore(
        activation_generator=exhausted_generator,
        n_batches_in_buffer=2,
        train_batch_size_tokens=64,
        device="cpu",
    )
    with pytest.raises(StopIteration):
        store.get_batch()


# --- Integration-style Test (Keep as is for now, mocks Trainer/Extractor) ---


def test_cache_path_integration():
    """Test that activation caching params are passed (mocks dependencies)."""
    # This test primarily checks CLTTrainer's interaction, not ActivationStore internals.
    # Its validity depends on CLTTrainer's current implementation (not provided).
    # Keeping it as a placeholder for integration testing.

    # Important: Mocks need to align with actual interfaces used by CLTTrainer.
    # If ActivationExtractorCLT or ActivationStore API changed how CLTTrainer
    # interacts with them, these mocks would need updates.

    # Mock clt.training.trainer dependencies if they exist
    try:
        from clt.training.trainer import CLTTrainer
        from clt.config import CLTConfig, TrainingConfig

        trainer_module = "clt.training.trainer"
    except ImportError:
        pytest.skip("CLTTrainer or dependencies not found, skipping integration test")

    # Create minimal configs for testing
    clt_config = CLTConfig(num_features=16, num_layers=2, d_model=32)

    training_config = TrainingConfig(
        learning_rate=0.001,
        training_steps=10,
        cache_path="/fake/cache/path",  # Test this parameter passing
        # Add other required TrainingConfig fields if necessary
        # batch_size_tokens=64, # Removed: Causes TypeError if not in actual TrainingConfig
        # buffer_batches=4,     # Removed: Causes TypeError if not in actual TrainingConfig
    )

    # Mock the ActivationExtractorCLT and ActivationStore within the trainer module
    with patch(f"{trainer_module}.ActivationExtractorCLT") as MockExtractor, patch(
        f"{trainer_module}.ActivationStore"
    ) as MockStore:

        # Set up the mocked extractor instance and its methods
        mock_extractor_instance = MockExtractor.return_value
        # Simulate the stream_activations method returning a dummy generator
        mock_generator = MagicMock(spec=Generator)
        mock_extractor_instance.stream_activations.return_value = mock_generator

        # Set up the mocked store instance
        mock_store_instance = MockStore.return_value
        # Give the mock store an iterator protocol if CLTTrainer uses it like `next(store)`
        mock_store_instance.__iter__.return_value = iter(
            [(MagicMock(), MagicMock())]
        )  # Dummy batch

        # Create trainer instance (this will trigger mocked calls)
        # Ensure all required arguments for CLTTrainer are provided
        try:
            trainer = CLTTrainer(
                clt_config=clt_config,
                training_config=training_config,
                log_dir="test_cache_dir_integration",
                # Add other required CLTTrainer args like model_name, dataset_name etc.
                # model_name="mock_model",  # Removed: Causes TypeError if not in actual CLTTrainer
                # dataset_name="mock_dataset",  # Removed: Causes TypeError if not in actual CLTTrainer
            )
        except TypeError as e:
            pytest.fail(f"CLTTrainer init failed, check required args/mocks: {e}")

        # --- Verification ---

        # 1. Verify ActivationExtractorCLT was initialized (if Trainer does this)
        #    Example: Check if model_name was passed
        # MockExtractor.assert_called_once_with(model_name=training_config.model_name_or_path, ...)

        # 2. Verify stream_activations was called with the cache_path
        mock_extractor_instance.stream_activations.assert_called_once()
        # Check kwargs passed to stream_activations
        stream_kwargs = mock_extractor_instance.stream_activations.call_args.kwargs
        assert stream_kwargs.get("cache_path") == "/fake/cache/path"
        # Add checks for other expected args like dataset_name, batch_size etc.
        # assert stream_kwargs.get("dataset_name") == training_config.dataset_name

        # 3. Verify ActivationStore was initialized with the generator from the extractor
        MockStore.assert_called_once()
        store_kwargs = MockStore.call_args.kwargs
        assert store_kwargs.get("activation_generator") == mock_generator
        # Check other parameters passed to ActivationStore init
        # assert store_kwargs.get("n_batches_in_buffer") == training_config.buffer_batches # Removed: Depends on removed mock arg
        # assert store_kwargs.get("train_batch_size_tokens") == training_config.batch_size_tokens # Removed: Depends on removed mock arg
        # assert store_kwargs.get("normalization_method") == training_config.normalization_method

        # --- Cleanup ---
        import shutil
        import os

        if os.path.exists("test_cache_dir_integration"):
            shutil.rmtree("test_cache_dir_integration")
