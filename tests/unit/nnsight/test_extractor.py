import pytest
import torch
from datasets import Dataset
from clt.nnsight.extractor import ActivationExtractorCLT
from unittest.mock import patch
import gc

# Use a small standard model for testing
TEST_MODEL_NAME = "gpt2"
TEST_DEVICE = "cpu"  # Force CPU for easier testing


@pytest.fixture(scope="module")
def extractor():
    """Fixture to create an ActivationExtractorCLT instance for testing."""
    instance = ActivationExtractorCLT(
        model_name=TEST_MODEL_NAME,
        mlp_input_module_path_template="transformer.h.{}.mlp.input",  # Valid path for GPT-2
        mlp_output_module_path_template="transformer.h.{}.mlp.output",  # Valid path for GPT-2
        device=TEST_DEVICE,
        context_size=32,  # Keep small for testing
        inference_batch_size=4,  # Small batch size
    )
    yield instance
    # Cleanup - necessary for nnsight models?
    del instance.model
    del instance.tokenizer
    del instance
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def dummy_dataset():
    """Fixture for a small dummy dataset."""
    data = {
        "text": [
            "This is the first sentence.",
            "Here is another sentence.",
            "A short one.",
            "This is the fourth piece of text.",
            "Sentence five.",
            "And the final sixth sentence for testing.",
        ]
    }
    return Dataset.from_dict(data)


# --- Initialization Tests ---


def test_init_loads_model_tokenizer(extractor):
    """Test if model and tokenizer are loaded during initialization."""
    assert extractor.model is not None, "Model should be loaded"
    assert extractor.tokenizer is not None, "Tokenizer should be loaded"
    assert extractor.tokenizer.pad_token is not None, "Pad token should be set"


def test_init_detects_layers(extractor):
    """Test if the number of layers is detected correctly."""
    # GPT-2 has 12 layers
    assert extractor.num_layers == 12, f"Expected 12 layers for gpt2, found {extractor.num_layers}"


def test_init_device(extractor):
    """Test if the device is set correctly."""
    assert str(extractor.device) == TEST_DEVICE, f"Expected device {TEST_DEVICE}, got {extractor.device}"


def test_init_custom_paths():
    """Test initialization with custom MLP path templates."""
    custom_input = "custom.input.path.{}"
    custom_output = "custom.output.path.{}"
    instance = ActivationExtractorCLT(
        model_name=TEST_MODEL_NAME,
        device=TEST_DEVICE,
        mlp_input_module_path_template=custom_input,
        mlp_output_module_path_template=custom_output,
    )
    assert instance.mlp_input_module_path_template == custom_input
    assert instance.mlp_output_module_path_template == custom_output
    del instance.model, instance.tokenizer, instance
    gc.collect()
    torch.cuda.empty_cache()


# --- Activation Streaming Tests ---


# Mock load_dataset to avoid actual disk/network access during tests
@patch("clt.nnsight.extractor.load_dataset")
def test_stream_activations_basic(mock_load_dataset, extractor, dummy_dataset):
    """Test basic activation streaming yields correct format."""
    mock_load_dataset.return_value = dummy_dataset
    d_model = extractor.model.config.hidden_size  # 768 for gpt2

    activation_generator = extractor.stream_activations(
        dataset_path="dummy_path",  # Path doesn't matter due to mocking
        dataset_split="train",
        dataset_text_column="text",
        streaming=False,  # Use non-streaming for predictable batching with small dataset
    )

    # Get the first yielded batch
    first_batch_inputs, first_batch_targets = next(activation_generator)

    assert isinstance(first_batch_inputs, dict), "Inputs should be a dict"
    assert isinstance(first_batch_targets, dict), "Targets should be a dict"
    assert len(first_batch_inputs) == extractor.num_layers, "Inputs dict should have all layers"
    assert len(first_batch_targets) == extractor.num_layers, "Targets dict should have all layers"

    # Check activations for layer 0
    assert 0 in first_batch_inputs
    assert 0 in first_batch_targets
    assert isinstance(first_batch_inputs[0], torch.Tensor)
    assert isinstance(first_batch_targets[0], torch.Tensor)

    # Check shape: [n_valid_tokens, d_model]
    # n_valid_tokens depends on tokenization and batch size
    assert first_batch_inputs[0].ndim == 2, "Input tensor should be 2D"
    assert first_batch_inputs[0].shape[1] == d_model, f"Input tensor dim 1 should be d_model ({d_model})"
    assert first_batch_targets[0].ndim == 2, "Target tensor should be 2D"
    assert first_batch_targets[0].shape[1] == d_model, f"Target tensor dim 1 should be d_model ({d_model})"
    assert (
        first_batch_inputs[0].shape[0] == first_batch_targets[0].shape[0]
    ), "Input and target should have same number of tokens"
    assert first_batch_inputs[0].shape[0] > 0, "Should have extracted some tokens"

    # Check device
    assert str(first_batch_inputs[0].device) == TEST_DEVICE
    assert str(first_batch_targets[0].device) == TEST_DEVICE

    # Consume the rest of the generator to ensure no errors
    for _ in activation_generator:
        pass


@patch("clt.nnsight.extractor.load_dataset")
def test_stream_activations_padding_exclusion(mock_load_dataset, extractor):
    """Test that padding tokens are excluded."""
    # Create data with varying lengths to force padding
    data = {
        "text": [
            "Short.",
            "This is a slightly longer sentence.",
            "Medium length text.",
            "Tiny.",
        ]
    }
    dataset = Dataset.from_dict(data)
    mock_load_dataset.return_value = dataset
    d_model = extractor.model.config.hidden_size

    # Tokenize manually to find total non-pad tokens
    tokenizer_args = {
        "truncation": True,
        "max_length": extractor.context_size,
        "padding": "longest",
        "return_tensors": "pt",
    }
    tokenized = extractor.tokenizer(data["text"], **tokenizer_args)
    total_non_pad_tokens = tokenized["attention_mask"].sum().item()

    # Stream activations
    activation_generator = extractor.stream_activations(
        dataset_path="dummy_path",
        dataset_split="train",
        dataset_text_column="text",
        streaming=False,
    )

    total_yielded_tokens = 0
    for inputs_dict, targets_dict in activation_generator:
        # Check layer 0 for token count
        if 0 in inputs_dict:
            total_yielded_tokens += inputs_dict[0].shape[0]
            assert inputs_dict[0].shape[1] == d_model
        if 0 in targets_dict:
            assert targets_dict[0].shape[1] == d_model

    assert (
        total_yielded_tokens == total_non_pad_tokens
    ), f"Total yielded tokens ({total_yielded_tokens}) should equal total non-pad tokens ({total_non_pad_tokens})"


@patch("clt.nnsight.extractor.load_dataset")
def test_stream_activations_max_samples(mock_load_dataset, extractor, dummy_dataset):
    """Test processing a limited number of samples."""
    mock_load_dataset.return_value = dummy_dataset
    max_samples = 2
    store_batch_size = 1  # Ensure we process one sample at a time for simplicity

    instance = ActivationExtractorCLT(
        model_name=TEST_MODEL_NAME,
        mlp_input_module_path_template="transformer.h.{}.mlp.input",  # Valid path for GPT-2
        mlp_output_module_path_template="transformer.h.{}.mlp.output",  # Valid path for GPT-2
        device=TEST_DEVICE,
        context_size=32,
        inference_batch_size=store_batch_size,
    )

    # Tokenize the first max_samples manually to get expected token count
    tokenizer_args = {
        "truncation": True,
        "max_length": instance.context_size,
        "padding": "longest",
        "return_tensors": "pt",
    }
    first_samples_text = dummy_dataset[:max_samples]["text"]
    tokenized = instance.tokenizer(first_samples_text, **tokenizer_args)
    # Ensure we're working with the actual tensor
    attention_mask_tensor = tokenized["attention_mask"]
    if hasattr(attention_mask_tensor, "to"):  # It's already a tensor
        expected_tokens = attention_mask_tensor.sum().item()  # type: ignore
    else:  # It might be a list or numpy array
        expected_tokens = torch.tensor(attention_mask_tensor).sum().item()

    # Create a limited dataset with only the first max_samples
    limited_data = {"text": dummy_dataset[:max_samples]["text"]}
    limited_dataset = Dataset.from_dict(limited_data)
    mock_load_dataset.return_value = limited_dataset

    activation_generator = instance.stream_activations(
        dataset_path="dummy_path",
        dataset_split="train",
        dataset_text_column="text",
        streaming=False,  # Use non-streaming for predictable behavior
    )

    total_yielded_tokens = 0
    for inputs_dict, _ in activation_generator:
        if 0 in inputs_dict:
            total_yielded_tokens += inputs_dict[0].shape[0]

    assert (
        total_yielded_tokens == expected_tokens
    ), f"Expected {expected_tokens} tokens for {max_samples} samples, got {total_yielded_tokens}"

    del instance.model, instance.tokenizer, instance
    gc.collect()
    torch.cuda.empty_cache()


@patch("clt.nnsight.extractor.load_dataset")
def test_stream_activations_final_batch(mock_load_dataset, extractor, dummy_dataset):
    """Test processing of the final, potentially smaller, batch."""
    mock_load_dataset.return_value = dummy_dataset
    num_samples = len(dummy_dataset)  # 6
    batch_size = extractor.inference_batch_size  # 4
    num_full_batches = num_samples // batch_size  # 6 // 4 = 1
    final_batch_size = num_samples % batch_size  # 6 % 4 = 2
    assert final_batch_size > 0, "Test setup assumes a non-empty final batch"

    # Tokenize the last few samples manually
    tokenizer_args = {
        "truncation": True,
        "max_length": extractor.context_size,
        "padding": "longest",
        "return_tensors": "pt",
    }
    final_samples_text = dummy_dataset[num_full_batches * batch_size :]["text"]
    tokenized = extractor.tokenizer(final_samples_text, **tokenizer_args)
    expected_final_batch_tokens = tokenized["attention_mask"].sum().item()

    activation_generator = extractor.stream_activations(
        dataset_path="dummy_path",
        dataset_split="train",
        dataset_text_column="text",
        streaming=False,  # Easier to track batches
    )

    batches = list(activation_generator)
    assert len(batches) == num_full_batches + 1, f"Expected {num_full_batches + 1} batches, got {len(batches)}"

    # Check the last batch
    final_inputs_dict, final_targets_dict = batches[-1]
    final_yielded_tokens = 0
    if 0 in final_inputs_dict:
        final_yielded_tokens = final_inputs_dict[0].shape[0]
        assert final_inputs_dict[0].ndim == 2
        assert final_inputs_dict[0].shape[1] == extractor.model.config.hidden_size
    if 0 in final_targets_dict:
        assert final_targets_dict[0].ndim == 2
        assert final_targets_dict[0].shape[1] == extractor.model.config.hidden_size

    assert (
        final_yielded_tokens == expected_final_batch_tokens
    ), f"Expected {expected_final_batch_tokens} tokens in final batch, got {final_yielded_tokens}"


@patch("clt.nnsight.extractor.load_dataset")
def test_preprocess_text(mock_load_dataset, extractor):
    """Test the internal _preprocess_text method for chunking."""
    # Access the protected method for testing (common in Python testing)
    preprocess_func = extractor._preprocess_text

    short_text = "This is short."
    chunks = preprocess_func(short_text)
    assert chunks == [short_text], "Short text shouldn't be chunked"

    # Create text slightly longer than context_size requires chunking
    # context_size is 32. Tokens are roughly <= chars.
    # Need > context_size tokens. Add padding for special tokens.
    long_text = "word " * (extractor.context_size) + "extra words"
    long_text_tokens = extractor.tokenizer.encode(long_text, add_special_tokens=False)
    assert len(long_text_tokens) > extractor.context_size - 2  # Ensure it needs chunking

    chunks = preprocess_func(long_text)
    assert len(chunks) > 1, "Long text should be chunked into multiple parts"

    # Check if chunks reconstruct roughly the original (minus tokenization artifacts)
    reconstructed = "".join(chunks)
    tokenized_reconstructed = extractor.tokenizer.encode(reconstructed, add_special_tokens=False)

    # Check token IDs match, allowing for minor differences due to chunk boundaries
    assert len(tokenized_reconstructed) >= len(long_text_tokens) - len(chunks)  # Allow for boundary effects
    assert all(
        t_orig == t_recon for t_orig, t_recon in zip(long_text_tokens, tokenized_reconstructed)
    ), "Reconstructed text tokens should largely match original"


def test_resolve_dtype(extractor):
    """Test the _resolve_dtype utility method."""
    assert extractor._resolve_dtype(None) is None
    assert extractor._resolve_dtype(torch.float32) == torch.float32
    assert extractor._resolve_dtype("float16") == torch.float16
    assert extractor._resolve_dtype("bfloat16") == torch.bfloat16
    # Check warning for invalid string (requires capturing logs or mocking logger)
    with patch("clt.nnsight.extractor.logger.warning") as mock_warning:
        assert extractor._resolve_dtype("invalid_dtype_string") is None
        mock_warning.assert_called_once()
