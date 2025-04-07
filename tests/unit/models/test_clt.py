import pytest
import torch
import torch.nn as nn

# Import the classes to test first
from clt.models.clt import JumpReLU, CrossLayerTranscoder


# Mock CLTConfig
class MockCLTConfig:
    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 32)
        self.num_features = kwargs.get("num_features", 64)
        self.num_layers = kwargs.get("num_layers", 4)
        self.activation_fn = kwargs.get("activation_fn", "relu")
        self.jumprelu_threshold = kwargs.get("jumprelu_threshold", 0.03)
        self.__dict__.update(kwargs)  # Allow other attributes


# --- Test JumpReLU ---


@pytest.fixture
def jumprelu_input():
    return torch.randn(10, 20, requires_grad=True)


@pytest.mark.parametrize("threshold", [0.0, 0.03, 0.1])
def test_jumprelu_forward(jumprelu_input, threshold):
    """Test the forward pass of JumpReLU."""
    output = JumpReLU.apply(jumprelu_input, threshold, 1.0)
    expected_output = (jumprelu_input >= threshold).float() * jumprelu_input
    assert isinstance(output, torch.Tensor)  # Help linter with type
    assert isinstance(expected_output, torch.Tensor)  # Help linter with type
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("threshold, bandwidth", [(0.03, 1.0), (0.1, 0.5)])
def test_jumprelu_backward(jumprelu_input, threshold, bandwidth):
    """Test the backward pass (straight-through estimator) of JumpReLU."""
    output = JumpReLU.apply(jumprelu_input, threshold, bandwidth)
    assert isinstance(output, torch.Tensor)  # Help linter with type

    # Compute theoretical gradient
    grad_output = torch.ones_like(output)
    grad_input_expected = grad_output.clone()
    mask = jumprelu_input.abs() < threshold * (1 + bandwidth)
    grad_input_expected = grad_input_expected * (~mask).float()

    # Compute autograd gradient
    output.backward(grad_output)
    grad_input_actual = jumprelu_input.grad

    assert grad_input_actual is not None
    assert torch.allclose(grad_input_actual, grad_input_expected)


# --- Test CrossLayerTranscoder ---


@pytest.fixture
def clt_config_relu():
    return MockCLTConfig(
        d_model=16, num_features=32, num_layers=3, activation_fn="relu"
    )


@pytest.fixture
def clt_config_jumprelu():
    return MockCLTConfig(
        d_model=16,
        num_features=32,
        num_layers=3,
        activation_fn="jumprelu",
        jumprelu_threshold=0.05,
    )


@pytest.fixture(params=["relu", "jumprelu"])
def clt_model(request):
    if request.param == "relu":
        config = MockCLTConfig(
            d_model=16, num_features=32, num_layers=3, activation_fn="relu"
        )
    else:
        config = MockCLTConfig(
            d_model=16,
            num_features=32,
            num_layers=3,
            activation_fn="jumprelu",
            jumprelu_threshold=0.05,
        )
    return CrossLayerTranscoder(config)


@pytest.fixture
def sample_inputs():
    batch_size = 4
    seq_len = 10
    d_model = 16
    num_layers = 3
    inputs = {i: torch.randn(batch_size, seq_len, d_model) for i in range(num_layers)}
    return inputs


def test_clt_init(clt_model):
    """Test CLT initialization."""
    config = clt_model.config
    assert isinstance(clt_model, nn.Module)
    # Linter might complain about ModuleList not being sized/iterable, but it is.
    assert len(clt_model.encoders) == config.num_layers
    assert all(isinstance(enc, nn.Linear) for enc in clt_model.encoders)
    assert all(enc.in_features == config.d_model for enc in clt_model.encoders)
    assert all(enc.out_features == config.num_features for enc in clt_model.encoders)
    assert all(enc.bias is None for enc in clt_model.encoders)

    num_expected_decoders = sum(range(1, config.num_layers + 1))
    # Linter might complain about ModuleDict not being sized, but it is.
    assert len(clt_model.decoders) == num_expected_decoders
    for src in range(config.num_layers):
        for tgt in range(src, config.num_layers):
            key = f"{src}->{tgt}"
            # Linter might complain about str key access on ModuleDict, but it works.
            assert key in clt_model.decoders
            dec = clt_model.decoders[key]
            assert isinstance(dec, nn.Linear)
            assert dec.in_features == config.num_features
            assert dec.out_features == config.d_model
            assert dec.bias is None

    if config.activation_fn == "jumprelu":
        assert isinstance(clt_model.threshold, nn.Parameter)
        assert clt_model.threshold.shape == (config.num_features,)
        assert torch.allclose(
            clt_model.threshold.data,
            torch.ones(config.num_features) * config.jumprelu_threshold,
        )

    # Calculate and print memory footprint
    total_params = sum(p.numel() for p in clt_model.parameters() if p.requires_grad)
    # Assuming float32 (4 bytes per parameter)
    estimated_memory_bytes = total_params * 4
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
    print(f"\n[Memory Footprint Info for {clt_model.config.activation_fn} CLT]")
    print(f"  - Total Trainable Parameters: {total_params:,}")
    print(f"  - Estimated Memory (MB): {estimated_memory_mb:.2f} MB")


def test_clt_get_preactivations(clt_model, sample_inputs):
    """Test getting pre-activations."""
    config = clt_model.config
    x = sample_inputs[0]
    preact = clt_model.get_preactivations(x, 0)
    assert preact.shape == (x.shape[0], x.shape[1], config.num_features)


def test_clt_encode(clt_model, sample_inputs):
    """Test the encode method."""
    config = clt_model.config
    x = sample_inputs[0]
    encoded = clt_model.encode(x, 0)
    assert encoded.shape == (x.shape[0], x.shape[1], config.num_features)

    # Check if activation applied (non-negative for ReLU, thresholded for JumpReLU)
    if config.activation_fn == "relu":
        assert torch.all(encoded >= 0)
    elif config.activation_fn == "jumprelu":
        preact = clt_model.get_preactivations(x, 0)
        threshold = clt_model.threshold
        # Check that values below threshold are zero
        # Need to broadcast threshold correctly
        threshold_expanded = threshold.view(1, 1, -1).expand_as(preact)
        assert torch.all(encoded[preact < threshold_expanded] == 0)
        # Check that values above or equal to threshold match preactivation
        assert torch.all(
            encoded[preact >= threshold_expanded]
            == preact[preact >= threshold_expanded]
        )


def test_clt_decode(clt_model, sample_inputs):
    """Test the decode method."""
    config = clt_model.config
    activations = {i: clt_model.encode(x, i) for i, x in sample_inputs.items()}

    for layer_idx in range(config.num_layers):
        reconstruction = clt_model.decode(activations, layer_idx)
        assert reconstruction.shape == (
            sample_inputs[0].shape[0],
            sample_inputs[0].shape[1],
            config.d_model,
        )

        # Check reconstruction calculation (simplified check)
        # Calculate expected reconstruction for layer_idx=1
        if layer_idx == 1 and 0 in activations and 1 in activations:
            dec_0_1 = clt_model.decoders["0->1"]
            dec_1_1 = clt_model.decoders["1->1"]
            expected_rec_1 = dec_0_1(activations[0]) + dec_1_1(activations[1])
            assert torch.allclose(reconstruction, expected_rec_1, atol=1e-6)


def test_clt_forward(clt_model, sample_inputs):
    """Test the forward pass of the CLT."""
    config = clt_model.config
    reconstructions = clt_model(sample_inputs)

    assert isinstance(reconstructions, dict)
    assert len(reconstructions) == len(sample_inputs)
    assert all(idx in reconstructions for idx in sample_inputs.keys())

    for layer_idx, recon in reconstructions.items():
        assert recon.shape == (
            sample_inputs[layer_idx].shape[0],
            sample_inputs[layer_idx].shape[1],
            config.d_model,
        )
        # Check if forward pass output matches manual encode/decode
        activations = {i: clt_model.encode(x, i) for i, x in sample_inputs.items()}
        expected_recon = clt_model.decode(activations, layer_idx)
        assert torch.allclose(recon, expected_recon, atol=1e-6)


def test_clt_get_feature_activations(clt_model, sample_inputs):
    """Test getting all feature activations."""
    config = clt_model.config
    activations = clt_model.get_feature_activations(sample_inputs)

    assert isinstance(activations, dict)
    assert len(activations) == len(sample_inputs)
    assert all(idx in activations for idx in sample_inputs.keys())

    for layer_idx, act in activations.items():
        assert act.shape == (
            sample_inputs[layer_idx].shape[0],
            sample_inputs[layer_idx].shape[1],
            config.num_features,
        )
        # Check if it matches encode output
        expected_act = clt_model.encode(sample_inputs[layer_idx], layer_idx)
        assert torch.allclose(act, expected_act)


def test_clt_get_decoder_norms(clt_model):
    """Test calculation of decoder norms."""
    config = clt_model.config
    decoder_norms = clt_model.get_decoder_norms()

    assert decoder_norms.shape == (config.num_layers, config.num_features)
    assert not torch.any(torch.isnan(decoder_norms))
    assert torch.all(decoder_norms >= 0)

    # Manual check for one feature (e.g., feature 0) at layer 0
    expected_norm_0_0_sq = 0
    for tgt_layer in range(config.num_layers):  # From src_layer=0
        decoder = clt_model.decoders[f"0->{tgt_layer}"]
        # Norm of the first column (feature 0)
        expected_norm_0_0_sq += torch.norm(decoder.weight[:, 0], dim=0) ** 2

    expected_norm_0_0 = torch.sqrt(expected_norm_0_0_sq)
    assert torch.allclose(decoder_norms[0, 0], expected_norm_0_0)


# --- Fixtures for GPT-2 Small Size Test Case ---


@pytest.fixture
def clt_config_gpt2_small():
    # Parameters similar to GPT-2 Small
    return MockCLTConfig(
        d_model=768,
        num_features=3072,  # d_model * 4
        num_layers=12,
        activation_fn="relu",  # Keep it simple for size test
    )


@pytest.fixture
def clt_model_gpt2_small(clt_config_gpt2_small):
    return CrossLayerTranscoder(clt_config_gpt2_small)


@pytest.fixture
def sample_inputs_gpt2_small(clt_config_gpt2_small):
    # Smaller batch/seq_len for faster test with large d_model
    batch_size = 1
    seq_len = 4
    config = clt_config_gpt2_small
    inputs = {
        i: torch.randn(batch_size, seq_len, config.d_model)
        for i in range(config.num_layers)
    }
    return inputs


# --- Tests for Small Case (Parametrized for relu/jumprelu) ---


@pytest.fixture(params=["relu", "jumprelu"])
def clt_model(request):
    if request.param == "relu":
        config = MockCLTConfig(
            d_model=16, num_features=32, num_layers=3, activation_fn="relu"
        )
    else:
        config = MockCLTConfig(
            d_model=16,
            num_features=32,
            num_layers=3,
            activation_fn="jumprelu",
            jumprelu_threshold=0.05,
        )
    return CrossLayerTranscoder(config)


# --- Tests Duplicated for GPT-2 Small Size ---
# (Using the specific gpt2_small fixtures)


def test_clt_init_gpt2_small(clt_model_gpt2_small):
    """Test CLT initialization with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small  # Rename for clarity within test
    config = clt_model.config
    assert isinstance(clt_model, nn.Module)
    # Linter might complain about ModuleList not being sized/iterable, but it is.
    assert len(clt_model.encoders) == config.num_layers
    assert all(isinstance(enc, nn.Linear) for enc in clt_model.encoders)
    assert all(enc.in_features == config.d_model for enc in clt_model.encoders)
    assert all(enc.out_features == config.num_features for enc in clt_model.encoders)
    assert all(enc.bias is None for enc in clt_model.encoders)

    num_expected_decoders = sum(range(1, config.num_layers + 1))
    # Linter might complain about ModuleDict not being sized, but it is.
    assert len(clt_model.decoders) == num_expected_decoders
    for src in range(config.num_layers):
        for tgt in range(src, config.num_layers):
            key = f"{src}->{tgt}"
            # Linter might complain about str key access on ModuleDict, but it works.
            assert key in clt_model.decoders
            dec = clt_model.decoders[key]
            assert isinstance(dec, nn.Linear)
            assert dec.in_features == config.num_features
            assert dec.out_features == config.d_model
            assert dec.bias is None

    # Calculate and print memory footprint
    total_params = sum(p.numel() for p in clt_model.parameters() if p.requires_grad)
    # Assuming float32 (4 bytes per parameter)
    estimated_memory_bytes = total_params * 4
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
    print(
        f"\n[Memory Footprint Info for GPT-2 Small Size ({clt_model.config.activation_fn} CLT)]"
    )
    print(f"  - Total Trainable Parameters: {total_params:,}")
    print(f"  - Estimated Memory (MB): {estimated_memory_mb:.2f} MB")


def test_clt_get_preactivations_gpt2_small(
    clt_model_gpt2_small, sample_inputs_gpt2_small
):
    """Test getting pre-activations with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    x = sample_inputs[0]
    preact = clt_model.get_preactivations(x, 0)
    assert preact.shape == (x.shape[0], x.shape[1], config.num_features)


def test_clt_encode_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the encode method with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    x = sample_inputs[0]
    encoded = clt_model.encode(x, 0)
    assert encoded.shape == (x.shape[0], x.shape[1], config.num_features)
    # Only checking ReLU here as per fixture config
    assert torch.all(encoded >= 0)


def test_clt_decode_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the decode method with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    # Use smaller set for faster test execution
    activations = {
        i: clt_model.encode(sample_inputs[i], i) for i in range(config.num_layers)
    }

    for layer_idx in range(config.num_layers):
        reconstruction = clt_model.decode(activations, layer_idx)
        assert reconstruction.shape == (
            sample_inputs[0].shape[0],
            sample_inputs[0].shape[1],
            config.d_model,
        )
        # Skip detailed calculation check for large model test


def test_clt_forward_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the forward pass of the CLT with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    reconstructions = clt_model(sample_inputs)

    assert isinstance(reconstructions, dict)
    assert len(reconstructions) == len(sample_inputs)
    assert all(idx in reconstructions for idx in sample_inputs.keys())

    for layer_idx, recon in reconstructions.items():
        assert recon.shape == (
            sample_inputs[layer_idx].shape[0],
            sample_inputs[layer_idx].shape[1],
            config.d_model,
        )
        # Skip detailed calculation check for large model test


def test_clt_get_feature_activations_gpt2_small(
    clt_model_gpt2_small, sample_inputs_gpt2_small
):
    """Test getting all feature activations with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    activations = clt_model.get_feature_activations(sample_inputs)

    assert isinstance(activations, dict)
    assert len(activations) == len(sample_inputs)
    assert all(idx in activations for idx in sample_inputs.keys())

    for layer_idx, act in activations.items():
        assert act.shape == (
            sample_inputs[layer_idx].shape[0],
            sample_inputs[layer_idx].shape[1],
            config.num_features,
        )
        # Skip detailed calculation check


def test_clt_get_decoder_norms_gpt2_small(clt_model_gpt2_small):
    """Test calculation of decoder norms with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    config = clt_model.config
    decoder_norms = clt_model.get_decoder_norms()

    assert decoder_norms.shape == (config.num_layers, config.num_features)
    assert not torch.any(torch.isnan(decoder_norms))
    assert torch.all(decoder_norms >= 0)
    # Skip detailed calculation check for large model test


# Note: Tests assume existence of clt.config.CLTConfig, which is mocked here.
# Need to ensure the actual CLTConfig class aligns or adjust mock.
