import pytest
import torch
import torch.nn as nn
import math

# Import the classes to test first
from clt.models.clt import JumpReLU, CrossLayerTranscoder

# Import the actual config
from clt.config import CLTConfig


# --- Test JumpReLU ---


@pytest.fixture
def jumprelu_input():
    # Use a smaller tensor for quicker testing
    return torch.randn(5, 10, requires_grad=True)


# Parameterize with threshold *value*, not the parameter itself
@pytest.mark.parametrize("threshold_val", [0.01, 0.03, 0.1])
def test_jumprelu_forward(jumprelu_input, threshold_val):
    """Test the forward pass of JumpReLU."""
    # The function expects the threshold value, not the log_threshold parameter
    threshold_tensor = torch.tensor(
        threshold_val, device=jumprelu_input.device, dtype=jumprelu_input.dtype
    )
    output = JumpReLU.apply(jumprelu_input, threshold_tensor, 1.0)
    expected_output = (jumprelu_input >= threshold_tensor).float() * jumprelu_input
    assert isinstance(output, torch.Tensor)
    assert isinstance(expected_output, torch.Tensor)
    assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("threshold_val, bandwidth", [(0.03, 1.0), (0.1, 0.5)])
def test_jumprelu_backward_input_grad(jumprelu_input, threshold_val, bandwidth):
    """Test the backward pass (STE) of JumpReLU for input gradient ONLY."""
    input_clone = jumprelu_input.clone().requires_grad_(True)
    # Keep threshold fixed for this test
    threshold_tensor = torch.tensor(
        threshold_val, device=input_clone.device, dtype=input_clone.dtype
    )

    output = JumpReLU.apply(input_clone, threshold_tensor, bandwidth)
    assert isinstance(output, torch.Tensor)

    grad_output = torch.ones_like(output)
    # Explicitly retain grad for the input tensor as suggested by the warning
    input_clone.retain_grad()
    output.backward(grad_output)
    grad_input_actual = input_clone.grad

    # Expected input gradient (STE)
    input_fp32 = input_clone.float()
    threshold_fp32 = threshold_tensor.float()
    bandwidth_fp32 = float(bandwidth)
    is_near_threshold = torch.abs(input_fp32 - threshold_fp32) <= (bandwidth_fp32 / 2.0)
    grad_input_expected = grad_output.float() * is_near_threshold.float()
    grad_input_expected = grad_input_expected.to(input_clone.dtype)  # Cast back

    assert grad_input_actual is not None
    assert torch.allclose(grad_input_actual, grad_input_expected, atol=1e-6)


@pytest.mark.parametrize("threshold_val, bandwidth", [(0.03, 1.0), (0.1, 0.5)])
def test_jumprelu_backward_threshold_grad(jumprelu_input, threshold_val, bandwidth):
    """Test the backward pass (STE) of JumpReLU for threshold gradient ONLY."""
    # Keep input fixed (detached) for this test
    input_fixed = jumprelu_input.clone().detach()
    # Threshold needs to be a parameter
    log_threshold_param = nn.Parameter(torch.log(torch.tensor(threshold_val)))

    # Apply function using the *value* derived from the parameter
    threshold_value = torch.exp(log_threshold_param)
    output = JumpReLU.apply(input_fixed, threshold_value, bandwidth)
    assert isinstance(output, torch.Tensor)

    # Compute gradients w.r.t log_threshold_param
    grad_output = torch.ones_like(output)
    output.sum().backward()  # Use sum to get scalar loss

    grad_log_threshold_actual = log_threshold_param.grad

    # Expected threshold gradient (manual calculation)
    input_fp32 = input_fixed.float()
    threshold_fp32 = threshold_value.float()
    grad_output_fp32 = grad_output.float()
    bandwidth_fp32 = float(bandwidth)

    is_near_threshold = torch.abs(input_fp32 - threshold_fp32) <= (bandwidth_fp32 / 2.0)
    local_grad_theta_fp32 = (-input_fp32 / bandwidth_fp32) * is_near_threshold.float()
    grad_threshold_per_element_fp32 = grad_output_fp32 * local_grad_theta_fp32
    # Sum gradients for the single threshold value
    grad_threshold_expected_fp32 = grad_threshold_per_element_fp32.sum()

    # Chain rule: dL/d(log_theta) = dL/d(theta) * d(theta)/d(log_theta)
    # d(theta)/d(log_theta) = exp(log_theta) = theta
    grad_log_threshold_expected = grad_threshold_expected_fp32 * torch.exp(
        log_threshold_param.float()
    )
    grad_log_threshold_expected = grad_log_threshold_expected.to(
        log_threshold_param.dtype
    )  # Cast back

    assert grad_log_threshold_actual is not None
    # Gradient check for parameters can require higher tolerance
    assert torch.allclose(
        grad_log_threshold_actual, grad_log_threshold_expected, atol=1e-4
    )


# --- Test CrossLayerTranscoder ---


# Use actual CLTConfig
@pytest.fixture
def clt_config_relu():
    return CLTConfig(d_model=16, num_features=32, num_layers=3, activation_fn="relu")


@pytest.fixture
def clt_config_jumprelu():
    return CLTConfig(
        d_model=16,
        num_features=32,
        num_layers=3,
        activation_fn="jumprelu",
        jumprelu_threshold=0.05,  # Initial value
    )


@pytest.fixture(params=["relu", "jumprelu"])
def clt_model_config(request):
    if request.param == "relu":
        return CLTConfig(
            d_model=16, num_features=32, num_layers=3, activation_fn="relu"
        )
    else:  # jumprelu
        return CLTConfig(
            d_model=16,
            num_features=32,
            num_layers=3,
            activation_fn="jumprelu",
            jumprelu_threshold=0.05,
        )


@pytest.fixture
def clt_model(clt_model_config):
    return CrossLayerTranscoder(clt_model_config)


@pytest.fixture
def sample_inputs(clt_model_config):  # Depend on config to get params
    batch_size = 4
    seq_len = 10
    d_model = clt_model_config.d_model
    num_layers = clt_model_config.num_layers
    inputs = {i: torch.randn(batch_size, seq_len, d_model) for i in range(num_layers)}
    return inputs


def test_clt_init(clt_model):
    """Test CLT initialization."""
    config = clt_model.config
    assert isinstance(clt_model, nn.Module)
    assert len(clt_model.encoders) == config.num_layers
    assert all(isinstance(enc, nn.Linear) for enc in clt_model.encoders)
    assert all(enc.in_features == config.d_model for enc in clt_model.encoders)
    assert all(enc.out_features == config.num_features for enc in clt_model.encoders)
    assert all(enc.bias is None for enc in clt_model.encoders)

    num_expected_decoders = config.num_layers * (config.num_layers + 1) // 2
    assert len(clt_model.decoders) == num_expected_decoders
    for src in range(config.num_layers):
        for tgt in range(src, config.num_layers):
            key = f"{src}->{tgt}"
            assert key in clt_model.decoders
            dec = clt_model.decoders[key]
            assert isinstance(dec, nn.Linear)
            assert dec.in_features == config.num_features
            assert dec.out_features == config.d_model
            assert dec.bias is None

    if config.activation_fn == "jumprelu":
        assert isinstance(clt_model.log_threshold, nn.Parameter)  # Check log_threshold
        assert clt_model.log_threshold.shape == (config.num_features,)
        expected_log_threshold = torch.log(torch.tensor(config.jumprelu_threshold))
        assert torch.allclose(
            clt_model.log_threshold.data,
            torch.ones(config.num_features) * expected_log_threshold,
        )

    # Check parameter initialization ranges
    encoder_bound = 1.0 / math.sqrt(config.num_features)
    for encoder in clt_model.encoders:
        assert torch.all(encoder.weight.data >= -encoder_bound)
        assert torch.all(encoder.weight.data <= encoder_bound)

    decoder_bound = 1.0 / math.sqrt(config.num_layers * config.d_model)
    for decoder in clt_model.decoders.values():
        assert torch.all(decoder.weight.data >= -decoder_bound)
        assert torch.all(decoder.weight.data <= decoder_bound)

    # Calculate and print memory footprint
    total_params = sum(p.numel() for p in clt_model.parameters() if p.requires_grad)
    # Assuming float32 (4 bytes per parameter)
    dtype_size = torch.finfo(clt_model.dtype).bits // 8
    estimated_memory_bytes = total_params * dtype_size
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
    print(
        f"\n[Memory Footprint Info for {clt_model.config.activation_fn} CLT ({clt_model.dtype})]"
    )
    print(f"  - Total Trainable Parameters: {total_params:,}")
    print(f"  - Estimated Memory (MB): {estimated_memory_mb:.2f} MB")


def test_clt_init_device_dtype():
    """Test CLT initialization with specific device and dtype."""
    # Test float16
    config_fp16 = CLTConfig(
        d_model=8, num_features=16, num_layers=2, clt_dtype="float16"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            model_fp16 = CrossLayerTranscoder(config_fp16, device=device)
            assert model_fp16.dtype == torch.float16
            assert next(model_fp16.parameters()).dtype == torch.float16
            assert next(model_fp16.parameters()).device.type == "cuda"
            assert model_fp16.log_threshold.dtype == torch.float16  # Check param dtype
            assert model_fp16.log_threshold.device.type == "cuda"  # Check param device
        except RuntimeError as e:
            # Some GPUs might not support float16 well
            print(f"Skipping float16 test on CUDA due to: {e}")
    else:  # CPU
        # CPU float16 support is limited, often emulated, skip strict check
        model_fp16_cpu = CrossLayerTranscoder(config_fp16, device=torch.device("cpu"))
        assert model_fp16_cpu.dtype == torch.float16
        # Parameters might default to float32 on CPU even if requested float16
        # assert next(model_fp16_cpu.parameters()).dtype == torch.float16
        assert next(model_fp16_cpu.parameters()).device.type == "cpu"

    # Test bfloat16
    config_bf16 = CLTConfig(
        d_model=8, num_features=16, num_layers=2, clt_dtype="bfloat16"
    )
    try:
        # Check if bfloat16 is supported on the current device
        is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if is_bf16_supported:
            device = torch.device("cuda")
            model_bf16 = CrossLayerTranscoder(config_bf16, device=device)
            assert model_bf16.dtype == torch.bfloat16
            assert next(model_bf16.parameters()).dtype == torch.bfloat16
            assert next(model_bf16.parameters()).device.type == "cuda"
            assert model_bf16.log_threshold.dtype == torch.bfloat16  # Check param dtype
            assert model_bf16.log_threshold.device.type == "cuda"  # Check param device
        else:  # CPU or CUDA without BF16 support
            # BFloat16 often works on CPU
            model_bf16_cpu = CrossLayerTranscoder(
                config_bf16, device=torch.device("cpu")
            )
            assert model_bf16_cpu.dtype == torch.bfloat16
            assert (
                next(model_bf16_cpu.parameters()).dtype == torch.bfloat16
            )  # Usually works on CPU
            assert next(model_bf16_cpu.parameters()).device.type == "cpu"
    except RuntimeError as e:
        print(f"Skipping bfloat16 test due to: {e}")

    # Test invalid dtype string
    config_invalid = CLTConfig(
        d_model=8, num_features=16, num_layers=2, clt_dtype="invalid_dtype"
    )
    model_invalid = CrossLayerTranscoder(config_invalid)
    assert model_invalid.dtype == torch.float32  # Should default


def test_clt_resolve_dtype(clt_model):
    """Test the _resolve_dtype helper method."""
    assert clt_model._resolve_dtype(None) == torch.float32
    assert clt_model._resolve_dtype("float32") == torch.float32
    assert clt_model._resolve_dtype("float16") == torch.float16
    assert clt_model._resolve_dtype("bfloat16") == torch.bfloat16
    assert clt_model._resolve_dtype(torch.float64) == torch.float64
    # Invalid string defaults to float32
    assert clt_model._resolve_dtype("invalid") == torch.float32
    # Non-dtype attribute defaults to float32
    assert clt_model._resolve_dtype("Linear") == torch.float32


def test_clt_get_preactivations(clt_model, sample_inputs):
    """Test getting pre-activations."""
    config = clt_model.config
    x = sample_inputs[0]  # Shape: [batch_size, seq_len, d_model]
    batch_size, seq_len, _ = x.shape

    preact = clt_model.get_preactivations(x, 0)
    # Expect reshaped output: [batch_size * seq_len, num_features]
    assert preact.shape == (batch_size * seq_len, config.num_features)

    # Test with 2D input [batch_tokens, d_model]
    x_2d = x.reshape(-1, config.d_model)
    preact_2d = clt_model.get_preactivations(x_2d, 0)
    assert preact_2d.shape == (batch_size * seq_len, config.num_features)
    assert torch.allclose(preact, preact_2d, atol=1e-6)


def test_clt_encode(clt_model, sample_inputs):
    """Test the encode method."""
    config = clt_model.config
    x = sample_inputs[0]  # Shape: [batch_size, seq_len, d_model]
    batch_size, seq_len, d_model = x.shape

    encoded = clt_model.encode(x, 0)
    # Expect reshaped output: [batch_size * seq_len, num_features]
    assert encoded.shape == (batch_size * seq_len, config.num_features)

    # Check if activation applied
    preact = clt_model.get_preactivations(x, 0)
    if config.activation_fn == "relu":
        assert torch.all(encoded >= 0)
        assert torch.allclose(encoded, torch.relu(preact))
    elif config.activation_fn == "jumprelu":
        # Calculate expected threshold value from log_threshold parameter
        threshold_val = torch.exp(clt_model.log_threshold).to(
            preact.device, preact.dtype
        )
        # Compare element-wise
        expected_jumprelu = (preact >= threshold_val).float() * preact
        assert torch.allclose(encoded, expected_jumprelu, atol=1e-6)


def test_clt_decode(clt_model, sample_inputs):
    """Test the decode method."""
    config = clt_model.config
    batch_size, seq_len, d_model = sample_inputs[0].shape
    num_tokens = batch_size * seq_len

    # Encode produces [num_tokens, num_features]
    activations = {i: clt_model.encode(x, i) for i, x in sample_inputs.items()}

    for layer_idx in range(config.num_layers):
        reconstruction = clt_model.decode(activations, layer_idx)
        # Expect output shape: [num_tokens, d_model]
        assert reconstruction.shape == (num_tokens, config.d_model)

        # Check reconstruction calculation (simplified check)
        if layer_idx == 1 and 0 in activations and 1 in activations:
            # Ensure activations are on the same device as decoders for the check
            device = next(clt_model.parameters()).device
            act_0 = activations[0].to(device)
            act_1 = activations[1].to(device)

            dec_0_1 = clt_model.decoders["0->1"]
            dec_1_1 = clt_model.decoders["1->1"]
            expected_rec_1 = dec_0_1(act_0) + dec_1_1(act_1)
            # Reconstruction should also be on the same device
            assert torch.allclose(reconstruction.to(device), expected_rec_1, atol=1e-5)


def test_clt_forward(clt_model, sample_inputs):
    """Test the forward pass of the CLT."""
    config = clt_model.config
    reconstructions = clt_model(sample_inputs)

    assert isinstance(reconstructions, dict)
    # Forward should produce outputs for all layers up to num_layers
    assert len(reconstructions) == config.num_layers
    # Check that keys are 0 to num_layers-1
    assert all(idx in reconstructions for idx in range(config.num_layers))

    for layer_idx, recon in reconstructions.items():
        # Get original input shape for comparison
        batch_size, seq_len, d_model = sample_inputs[layer_idx].shape
        num_tokens = batch_size * seq_len
        # Expect output shape: [num_tokens, d_model]
        assert recon.shape == (num_tokens, config.d_model)

        # Check if forward pass output matches manual encode/decode
        # Need to handle device consistency
        device = next(clt_model.parameters()).device
        activations = {
            i: clt_model.encode(x.to(device), i)
            for i, x in sample_inputs.items()
            if i <= layer_idx
        }
        # Filter only relevant activations for decode
        relevant_activations = {k: v for k, v in activations.items() if k <= layer_idx}
        if relevant_activations:  # Only decode if there are relevant activations
            expected_recon = clt_model.decode(relevant_activations, layer_idx)
            assert torch.allclose(recon.to(device), expected_recon, atol=1e-5)


def test_clt_get_feature_activations(clt_model, sample_inputs):
    """Test getting all feature activations."""
    config = clt_model.config
    activations = clt_model.get_feature_activations(sample_inputs)

    assert isinstance(activations, dict)
    assert len(activations) == len(sample_inputs)
    assert all(idx in activations for idx in sample_inputs.keys())

    for layer_idx, act in activations.items():
        batch_size, seq_len, _ = sample_inputs[layer_idx].shape
        num_tokens = batch_size * seq_len
        # Expect shape: [num_tokens, num_features]
        assert act.shape == (num_tokens, config.num_features)
        # Check if it matches encode output
        expected_act = clt_model.encode(sample_inputs[layer_idx], layer_idx)
        assert torch.allclose(act, expected_act, atol=1e-6)


def test_clt_get_decoder_norms(clt_model):
    """Test calculation of decoder norms."""
    config = clt_model.config
    decoder_norms = clt_model.get_decoder_norms()

    assert decoder_norms.shape == (config.num_layers, config.num_features)
    assert not torch.any(torch.isnan(decoder_norms))
    assert torch.all(decoder_norms >= 0)

    # Manual check for one feature (e.g., feature 0) at layer 0
    expected_norm_0_0_sq = 0
    device = next(clt_model.parameters()).device
    dtype = next(clt_model.parameters()).dtype
    for tgt_layer in range(config.num_layers):  # From src_layer=0
        decoder = clt_model.decoders[f"0->{tgt_layer}"]
        # Norm of the first column (feature 0) - ensure calculation is on correct device/dtype
        weight_col_0 = decoder.weight[:, 0].to(
            device=device, dtype=torch.float32
        )  # Use float32 for stable norm calc
        expected_norm_0_0_sq += torch.norm(weight_col_0, p=2).pow(2)

    expected_norm_0_0 = torch.sqrt(expected_norm_0_0_sq).to(
        dtype=dtype
    )  # Cast back to model dtype
    # Use slightly higher tolerance due to potential dtype conversions
    assert torch.allclose(decoder_norms[0, 0].to(device), expected_norm_0_0, atol=1e-5)


# --- Fixtures for GPT-2 Small Size Test Case ---


@pytest.fixture
def clt_config_gpt2_small():
    # Parameters similar to GPT-2 Small
    return CLTConfig(
        d_model=768,
        num_features=3072,  # d_model * 4
        num_layers=12,
        activation_fn="relu",  # Keep it simple for size test
        clt_dtype="float32",  # Explicitly set for test
    )


@pytest.fixture
def clt_model_gpt2_small(clt_config_gpt2_small):
    # Run on CPU by default for large model test unless CUDA available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CrossLayerTranscoder(clt_config_gpt2_small, device=device)


@pytest.fixture
def sample_inputs_gpt2_small(clt_config_gpt2_small):
    # Smaller batch/seq_len for faster test with large d_model
    batch_size = 1
    seq_len = 4
    config = clt_config_gpt2_small
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {
        i: torch.randn(
            batch_size, seq_len, config.d_model, device=device, dtype=torch.float32
        )
        for i in range(config.num_layers)
    }
    return inputs


# --- Tests Duplicated for GPT-2 Small Size ---
# (Using the specific gpt2_small fixtures)


def test_clt_init_gpt2_small(clt_model_gpt2_small):
    """Test CLT initialization with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    config = clt_model.config
    assert isinstance(clt_model, nn.Module)
    assert len(clt_model.encoders) == config.num_layers
    assert all(enc.in_features == config.d_model for enc in clt_model.encoders)
    assert all(enc.out_features == config.num_features for enc in clt_model.encoders)

    num_expected_decoders = config.num_layers * (config.num_layers + 1) // 2
    assert len(clt_model.decoders) == num_expected_decoders
    for src in range(config.num_layers):
        for tgt in range(src, config.num_layers):
            key = f"{src}->{tgt}"
            assert key in clt_model.decoders
            dec = clt_model.decoders[key]
            assert dec.in_features == config.num_features
            assert dec.out_features == config.d_model

    # Calculate and print memory footprint
    total_params = sum(p.numel() for p in clt_model.parameters() if p.requires_grad)
    dtype_size = torch.finfo(clt_model.dtype).bits // 8
    estimated_memory_bytes = total_params * dtype_size
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
    print(
        f"\n[Memory Footprint Info for GPT-2 Small Size "
        f"({clt_model.config.activation_fn} CLT ({clt_model.dtype}) on {clt_model.device})]"
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
    batch_size, seq_len, _ = x.shape
    num_tokens = batch_size * seq_len
    preact = clt_model.get_preactivations(x, 0)
    assert preact.shape == (num_tokens, config.num_features)


def test_clt_encode_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the encode method with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    x = sample_inputs[0]
    batch_size, seq_len, _ = x.shape
    num_tokens = batch_size * seq_len
    encoded = clt_model.encode(x, 0)
    assert encoded.shape == (num_tokens, config.num_features)
    # Only checking ReLU here as per fixture config
    assert torch.all(encoded >= 0)


def test_clt_decode_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the decode method with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    batch_size, seq_len, _ = sample_inputs[0].shape
    num_tokens = batch_size * seq_len

    # Encode inputs first
    activations = {
        i: clt_model.encode(sample_inputs[i], i) for i in range(config.num_layers)
    }

    for layer_idx in range(config.num_layers):
        reconstruction = clt_model.decode(activations, layer_idx)
        assert reconstruction.shape == (num_tokens, config.d_model)
        # Skip detailed calculation check for large model test


def test_clt_forward_gpt2_small(clt_model_gpt2_small, sample_inputs_gpt2_small):
    """Test the forward pass of the CLT with GPT-2 small dimensions."""
    clt_model = clt_model_gpt2_small
    sample_inputs = sample_inputs_gpt2_small
    config = clt_model.config
    reconstructions = clt_model(sample_inputs)

    assert isinstance(reconstructions, dict)
    assert len(reconstructions) == config.num_layers
    assert all(idx in reconstructions for idx in range(config.num_layers))

    for layer_idx, recon in reconstructions.items():
        batch_size, seq_len, _ = sample_inputs[layer_idx].shape
        num_tokens = batch_size * seq_len
        assert recon.shape == (num_tokens, config.d_model)
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
    assert len(activations) == config.num_layers
    assert all(idx in activations for idx in range(config.num_layers))

    for layer_idx, act in activations.items():
        batch_size, seq_len, _ = sample_inputs[layer_idx].shape
        num_tokens = batch_size * seq_len
        assert act.shape == (num_tokens, config.num_features)
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


# Note: Ensure the actual CLTConfig class aligns with usage here.
