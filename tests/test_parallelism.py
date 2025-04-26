import torch
import torch.distributed as dist
import pytest
import os
import math
from typing import Optional
import torch.nn.functional as F  # Add F for padding
import torch.nn as nn  # Add nn for Linear layer

# from clt.config import TrainingConfig # Assuming these are needed later -> remove for now
from clt.config import CLTConfig, TrainingConfig  # Add TrainingConfig back
from clt.models.clt import CrossLayerTranscoder
from clt.models.parallel import ColumnParallelLinear, RowParallelLinear  # Import both
from clt.models.parallel import _gather  # Import _gather specifically

# from clt.models.parallel import ColumnParallelLinear, RowParallelLinear # Unused for now
from clt.training.losses import LossManager  # Import LossManager

# Environment variable to simulate world size if not run with torchrun
# Example: `TP_WORLD_SIZE=2 pytest tests/test_parallelism.py`
# Default to 1 if not set (single GPU/CPU mode)
DEFAULT_WORLD_SIZE = 1
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", DEFAULT_WORLD_SIZE))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


# --- Distributed Setup Helper ---
def setup_distributed(backend="nccl" if torch.cuda.is_available() else "gloo"):
    """Initializes the distributed environment."""
    if WORLD_SIZE > 1 and not dist.is_initialized():
        # These env vars are typically set by torchrun
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            # Find a free port dynamically? For now, use a default.
            os.environ["MASTER_PORT"] = "29501"  # Adjust if needed

        print(f"Initializing DDP: Rank {RANK}/{WORLD_SIZE} on local rank {LOCAL_RANK} backend {backend}...")
        # Determine device_id based on backend
        device_id_to_pass = None
        if backend == "nccl" and torch.cuda.is_available():
            device_id_to_pass = torch.device(f"cuda:{LOCAL_RANK}")

        dist.init_process_group(
            backend=backend,
            rank=RANK,
            world_size=WORLD_SIZE,
            # Pass the determined device object or None
            device_id=device_id_to_pass,  # Pass device object or None
        )
        print("DDP Initialized.")
        if torch.cuda.is_available():
            torch.cuda.set_device(LOCAL_RANK)  # Crucial for multi-GPU per node

    # Barrier to ensure all processes are initialized
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module", autouse=True)
def distributed_fixture():
    """Pytest fixture to setup/teardown distributed environment."""
    setup_distributed()
    yield
    cleanup_distributed()


# --- Test Configuration ---
@pytest.fixture(scope="module")
def base_config() -> CLTConfig:
    """Base CLT configuration for tests."""
    return CLTConfig(
        d_model=32,  # Small hidden size
        num_features=64,  # Number of CLT features (keep power of 2 for TP)
        num_layers=4,  # Number of layers
        activation_fn="jumprelu",
        jumprelu_threshold=0.01,
        clt_dtype="float32",  # Use float32 for easier comparison
    )


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Determine the device based on availability and distributed setup."""
    if WORLD_SIZE > 1 and torch.cuda.is_available():
        return torch.device(f"cuda:{LOCAL_RANK}")
    elif torch.cuda.is_available():
        # Single GPU test case, use cuda:0
        # Note: Ensure tests run on a machine with at least one GPU if CUDA is intended
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="module")
def training_config() -> TrainingConfig:
    """Base Training configuration for loss tests."""
    return TrainingConfig(
        # Required basic params
        learning_rate=1e-4,
        training_steps=100,
        train_batch_size_tokens=1024,  # From default
        # Activation source (placeholders, but required)
        activation_source="local_manifest",
        activation_path="dummy",
        # Loss specific params (using correct names from definition)
        sparsity_lambda=0.01,
        sparsity_c=1.0,
        preactivation_coef=0.0,  # Disable preactivation loss for simplicity
        # Optional params with defaults used in LossManager or Trainer
        log_interval=10,
        eval_interval=20,
        checkpoint_interval=50,
        optimizer="adamw",
        lr_scheduler=None,
        seed=42,
        activation_dtype="float32",
        # Normalization - set to none to avoid dependency on stats files/estimation
        normalization_method="none",
        # Other defaults that might be relevant
        n_batches_in_buffer=16,
        dead_feature_window=1000,
        enable_wandb=False,
    )


@pytest.fixture(scope="module")
def loss_manager(training_config: TrainingConfig) -> LossManager:
    """Create a LossManager instance."""
    return LossManager(training_config)


# --- Helper to Scatter Full Parameter to Shards ---
def scatter_full_parameter(full_param: torch.Tensor, model_param: torch.nn.Parameter, partition_dim: int):
    """Splits a full parameter tensor and loads the correct shard onto the current rank."""
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        # Single GPU: model_param should already hold the full_param data (or be assigned)
        # This might require ensuring the single_gpu_model fixture runs first if we copy this way.
        # Let's assume direct assignment works if needed, but focus on multi-GPU case.
        if model_param.shape == full_param.shape:
            model_param.data.copy_(full_param)
        else:
            # This case shouldn't happen if WORLD_SIZE <=1
            print(f"Warning: Mismatched shapes in scatter for single GPU: {model_param.shape} vs {full_param.shape}")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    full_dim_size = full_param.size(partition_dim)
    local_dim_padded = math.ceil(full_dim_size / world_size)

    # Calculate start/end index for this rank's shard in the full tensor
    start_index = rank * local_dim_padded
    # Use the actual model parameter's dimension size for the end index calculation
    # This accounts for the padding applied during model creation
    local_dim_actual = model_param.size(partition_dim)
    end_index = start_index + local_dim_actual  # Use actual shard size

    # Ensure end_index doesn't exceed the original full dimension
    end_index = min(end_index, full_dim_size)
    actual_local_dim_size = max(0, end_index - start_index)

    if actual_local_dim_size > 0:
        # Create slice objects
        indices = [slice(None)] * full_param.dim()
        indices[partition_dim] = slice(start_index, end_index)

        # Extract the shard from the full parameter
        param_shard = full_param[tuple(indices)].clone()

        # --- Padding (if necessary) ---
        # If the model parameter shard is larger due to padding during creation,
        # we need to pad the shard extracted from the single_gpu_model before copying.
        pad_amount = model_param.size(partition_dim) - param_shard.size(partition_dim)
        if pad_amount > 0:
            pad_dims = [0, 0] * model_param.dim()
            # Determine padding side based on partition_dim relative to tensor dims
            # Example: For 2D weight [out, in] partitioned along dim 0 (column), pad on the right of dim 0.
            # Example: For 2D weight [out, in] partitioned along dim 1 (row), pad on the right of dim 1.
            # F.pad takes pads in reverse order of dimensions: (pad_left_dimN, pad_right_dimN, pad_left_dimN-1, ...)
            pad_idx = model_param.dim() - 1 - partition_dim
            pad_dims[2 * pad_idx + 1] = pad_amount  # Pad right for the partition dimension
            param_shard = F.pad(param_shard, tuple(pad_dims))
        # --- End Padding ---

        # Copy the (potentially padded) shard data to the model parameter on the current rank
        if model_param.shape == param_shard.shape:
            model_param.data.copy_(param_shard.to(model_param.device, model_param.dtype))
        else:
            # This indicates an error in slicing or padding logic
            print(
                f"Rank {rank} scatter ERROR: Shape mismatch {model_param.shape} != {param_shard.shape} for dim {partition_dim}"
            )
            # Fallback: maybe zero pad? Or fail? Let's print and maybe fail test later.


# --- Model Fixtures ---
@pytest.fixture(scope="module")
def single_gpu_model(base_config: CLTConfig, device: torch.device) -> CrossLayerTranscoder:
    """Create a standard, non-distributed CLT model.

    Ensures parameters are identical across all ranks if running in distributed mode
    by broadcasting from Rank 0.
    """
    # Ensure consistent initialization state across ranks *before* creating the model
    # (Optional but good practice, broadcasting below is the main sync point)
    torch.manual_seed(42)  # Use a fixed global seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Create the model instance
    # Pass process_group=None for single GPU/CPU
    model = CrossLayerTranscoder(base_config, process_group=None, device=device)
    model.eval()  # Set to eval mode

    # --- Broadcast from Rank 0 to ensure identical reference model --- #
    if WORLD_SIZE > 1 and dist.is_initialized():
        print(f"Rank {RANK}: Broadcasting single_gpu_model parameters from Rank 0...")
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        dist.barrier()  # Ensure broadcast is complete
        print(f"Rank {RANK}: single_gpu_model broadcast complete.")
    # --- End Broadcast --- #

    return model


@pytest.fixture(scope="module")
def multi_gpu_model(
    single_gpu_model: CrossLayerTranscoder,  # Add single_gpu_model as dependency
    base_config: CLTConfig,
    device: torch.device,
) -> Optional[CrossLayerTranscoder]:
    """Create a distributed CLT model if WORLD_SIZE > 1, copying weights from single_gpu_model."""
    if WORLD_SIZE <= 1:
        pytest.skip("Skipping multi-GPU model creation (WORLD_SIZE <= 1)")
        return None

    if not dist.is_initialized():
        pytest.fail("Distributed environment not initialized for multi-GPU model fixture.")

    # Create the multi-GPU model (initializes its own parameters/shards)
    model = CrossLayerTranscoder(base_config, process_group=dist.group.WORLD, device=device)
    model.eval()

    # --- Copy weights from single_gpu_model to multi_gpu_model shards ---
    print(f"Rank {RANK}: Copying weights from single GPU model to multi-GPU model shards...")
    single_params_dict = dict(single_gpu_model.named_parameters())
    with torch.no_grad():
        for name, multi_param in model.named_parameters():
            if name not in single_params_dict:
                print(f"Rank {RANK}: Warning - Parameter {name} not found in single_gpu_model.")
                continue

            single_param = single_params_dict[name]
            single_param_data = single_param.data.to(device)  # Ensure data is on correct device

            # Identify parameter type and scatter/copy accordingly
            if name == "log_threshold":  # Replicated parameter
                multi_param.data.copy_(single_param_data)
            elif "encoders." in name and ".weight" in name:
                # Encoder weights are ColumnParallelLinear (partition_dim=0)
                scatter_full_parameter(single_param_data, multi_param, partition_dim=0)
            elif "decoders." in name and ".weight" in name:
                # Decoder weights are RowParallelLinear (partition_dim=1)
                scatter_full_parameter(single_param_data, multi_param, partition_dim=1)
            elif "encoders." in name and ".bias" in name:
                # Encoder bias is ColumnParallelLinear (partition_dim=0)
                scatter_full_parameter(single_param_data, multi_param, partition_dim=0)
            elif "decoders." in name and ".bias" in name:
                # Decoder bias is RowParallelLinear (replicated, not sharded)
                multi_param.data.copy_(single_param_data)
            else:
                print(f"Rank {RANK}: Warning - Unhandled parameter type for weight copying: {name}")
                # Attempt direct copy for any other unexpected params
                if multi_param.shape == single_param.shape:
                    multi_param.data.copy_(single_param_data)

    # Barrier to ensure copying is complete on all ranks
    if dist.is_initialized():
        dist.barrier()

    # Add extra barrier just in case
    if dist.is_initialized():
        dist.barrier()

    print(f"Rank {RANK}: Multi-GPU model created and weights copied from single GPU model.")
    return model


# --- Test Data ---
@pytest.fixture(scope="module")
def identical_input_data(base_config: CLTConfig, device: torch.device) -> torch.Tensor:
    """Create identical input data across all ranks."""
    # Use a fixed seed for reproducibility IF testing initialization is sensitive.
    # For forward pass comparison, just ensuring identical data is key.
    # torch.manual_seed(42) # Optional: If needed
    batch_size = 4
    seq_len = 16  # Or just batch_tokens
    input_tensor = torch.randn(batch_size * seq_len, base_config.d_model, device=device, dtype=torch.float32)

    # Ensure all ranks have the exact same tensor if distributed
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.broadcast(input_tensor, src=0)
        dist.barrier()  # Ensure broadcast is complete

    return input_tensor


# --- Test Cases ---


# Test 1: Encoder Pre-activations (get_preactivations)
def test_encoder_preactivations(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    base_config: CLTConfig,
    device: torch.device,
):
    """Compare encoder pre-activation outputs between single and multi-GPU."""

    # --- Single GPU Execution --- # Moved multi-GPU model check earlier
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return  # Should not be reached

    # --- Check parameters and run forward layer by layer --- #
    single_params_dict = dict(single_gpu_model.named_parameters())
    multi_params_dict = dict(multi_gpu_model.named_parameters())

    for layer_idx in range(base_config.num_layers):
        print(f"\n--- Testing Layer {layer_idx} --- ")
        # 1. Verify Parameters for this layer just before use
        single_encoder_weight_name = f"encoders.{layer_idx}.weight"
        multi_encoder_weight_name = f"encoders.{layer_idx}.weight"

        if RANK == 0:
            print(f"Rank {RANK}: Verifying parameters for layer {layer_idx}...")
            # Get full params from single model
            single_weight = single_params_dict[single_encoder_weight_name].data.to(device)
            # Get corresponding shard from multi model
            multi_weight_shard = multi_params_dict[multi_encoder_weight_name].data.to(device)

            # Calculate the expected shard shape based on single model param
            out_features = single_weight.shape[0]
            local_out_features_padded = math.ceil(out_features / WORLD_SIZE)
            expected_weight_shard = single_weight[:local_out_features_padded, :]

            # Check shapes match
            assert multi_weight_shard.shape == expected_weight_shard.shape, f"Weight shape mismatch layer {layer_idx}"

            # Check content matches (using torch.equal for exact match after copy)
            assert torch.equal(multi_weight_shard, expected_weight_shard), f"Weight content mismatch layer {layer_idx}"
            print(f"Rank {RANK}: Parameters for layer {layer_idx} verified.")
        # Barrier to ensure rank 0 finishes verification before others proceed (optional)
        if dist.is_initialized():
            dist.barrier()

        # --- MANUAL CALCULATION TEST for Layer 0 --- #
        if layer_idx == 0:
            print(f"Rank {RANK}: Performing manual calculation test for layer {layer_idx}...")
            input_clone = identical_input_data.clone()
            # Get parameters
            single_weight_full = single_params_dict[single_encoder_weight_name].data.to(device)
            multi_weight_shard = multi_params_dict[multi_encoder_weight_name].data.to(device)

            # --- Re-verify parameters right before manual use --- #
            if RANK == 0:
                print(f"Rank {RANK}: RE-VERIFYING parameters for layer {layer_idx} INSIDE manual calc...")
                out_features = single_weight_full.shape[0]
                local_out_features_padded = math.ceil(out_features / WORLD_SIZE)
                expected_weight_shard = single_weight_full[:local_out_features_padded, :]
                assert (
                    multi_weight_shard.shape == expected_weight_shard.shape
                ), f"RE-VERIFY Weight shape mismatch layer {layer_idx}"
                assert torch.equal(
                    multi_weight_shard, expected_weight_shard
                ), f"RE-VERIFY Weight content mismatch layer {layer_idx}"
                print(f"Rank {RANK}: RE-VERIFICATION PASSED.")
            if dist.is_initialized():
                dist.barrier()
            # --- End Re-verify ---

            # Single GPU manual calculation
            # Bias is False for encoders
            single_out_manual = F.linear(input_clone, single_weight_full)

            # Multi GPU manual calculation (local matmul + gather)
            # Bias is False for encoders
            multi_out_local_manual = F.linear(input_clone, multi_weight_shard)
            multi_out_manual_gathered = _gather(
                multi_out_local_manual.contiguous(), dist.group.WORLD, dim=-1, full_dim_size=base_config.num_features
            )

            # Compare on Rank 0
            if RANK == 0:
                print("Comparing MANUAL Calculation Outputs (Rank 0):")
                print(
                    f"  Manual Single shape={single_out_manual.shape}, Manual Multi shape={multi_out_manual_gathered.shape}"
                )
                assert torch.allclose(
                    single_out_manual, multi_out_manual_gathered, atol=1e-5, rtol=1e-4
                ), f"MANUAL CALCULATION Mismatch for layer {layer_idx}. Max diff: {(single_out_manual - multi_out_manual_gathered).abs().max()}"
                print("MANUAL CALCULATION check PASSED.")
            if dist.is_initialized():
                dist.barrier()
        # --- END MANUAL CALCULATION TEST --- #

        # 2. Run Forward Passes for this layer
        with torch.no_grad():
            single_gpu_output = single_gpu_model.get_preactivations(identical_input_data.clone(), layer_idx)
            multi_gpu_output = multi_gpu_model.get_preactivations(identical_input_data.clone(), layer_idx)

            # Sanity checks (as before)
            expected_shape = (identical_input_data.shape[0], base_config.num_features)
            assert single_gpu_output.shape == expected_shape
            assert multi_gpu_output.shape == expected_shape
            assert single_gpu_output.device == device
            # Note: multi_gpu_output should be on rank's specific device
            assert multi_gpu_output.device == device  # device fixture already resolves to rank's device

        # 3. Compare Outputs (Rank 0)
        if RANK == 0:
            print(f"Comparing Encoder Pre-activations for Layer {layer_idx} (Rank 0):")
            single_out = single_gpu_output.to(multi_gpu_output.device)
            multi_out = multi_gpu_output

            print(f"  Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            assert torch.allclose(single_out, multi_out, atol=1e-6), (
                f"Mismatch in pre-activations for layer {layer_idx} between single and multi-GPU."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )
            print(f"  Layer {layer_idx} Pre-activation Check PASSED.")
        # Barrier before next layer (ensures prints are ordered)
        if dist.is_initialized():
            dist.barrier()


# Test 2: Feature Activations (encode)
def test_feature_activations(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    base_config: CLTConfig,
    device: torch.device,
):
    """Compare feature activation outputs (after nonlinearity) between single and multi-GPU."""

    # --- Single GPU Execution ---
    with torch.no_grad():
        single_gpu_outputs = {}
        for layer_idx in range(base_config.num_layers):
            single_gpu_outputs[layer_idx] = single_gpu_model.encode(identical_input_data.clone(), layer_idx)
            # Sanity check output shape
            expected_shape = (identical_input_data.shape[0], base_config.num_features)
            assert (
                single_gpu_outputs[layer_idx].shape == expected_shape
            ), f"Single GPU encode layer {layer_idx} output shape mismatch: {single_gpu_outputs[layer_idx].shape} != {expected_shape}"
            assert (
                single_gpu_outputs[layer_idx].device == device
            ), f"Single GPU encode layer {layer_idx} output device mismatch: {single_gpu_outputs[layer_idx].device} != {device}"

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    multi_gpu_outputs = {}
    with torch.no_grad():
        for layer_idx in range(base_config.num_layers):
            # encode() applies nonlinearity to the full preactivation tensor
            # returned by get_preactivations (which uses ColumnParallelLinear).
            # The nonlinearity uses the replicated log_threshold.
            # Result should be the full activation tensor on all ranks.
            multi_gpu_output = multi_gpu_model.encode(identical_input_data.clone(), layer_idx)
            multi_gpu_outputs[layer_idx] = multi_gpu_output

            # Sanity check output shape (should be full shape on all ranks)
            expected_shape = (identical_input_data.shape[0], base_config.num_features)
            assert (
                multi_gpu_output.shape == expected_shape
            ), f"Multi GPU Rank {RANK} encode layer {layer_idx} output shape mismatch: {multi_gpu_output.shape} != {expected_shape}"
            assert (
                multi_gpu_output.device == device
            ), f"Multi GPU Rank {RANK} encode layer {layer_idx} output device mismatch: {multi_gpu_output.device} != {device}"

    # --- Comparison (only on Rank 0) ---
    if RANK == 0:
        print("\nComparing Feature Activations (Rank 0):")
        for layer_idx in range(base_config.num_layers):
            single_out = single_gpu_outputs[layer_idx]
            multi_out = multi_gpu_outputs[layer_idx]
            single_out = single_out.to(multi_out.device)

            print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            assert torch.allclose(single_out, multi_out, atol=1e-6), (
                f"Mismatch in feature activations for layer {layer_idx} between single and multi-GPU."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )


# Test 3: Decoder Forward Pass (decode)
def test_decoder_decode(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    base_config: CLTConfig,
    device: torch.device,
):
    """Compare decoder reconstruction outputs (decode method) between single and multi-GPU."""

    # --- Generate Feature Activations (using single GPU model for simplicity) ---
    # We need the full activations for the decode input dictionary.
    # Using the single_gpu_model ensures we have a consistent starting point.
    feature_activations = {}
    with torch.no_grad():
        for layer_idx in range(base_config.num_layers):
            # Use encode, which gives full activations after nonlinearity
            feature_activations[layer_idx] = single_gpu_model.encode(identical_input_data.clone(), layer_idx)
            # Ensure they are on the test device
            feature_activations[layer_idx] = feature_activations[layer_idx].to(device)

    # --- Single GPU Execution ---
    single_gpu_reconstructions = {}
    with torch.no_grad():
        for layer_idx in range(base_config.num_layers):
            # Create input dict with activations up to the current layer
            current_activations = {k: v.clone() for k, v in feature_activations.items() if k <= layer_idx}
            single_gpu_reconstructions[layer_idx] = single_gpu_model.decode(current_activations, layer_idx)
            # Sanity check output shape
            expected_shape = (identical_input_data.shape[0], base_config.d_model)
            assert (
                single_gpu_reconstructions[layer_idx].shape == expected_shape
            ), f"Single GPU decode layer {layer_idx} output shape mismatch: {single_gpu_reconstructions[layer_idx].shape} != {expected_shape}"
            assert (
                single_gpu_reconstructions[layer_idx].device == device
            ), f"Single GPU decode layer {layer_idx} output device mismatch: {single_gpu_reconstructions[layer_idx].device} != {device}"

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    multi_gpu_reconstructions = {}
    with torch.no_grad():
        for layer_idx in range(base_config.num_layers):
            # Prepare the same input dict for the multi-GPU model
            # The activations are full tensors, RowParallelLinear splits them internally
            current_activations_multi = {k: v.clone() for k, v in feature_activations.items() if k <= layer_idx}

            # decode() uses RowParallelLinear -> should return the *full* tensor
            # after internal split, matmul, and all_reduce.
            multi_gpu_output = multi_gpu_model.decode(current_activations_multi, layer_idx)
            multi_gpu_reconstructions[layer_idx] = multi_gpu_output

            # Sanity check output shape (should be full shape on all ranks)
            expected_shape = (identical_input_data.shape[0], base_config.d_model)
            assert (
                multi_gpu_output.shape == expected_shape
            ), f"Multi GPU Rank {RANK} decode layer {layer_idx} output shape mismatch: {multi_gpu_output.shape} != {expected_shape}"
            assert (
                multi_gpu_output.device == device
            ), f"Multi GPU Rank {RANK} decode layer {layer_idx} output device mismatch: {multi_gpu_output.device} != {device}"

    # --- Comparison (only on Rank 0) ---
    if RANK == 0:
        print("\nComparing Decoder Outputs (Rank 0):")
        for layer_idx in range(base_config.num_layers):
            single_out = single_gpu_reconstructions[layer_idx]
            multi_out = multi_gpu_reconstructions[layer_idx]
            single_out = single_out.to(multi_out.device)

            print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            # Check if shapes match before comparison
            if single_out.shape != multi_out.shape:
                pytest.fail(f"Shape mismatch for layer {layer_idx}: Single={single_out.shape}, Multi={multi_out.shape}")

            # Use torch.allclose for floating point comparison
            # Increase tolerance slightly for decode due to potential floating point differences in all_reduce sum
            assert torch.allclose(single_out, multi_out, atol=1e-5, rtol=1e-4), (
                f"Mismatch in decoder outputs for layer {layer_idx} between single and multi-GPU."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )


# Test 4: Full Forward Pass (forward)
def test_full_forward_pass(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    base_config: CLTConfig,
    device: torch.device,
):
    """Compare the full forward pass outputs between single and multi-GPU."""

    # --- Prepare Input Dictionary ---
    # The forward method expects a dictionary mapping layer index to input tensor.
    # For this test, we'll use the same identical_input_data for all layers.
    input_dict = {}
    for layer_idx in range(base_config.num_layers):
        input_dict[layer_idx] = identical_input_data.clone()

    # --- Single GPU Execution ---
    single_gpu_outputs = {}
    with torch.no_grad():
        single_gpu_outputs = single_gpu_model(input_dict)
        # Sanity check output shapes and devices
        for layer_idx, output in single_gpu_outputs.items():
            expected_shape = (identical_input_data.shape[0], base_config.d_model)
            assert (
                output.shape == expected_shape
            ), f"Single GPU forward layer {layer_idx} output shape mismatch: {output.shape} != {expected_shape}"
            assert (
                output.device == device
            ), f"Single GPU forward layer {layer_idx} output device mismatch: {output.device} != {device}"

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    multi_gpu_outputs = {}
    with torch.no_grad():
        # Input dict needs cloning for each model if modified internally (shouldn't be)
        input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
        multi_gpu_outputs = multi_gpu_model(input_dict_multi)

        # Sanity check output shapes and devices (should be full shape on all ranks)
        for layer_idx, output in multi_gpu_outputs.items():
            expected_shape = (identical_input_data.shape[0], base_config.d_model)
            assert (
                output.shape == expected_shape
            ), f"Multi GPU Rank {RANK} forward layer {layer_idx} output shape mismatch: {output.shape} != {expected_shape}"
            assert (
                output.device == device
            ), f"Multi GPU Rank {RANK} forward layer {layer_idx} output device mismatch: {output.device} != {device}"

    # --- Comparison (only on Rank 0) ---
    if RANK == 0:
        print("\nComparing Full Forward Pass Outputs (Rank 0):")
        assert (
            single_gpu_outputs.keys() == multi_gpu_outputs.keys()
        ), f"Output dictionary keys differ: {single_gpu_outputs.keys()} vs {multi_gpu_outputs.keys()}"

        for layer_idx in single_gpu_outputs.keys():
            single_out = single_gpu_outputs[layer_idx]
            multi_out = multi_gpu_outputs[layer_idx]
            single_out = single_out.to(multi_out.device)

            print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            # Check shapes
            if single_out.shape != multi_out.shape:
                pytest.fail(
                    f"Shape mismatch forward layer {layer_idx}: Single={single_out.shape}, Multi={multi_out.shape}"
                )

            # Use torch.allclose (using slightly increased tolerance from decode)
            assert torch.allclose(single_out, multi_out, atol=1e-5, rtol=1e-4), (
                f"Mismatch in full forward pass outputs for layer {layer_idx} between single and multi-GPU."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )


# Test 5: Reconstruction Loss Calculation
def test_reconstruction_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    base_config: CLTConfig,
    device: torch.device,
):
    """Compare the reconstruction loss component between single and multi-GPU."""

    # --- Prepare Inputs & Targets ---
    # Use the same input data for all layers as source
    input_dict = {}
    # Create dummy target tensors (e.g., slightly modified inputs)
    target_dict = {}
    for layer_idx in range(base_config.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        # Create targets with the same shape as model outputs (d_model)
        target_tensor = (
            torch.randn_like(input_tensor[:, : base_config.d_model]) * 0.5 + input_tensor[:, : base_config.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    # --- Single GPU Execution ---
    single_gpu_outputs = single_gpu_model(input_dict)
    # Targets need cloning if modified by loss function (shouldn't be)
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    # LossManager calculates loss based on model outputs and targets
    single_recon_loss = loss_manager.compute_reconstruction_loss(single_gpu_outputs, target_dict_single)
    # Check types
    assert isinstance(single_recon_loss, torch.Tensor) and single_recon_loss.numel() == 1
    print(f"\nSingle GPU Recon Loss: {single_recon_loss.item():.6f}")

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    # Input dict needs cloning for each model if modified internally
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    multi_gpu_outputs = multi_gpu_model(input_dict_multi)
    # Targets need cloning
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    # Loss is computed locally on each rank using the full outputs/targets
    multi_recon_loss = loss_manager.compute_reconstruction_loss(multi_gpu_outputs, target_dict_multi)
    # Check types
    assert isinstance(multi_recon_loss, torch.Tensor) and multi_recon_loss.numel() == 1
    print(f"Multi GPU Rank {RANK} Recon Loss: {multi_recon_loss.item():.6f}")

    # --- Comparison (only on Rank 0) ---
    # Reconstruction loss should be identical on all ranks as inputs/targets/model outputs are identical
    if RANK == 0:
        print("\nComparing Reconstruction Loss (Rank 0):")
        single_loss_val = single_recon_loss.item()
        multi_loss_val = multi_recon_loss.item()

        # Compare scalar loss values
        assert math.isclose(
            single_loss_val, multi_loss_val, rel_tol=1e-5, abs_tol=1e-6
        ), f"Mismatch in reconstruction loss scalar value between single ({single_loss_val}) and multi-GPU ({multi_loss_val})."


# Test 6: Sparsity Loss Calculation (via total loss)
def test_sparsity_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    base_config: CLTConfig,
    training_config: TrainingConfig,  # Need training config for total_steps
    device: torch.device,
):
    """Compare the sparsity loss component (extracted from total loss) between single and multi-GPU."""

    # --- Prepare Inputs & Targets ---
    # Reusing setup from reconstruction loss test
    input_dict = {}
    target_dict = {}
    for layer_idx in range(base_config.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : base_config.d_model]) * 0.5 + input_tensor[:, : base_config.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    # --- Single GPU Execution ---
    # Calculate total loss
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    _, single_loss_dict = loss_manager.compute_total_loss(
        single_gpu_model,
        input_dict,  # Use original input dict
        target_dict_single,
        current_step=0,  # Use step 0 for simplicity
        total_steps=training_config.training_steps,  # Use total steps from config
    )
    # Extract sparsity loss
    single_sparsity_loss_val = single_loss_dict.get("sparsity", 0.0)
    print(f"\nSingle GPU Sparsity Loss (from total): {single_sparsity_loss_val:.6f}")

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    # Calculate total loss
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    _, multi_loss_dict = loss_manager.compute_total_loss(
        multi_gpu_model, input_dict_multi, target_dict_multi, current_step=0, total_steps=training_config.training_steps
    )
    # Extract sparsity loss
    multi_sparsity_loss_val = multi_loss_dict.get("sparsity", 0.0)
    print(f"Multi GPU Rank {RANK} Sparsity Loss (from total): {multi_sparsity_loss_val:.6f}")

    # --- Comparison (only on Rank 0) ---
    # Sparsity loss might have slight differences due to all_reduce in get_decoder_norms
    if RANK == 0:
        print("\nComparing Sparsity Loss (Rank 0):")
        # Compare scalar loss values (allow slightly larger tolerance)
        assert math.isclose(
            single_sparsity_loss_val, multi_sparsity_loss_val, rel_tol=1e-4, abs_tol=1e-5
        ), f"Mismatch in sparsity loss scalar value between single ({single_sparsity_loss_val}) and multi-GPU ({multi_sparsity_loss_val})."


# Test 7: Total Loss Calculation
def test_total_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    base_config: CLTConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Compare the total loss value between single and multi-GPU."""

    # --- Prepare Inputs & Targets ---
    input_dict = {}
    target_dict = {}
    for layer_idx in range(base_config.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : base_config.d_model]) * 0.5 + input_tensor[:, : base_config.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    # --- Single GPU Execution ---
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    single_total_loss, _ = loss_manager.compute_total_loss(
        single_gpu_model, input_dict, target_dict_single, current_step=0, total_steps=training_config.training_steps
    )
    single_total_loss_val = single_total_loss.item()
    print(f"\nSingle GPU Total Loss: {single_total_loss_val:.6f}")

    # --- Multi GPU Execution ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    multi_total_loss, _ = loss_manager.compute_total_loss(
        multi_gpu_model, input_dict_multi, target_dict_multi, current_step=0, total_steps=training_config.training_steps
    )
    multi_total_loss_val = multi_total_loss.item()
    print(f"Multi GPU Rank {RANK} Total Loss: {multi_total_loss_val:.6f}")

    # --- Comparison (only on Rank 0) ---
    # Total loss combines reconstruction and sparsity, use tolerance from sparsity
    if RANK == 0:
        print("\nComparing Total Loss (Rank 0):")
        assert math.isclose(
            single_total_loss_val, multi_total_loss_val, rel_tol=1e-4, abs_tol=1e-5
        ), f"Mismatch in total loss scalar value between single ({single_total_loss_val}) and multi-GPU ({multi_total_loss_val})."


# --- Helper for Gradient Averaging (mirrors trainer logic) ---
def average_replicated_grads(model: CrossLayerTranscoder):
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    # Identify replicated parameters (currently just log_threshold)
    # Assumes bias terms in parallel layers are handled by TP logic (added after reduce/before gather)
    # If other parameters were replicated, add them here.
    replicated_params = [model.log_threshold]

    for p in replicated_params:
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            p.grad.data /= world_size


# --- Helper to Gather Sharded Gradients ---
def gather_sharded_gradient(
    local_grad: torch.Tensor, model_param: torch.nn.Parameter, full_param_shape: tuple, partition_dim: int
) -> Optional[torch.Tensor]:
    """Gathers gradient slices from all ranks for a sharded parameter."""
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        # If single GPU, local_grad is the full gradient
        return local_grad

    # Ensure consistent device and dtype
    gathered_grads = [torch.empty_like(local_grad) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_grads, local_grad, group=dist.group.WORLD)

    # Only rank 0 needs to reconstruct the full gradient
    if RANK == 0:
        try:
            full_grad = torch.cat(gathered_grads, dim=partition_dim)
            # Truncate if necessary due to padding during sharding
            if full_grad.shape != full_param_shape:
                indices = [slice(None)] * full_grad.dim()
                for dim_idx, (full_dim_size, grad_dim_size) in enumerate(zip(full_param_shape, full_grad.shape)):
                    if grad_dim_size > full_dim_size:
                        indices[dim_idx] = slice(0, full_dim_size)
                full_grad = full_grad[tuple(indices)]

            # Final shape check
            if full_grad.shape != full_param_shape:
                print(
                    f"Warning: Reconstructed gradient shape {full_grad.shape} != expected {full_param_shape} for param {model_param.shape}"
                )
                return None  # Indicate failure
            return full_grad
        except Exception as e:
            print(f"Error reconstructing gradient for param {model_param.shape}: {e}")
            return None
    else:
        return None  # Other ranks don't need the full grad


# Test 8: Gradient Calculation
def test_gradient_calculation(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    base_config: CLTConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Compare gradients between single and multi-GPU after backward pass."""

    # --- Prepare Inputs & Targets ---
    input_dict = {}
    target_dict = {}
    for layer_idx in range(base_config.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : base_config.d_model]) * 0.5 + input_tensor[:, : base_config.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    # --- Single GPU Backward Pass ---
    # Ensure model is in train mode for gradients
    single_gpu_model.train()
    single_gpu_model.zero_grad()
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    single_total_loss, _ = loss_manager.compute_total_loss(
        single_gpu_model, input_dict, target_dict_single, current_step=0, total_steps=training_config.training_steps
    )
    single_total_loss.backward()
    single_gpu_grads = {name: p.grad.clone() for name, p in single_gpu_model.named_parameters() if p.grad is not None}
    single_gpu_model.eval()  # Back to eval mode

    # --- Multi GPU Backward Pass ---
    if multi_gpu_model is None:
        pytest.skip("Multi-GPU model not available.")
        return

    multi_gpu_model.train()
    multi_gpu_model.zero_grad()
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    multi_total_loss, _ = loss_manager.compute_total_loss(
        multi_gpu_model, input_dict_multi, target_dict_multi, current_step=0, total_steps=training_config.training_steps
    )
    # Simulate trainer's backward pass:
    # 1. Backward
    multi_total_loss.backward()
    # 2. Average replicated grads (like in trainer)
    average_replicated_grads(multi_gpu_model)
    # 3. Barrier (optional but good practice)
    if dist.is_initialized():
        dist.barrier()

    multi_gpu_local_grads = {name: p.grad for name, p in multi_gpu_model.named_parameters() if p.grad is not None}
    multi_gpu_model.eval()  # Back to eval mode
    print(f"Multi GPU Rank {RANK} Backward Pass Completed. Found {len(multi_gpu_local_grads)} local grads.")

    # --- Barrier before comparison loop --- #
    if dist.is_initialized():
        dist.barrier()

    # --- Comparison Loop (All Ranks Participate in Gathering) --- #
    # Moved loop outside Rank 0 check
    # Ensure same parameters have gradients (check on rank 0 only)
    if RANK == 0:
        print("\nComparing Gradients (Rank 0):")
        assert (
            single_gpu_grads.keys() == multi_gpu_local_grads.keys()
        ), f"Gradient keys differ: {single_gpu_grads.keys()} vs {multi_gpu_local_grads.keys()}"

    multi_gpu_params_dict = dict(multi_gpu_model.named_parameters())

    # --------- NEW: Track mismatches to avoid premature assertion that breaks barriers ---------
    gradient_mismatch_detected = False
    mismatch_messages = []  # Collect detailed messages for debugging on Rank 0
    # -------------------------------------------------------------------------------------------

    for name, single_grad in single_gpu_grads.items():
        if name not in multi_gpu_local_grads:
            # This case should be caught by the key check on Rank 0, but good practice
            if RANK == 0:
                msg = f"Warning: Grad for {name} missing in multi-GPU model."
                print(msg)
                gradient_mismatch_detected = True
                mismatch_messages.append(msg)
            continue

        local_grad = multi_gpu_local_grads[name]
        param = multi_gpu_params_dict[name]

        # --- Perform Gathering on ALL ranks if sharded --- #
        full_multi_grad = None  # Initialize for Rank != 0
        is_sharded = False
        partition_dim = -1

        if "encoders." in name and ".weight" in name:
            is_sharded = True
            partition_dim = 0
        elif "decoders." in name and ".weight" in name:
            is_sharded = True
            partition_dim = 1
        # Add other sharded types here (e.g., encoder bias if ColumnParallel)
        elif "encoders." in name and ".bias" in name:  # Check if bias is sharded (depends on ColumnParallel impl)
            # Encoders have bias=False, this branch should not be hit
            if hasattr(multi_gpu_model.encoders[int(name.split(".")[1])], "bias_param"):
                is_sharded = True
                partition_dim = 0  # ColumnParallel bias is sharded along output features

        if is_sharded:
            # All ranks participate in the gather
            # gather_sharded_gradient returns the full tensor only on Rank 0
            full_multi_grad = gather_sharded_gradient(local_grad, param, single_grad.shape, partition_dim)

        # --- Comparison (only on Rank 0) --- #
        if RANK == 0:
            print(f"  Comparing grad for: {name} (shape {single_grad.shape})")
            single_grad = single_grad.to(local_grad.device)

            try:
                # Identify parameter type (replicated or sharded)
                if name == "log_threshold":  # Replicated parameter
                    print("    Type: Replicated")
                    if not torch.allclose(single_grad, local_grad, atol=1e-4, rtol=1e-4):
                        gradient_mismatch_detected = True
                        mismatch_messages.append(
                            f"Mismatch in replicated gradient for '{name}'. Max diff: {(single_grad - local_grad).abs().max()}"
                        )

                elif is_sharded:
                    # Access the gathered gradient computed above
                    print(f"    Type: Sharded (partition_dim={partition_dim})")
                    if full_multi_grad is not None:
                        # Increase tolerance for gathered sharded grads
                        assert torch.allclose(
                            single_grad, full_multi_grad, atol=1e-4, rtol=1e-3
                        ), f"Mismatch in gathered sharded gradient for '{name}'. Max diff: {(single_grad - full_multi_grad).abs().max()}"
                    else:
                        pytest.fail(f"Failed to gather gradient for sharded parameter {name} on Rank 0")

                # Handle non-sharded, non-log_threshold parameters (e.g., RowParallelLinear bias)
                elif "decoders." in name and ".bias" in name:
                    # Bias for RowParallelLinear should be replicated
                    print("    Type: Replicated (Decoder Bias)")
                    if not torch.allclose(single_grad, local_grad, atol=1e-5, rtol=1e-4):
                        gradient_mismatch_detected = True
                        mismatch_messages.append(f"Mismatch in replicated gradient for '{name}'.")
                else:
                    # Unknown/unhandled parameter type
                    gradient_mismatch_detected = True
                    mismatch_messages.append(f"Unhandled parameter type for gradient check: {name}")
            except Exception as exc:
                # Catch unexpected errors to ensure barrier below is still reached
                gradient_mismatch_detected = True
                mismatch_messages.append(f"Exception during gradient comparison for {name}: {exc}")

    # --- Final Barrier --- #
    # Ensure all ranks wait until Rank 0 finishes comparisons before proceeding
    if dist.is_initialized():
        dist.barrier()

    # After synchronization, raise once on Rank 0 if any mismatches were detected
    if RANK == 0 and gradient_mismatch_detected:
        pytest.fail("\n".join(mismatch_messages))

    # -------------------------------------------------------------------------------------------


# --- Isolated Layer Tests ---


def test_column_parallel_linear_forward(
    device: torch.device,
):
    """Test ColumnParallelLinear forward pass against nn.Linear."""
    if WORLD_SIZE <= 1:
        pytest.skip("Skipping ColumnParallelLinear test (WORLD_SIZE <= 1)")

    in_features = 32
    out_features = 64  # Must be divisible by WORLD_SIZE for simpler testing?
    # Using 64 / 2 = 32 works.
    batch_tokens = 128
    seed = 42 + RANK  # Use different seed per rank initially

    # Ensure layers are created on the correct device for the rank
    test_device = device
    print(f"Rank {RANK}: Running CPL test on device: {test_device}")

    # 1. Create standard nn.Linear layer (on rank 0, then broadcast)
    torch.manual_seed(seed)  # Seed for consistency if needed
    single_layer = nn.Linear(in_features, out_features, bias=True).to(test_device)
    # Broadcast Rank 0's weights/bias to ensure all ranks start comparison from same base
    if dist.is_initialized():
        dist.broadcast(single_layer.weight.data, src=0)
        dist.broadcast(single_layer.bias.data, src=0)
        dist.barrier()
    print(f"Rank {RANK}: Single nn.Linear layer created and weights broadcasted.")

    # 2. Create ColumnParallelLinear layer
    torch.manual_seed(seed)  # Re-seed *before* creating multi_layer if its init needs to differ
    multi_layer = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        process_group=dist.group.WORLD,
        device=test_device,
    )
    multi_layer.eval()
    print(f"Rank {RANK}: ColumnParallelLinear created.")

    # 3. Copy weights/bias from single_layer to multi_layer shards
    print(f"Rank {RANK}: Scattering weights/bias to ColumnParallelLinear...")
    with torch.no_grad():
        scatter_full_parameter(single_layer.weight.data, multi_layer.weight, partition_dim=0)
        scatter_full_parameter(single_layer.bias.data, multi_layer.bias_param, partition_dim=0)
    if dist.is_initialized():
        dist.barrier()
    print(f"Rank {RANK}: Scatter complete.")

    # --- Optional: Verify scattered weights/bias --- #
    # (Add prints here to compare slices if needed)
    if RANK == 0:
        print(f"Rank {RANK}: Checking scattered weights/bias (first few elements):")
        # Calculate expected shard for rank 0
        local_out_features = math.ceil(out_features / WORLD_SIZE)
        expected_weight_shard = single_layer.weight.data[:local_out_features, :]
        expected_bias_shard = single_layer.bias.data[:local_out_features]
        print(f"  Rank 0: multi.weight shape {multi_layer.weight.shape}, expected {expected_weight_shard.shape}")
        print(f"  Rank 0: multi.bias shape {multi_layer.bias_param.shape}, expected {expected_bias_shard.shape}")
        if not torch.equal(multi_layer.weight.data, expected_weight_shard):
            print(
                f"  Rank 0: WARNING - Scattered weight mismatch! Max diff: {(multi_layer.weight.data - expected_weight_shard).abs().max()}"
            )
        if not torch.equal(multi_layer.bias_param.data, expected_bias_shard):
            print(
                f"  Rank 0: WARNING - Scattered bias mismatch! Max diff: {(multi_layer.bias_param.data - expected_bias_shard).abs().max()}"
            )
    # --- End Verify --- #

    # 4. Create identical input data
    torch.manual_seed(42)  # Fixed seed for input data
    input_data = torch.randn(batch_tokens, in_features, device=test_device)
    if dist.is_initialized():
        dist.broadcast(input_data, src=0)
        dist.barrier()
    print(f"Rank {RANK}: Input data created and broadcasted.")

    # 5. Run forward passes
    with torch.no_grad():
        single_output = single_layer(input_data.clone())
        multi_output = multi_layer(input_data.clone())  # Should perform gather internally

    # 6. Compare outputs on Rank 0
    if RANK == 0:
        print("\nComparing ColumnParallelLinear outputs (Rank 0):")
        print(f"  Single output shape: {single_output.shape}")
        print(f"  Multi output shape: {multi_output.shape}")
        assert single_output.shape == multi_output.shape, "Output shapes mismatch"
        assert torch.allclose(
            single_output, multi_output, atol=1e-5, rtol=1e-4
        ), f"Mismatch between nn.Linear and ColumnParallelLinear output.\nMax diff: {(single_output - multi_output).abs().max()}"
        print("  ColumnParallelLinear output matches nn.Linear output.")


def test_row_parallel_linear_forward(
    base_config: CLTConfig,  # Need base_config for init args
    device: torch.device,
):
    """Test RowParallelLinear forward pass against nn.Linear."""
    if WORLD_SIZE <= 1:
        pytest.skip("Skipping RowParallelLinear test (WORLD_SIZE <= 1)")

    # Note: For RowParallel, input features are sharded.
    # Output features remain the full dimension.
    in_features = 64  # Feature dim (sharded)
    out_features = 32  # Model dim (full)
    batch_tokens = 128
    seed = 42 + RANK

    test_device = device
    print(f"Rank {RANK}: Running RPL test on device: {test_device}")

    # 1. Create standard nn.Linear layer (broadcast weights)
    torch.manual_seed(seed)
    single_layer = nn.Linear(in_features, out_features, bias=True).to(test_device)
    if dist.is_initialized():
        dist.broadcast(single_layer.weight.data, src=0)
        dist.broadcast(single_layer.bias.data, src=0)
        dist.barrier()
    print(f"Rank {RANK}: Single nn.Linear layer created and weights broadcasted.")

    # 2. Create RowParallelLinear layer
    torch.manual_seed(seed)
    multi_layer = RowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        process_group=dist.group.WORLD,
        input_is_parallel=False,  # Mimic usage in decode - layer handles split
        d_model_for_init=base_config.d_model,  # Need d_model for init bounds
        num_layers_for_init=base_config.num_layers,  # Need num_layers for init bounds
        device=test_device,
    )
    multi_layer.eval()
    print(f"Rank {RANK}: RowParallelLinear created.")

    # 3. Copy weights/bias from single_layer to multi_layer
    print(f"Rank {RANK}: Scattering/copying weights/bias to RowParallelLinear...")
    with torch.no_grad():
        # Scatter weight (partition_dim=1 for RowParallel)
        scatter_full_parameter(single_layer.weight.data, multi_layer.weight, partition_dim=1)
        # Copy bias directly (it's replicated, not sharded in RowParallelLinear)
        if multi_layer.bias_param is not None:
            multi_layer.bias_param.data.copy_(single_layer.bias.data.to(test_device))
    if dist.is_initialized():
        dist.barrier()
    print(f"Rank {RANK}: Scatter/copy complete.")

    # 4. Create identical *full* input data (input_is_parallel=False)
    torch.manual_seed(42)
    input_data = torch.randn(batch_tokens, in_features, device=test_device)
    if dist.is_initialized():
        dist.broadcast(input_data, src=0)
        dist.barrier()
    print(f"Rank {RANK}: Input data created and broadcasted.")

    # 5. Run forward passes
    with torch.no_grad():
        single_output = single_layer(input_data.clone())
        # RowParallel handles split and reduce internally
        multi_output = multi_layer(input_data.clone())

    # 6. Compare outputs on Rank 0
    if RANK == 0:
        print("\nComparing RowParallelLinear outputs (Rank 0):")
        print(f"  Single output shape: {single_output.shape}")
        print(f"  Multi output shape: {multi_output.shape}")
        assert single_output.shape == multi_output.shape, "Output shapes mismatch"
        # Increase tolerance slightly due to _reduce (SUM) potentially causing more diffs
        assert torch.allclose(
            single_output, multi_output, atol=1e-5, rtol=1e-4
        ), f"Mismatch between nn.Linear and RowParallelLinear output.\nMax diff: {(single_output - multi_output).abs().max()}"
        print("  RowParallelLinear output matches nn.Linear output.")


# Add more tests here for:
# - RowParallelLinear
