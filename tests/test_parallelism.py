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


# --- Test Configurations --- #
@pytest.fixture(scope="function")  # Changed scope to function for parametrization
def clt_config_fn(request):
    """Parameterized CLT configuration factory for tests."""
    params = request.param if hasattr(request, "param") else {}
    activation_fn = params.get("activation_fn", "jumprelu")
    num_features = params.get("num_features", 64)
    d_model = params.get("d_model", 32)
    num_layers = params.get("num_layers", 4)
    batchtopk_k = params.get("batchtopk_k", None)
    topk_k = params.get("topk_k", None)

    return CLTConfig(
        d_model=d_model,
        num_features=num_features,
        num_layers=num_layers,
        activation_fn=activation_fn,
        jumprelu_threshold=0.01,  # Keep a default
        batchtopk_k=batchtopk_k,
        topk_k=topk_k,
        clt_dtype="float32",
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


# --- Helper to Scatter Full Parameter to Shards --- #
def scatter_full_parameter(full_param: torch.Tensor, model_param: torch.nn.Parameter, partition_dim: int):
    """Splits a full parameter tensor and loads the correct shard onto the current rank."""
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        if model_param.shape == full_param.shape:
            model_param.data.copy_(full_param.to(model_param.device, model_param.dtype))
        else:
            if RANK == 0:  # Only print warning on rank 0 to avoid clutter
                print(
                    f"Warning: Mismatched shapes in scatter for single GPU/non-dist: {model_param.shape} vs {full_param.shape} for param name (not available)"
                )
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    full_dim_size = full_param.size(partition_dim)
    local_dim_padded = math.ceil(full_dim_size / world_size)

    start_index = rank * local_dim_padded
    local_dim_actual = model_param.size(partition_dim)
    end_index = start_index + local_dim_actual
    end_index = min(end_index, full_dim_size)
    actual_local_dim_size = max(0, end_index - start_index)

    if actual_local_dim_size > 0:
        indices = [slice(None)] * full_param.dim()
        indices[partition_dim] = slice(start_index, end_index)
        param_shard = full_param[tuple(indices)].clone()

        pad_amount = model_param.size(partition_dim) - param_shard.size(partition_dim)
        if pad_amount > 0:
            pad_dims = [0, 0] * model_param.dim()
            # F.pad takes pads in reverse order of dimensions: (pad_left_dimN, pad_right_dimN, pad_left_dimN-1, ...)
            pad_idx = model_param.dim() - 1 - partition_dim
            pad_dims[2 * pad_idx + 1] = pad_amount  # Pad right for the partition dimension
            param_shard = F.pad(param_shard, tuple(pad_dims))

        if model_param.shape == param_shard.shape:
            model_param.data.copy_(param_shard.to(model_param.device, model_param.dtype))
        else:
            if RANK == 0:
                print(
                    f"Rank {rank} scatter ERROR: Shape mismatch {model_param.shape} != {param_shard.shape} for dim {partition_dim}"
                )
    elif model_param.numel() > 0:
        if RANK == 0:
            print(
                f"Rank {rank} scatter INFO: actual_local_dim_size is 0 for partition_dim {partition_dim}, param shape {model_param.shape}. Full dim size {full_dim_size}. No data copied from full_param."
            )


# --- Helper for Gradient Averaging (mirrors trainer logic) ---
def average_replicated_grads(model: CrossLayerTranscoder):
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            is_rep = getattr(p, "_is_replicated", False)
            if is_rep or (p.dim() == 1 and not hasattr(p, "_is_replicated")):
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                p.grad.data /= world_size


# --- Helper to Gather Sharded Gradients ---
def gather_sharded_gradient(
    local_grad: torch.Tensor, model_param: torch.nn.Parameter, full_param_shape: tuple, partition_dim: int
) -> Optional[torch.Tensor]:
    """Gathers gradient slices from all ranks for a sharded parameter."""
    if WORLD_SIZE <= 1 or not dist.is_initialized():
        return local_grad.clone()

    gathered_grads_list = [torch.empty_like(local_grad, device=local_grad.device) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_grads_list, local_grad.contiguous(), group=dist.group.WORLD)

    if RANK == 0:
        try:
            device = gathered_grads_list[0].device
            gathered_grads_on_device = [g.to(device) for g in gathered_grads_list]
            full_grad = torch.cat(gathered_grads_on_device, dim=partition_dim)

            if full_grad.size(partition_dim) > full_param_shape[partition_dim]:
                slicing_indices = [slice(None)] * full_grad.dim()
                slicing_indices[partition_dim] = slice(0, full_param_shape[partition_dim])
                full_grad = full_grad[tuple(slicing_indices)]

            if full_grad.shape != full_param_shape:
                print(
                    f"Warning: Reconstructed gradient shape {full_grad.shape} != expected {full_param_shape} for param (model_param shape {model_param.shape}). Attempting to truncate all dims."
                )
                indices = [slice(None)] * full_grad.dim()
                valid_slice = True
                for dim_idx, (expected_dim_size, actual_dim_size) in enumerate(zip(full_param_shape, full_grad.shape)):
                    if actual_dim_size > expected_dim_size:
                        indices[dim_idx] = slice(0, expected_dim_size)
                    elif actual_dim_size < expected_dim_size:
                        valid_slice = False
                        break
                if valid_slice:
                    full_grad = full_grad[tuple(indices)]
                    if full_grad.shape != full_param_shape:
                        return None
                else:
                    return None
            return full_grad.contiguous()
        except Exception as e:
            print(f"Error reconstructing gradient for param (shape {model_param.shape}): {e}")
            return None
    else:
        return None


# --- Model Fixtures (now function-scoped due to config) --- #
@pytest.fixture(scope="function")
def single_gpu_model(clt_config_fn: CLTConfig, device: torch.device) -> CrossLayerTranscoder:
    """Create a standard, non-distributed CLT model.
    Ensures parameters are identical across all ranks if running in distributed mode
    by broadcasting from Rank 0.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = CrossLayerTranscoder(clt_config_fn, process_group=None, device=device)
    model.eval()

    # Removed per-parameter dist.broadcast to avoid potential collective-order
    # divergence between ranks. The main model parameters are deterministically
    # initialised with a fixed random seed on every rank, so they are generally
    # identical.
    # However, log_threshold (if used for JumpReLU) needs to be explicitly broadcast
    # as different GPU initializations, even with the same seed, can lead to
    # minor discrepancies that affect JumpReLU gating if not perfectly synced.
    if WORLD_SIZE > 1 and dist.is_initialized():
        if clt_config_fn.activation_fn == "jumprelu":
            # Ensure log_threshold exists and is a parameter before broadcasting
            if (
                hasattr(model, "log_threshold")
                and model.log_threshold is not None
                and isinstance(model.log_threshold, nn.Parameter)
            ):
                dist.broadcast(model.log_threshold.data, src=0)
            else:
                # This case should ideally not be hit if activation_fn is jumprelu
                # and the model is correctly initialized. Log if it occurs.
                if RANK == 0:  # RANK is a global constant in the file
                    print(
                        f"Rank {RANK}: Warning - single_gpu_model configured for jumprelu but log_threshold not found or not a Parameter. Skipping broadcast of log_threshold."
                    )
        dist.barrier()  # Barrier to ensure all ranks are synced after potential broadcast.
    return model


@pytest.fixture(scope="function")
def multi_gpu_model(
    single_gpu_model: CrossLayerTranscoder, clt_config_fn: CLTConfig, device: torch.device
) -> Optional[CrossLayerTranscoder]:
    """Create a distributed CLT model if WORLD_SIZE > 1, copying weights from single_gpu_model."""
    if WORLD_SIZE <= 1:
        return None

    if not dist.is_initialized():
        pytest.fail("Distributed environment not initialized for multi-GPU model fixture.")

    # Create the multi-GPU model (initializes its own parameters/shards)
    # Use a different seed for multi_gpu_model creation if its init is random and needs to differ from single_gpu_model before copy
    # However, since we copy all params, initial state of multi_gpu_model's weights doesn't strictly matter.
    model = CrossLayerTranscoder(clt_config_fn, process_group=dist.group.WORLD, device=device)
    model.eval()

    # Ensure single_gpu_model creation and its broadcasts are complete on all ranks
    if dist.is_initialized():
        dist.barrier()

    single_params_dict = dict(single_gpu_model.named_parameters())
    with torch.no_grad():
        for name, multi_param in model.named_parameters():
            if name not in single_params_dict:
                if RANK == 0:
                    print(
                        f"Rank {RANK}: Warning - Parameter {name} from multi_gpu_model not found in single_gpu_model. Skipping copy."
                    )
                continue

            single_param = single_params_dict[name]
            single_param_data = single_param.data.to(device)

            if name == "log_threshold":
                # Only copy if log_threshold exists on multi_param (e.g. if clt_config_fn.activation_fn is 'jumprelu')
                # and shapes match. It might not exist if single_gpu_model was batchtopk and converted,
                # while multi_gpu_model is being initialized fresh with a jumprelu config.
                # This fixture logic assumes clt_config_fn is the SAME for both models being compared in a test run.
                if (
                    hasattr(model, "log_threshold")
                    and model.log_threshold is not None
                    and multi_param is model.log_threshold
                ):
                    if multi_param.shape == single_param_data.shape:
                        multi_param.data.copy_(single_param_data)
                        if RANK == 0 and name == "log_threshold":
                            # Verify the copy for log_threshold on RANK 0
                            # Using a small slice to avoid excessive printing
                            s_slice = single_param_data.flatten()[: min(5, single_param_data.numel())]
                            m_slice = multi_param.data.flatten()[: min(5, multi_param.data.numel())]
                            print(
                                f"Rank {RANK} DEBUG: log_threshold copied. Single_model (src): {s_slice}. Multi_model (dst after copy): {m_slice}. Equal: {torch.equal(single_param_data, multi_param.data)}"
                            )
                    else:
                        if RANK == 0:
                            print(
                                f"Rank {RANK}: Warning - log_threshold shape mismatch: multi {multi_param.shape} vs single {single_param_data.shape}"
                            )
                # If single_gpu_model has log_threshold but multi_gpu_model doesn't (e.g. different configs), this is a test setup issue.

            elif "encoders." in name and ".weight" in name:
                scatter_full_parameter(single_param_data, multi_param, partition_dim=0)
            elif "decoders." in name and ".weight" in name:
                scatter_full_parameter(single_param_data, multi_param, partition_dim=1)
            elif "encoders." in name and ".bias_param" in name:
                # Check if the corresponding encoder layer in multi_gpu_model was configured with bias=True
                layer_idx_str = name.split(".")[1]
                if model.encoders[int(layer_idx_str)].bias:  # Accessing the boolean flag
                    scatter_full_parameter(single_param_data, multi_param, partition_dim=0)
            elif "decoders." in name and ".bias_param" in name:
                # Check if the corresponding decoder layer in multi_gpu_model was configured with bias=True
                decoder_key = name.split(".")[1]  # "src->tgt"
                if model.decoders[decoder_key].bias:  # Accessing the boolean flag
                    # Decoder bias is replicated, not sharded
                    multi_param.data.copy_(single_param_data)
            else:
                # Catch-all for any other parameters (e.g. future additions, or if naming changes)
                if RANK == 0:
                    print(
                        f"Rank {RANK}: Warning - Unhandled parameter type for weight copying: {name}. Attempting direct copy if shapes match."
                    )
                if multi_param.shape == single_param_data.shape:
                    multi_param.data.copy_(single_param_data)
                else:
                    if RANK == 0:
                        print(
                            f"Rank {RANK}: Error - Shape mismatch for unhandled param {name}: multi {multi_param.shape} vs single {single_param_data.shape}. NOT copying."
                        )

    if dist.is_initialized():
        dist.barrier()
    return model


# --- Test Data ---
@pytest.fixture(scope="function")  # Changed to function due to config changes
def identical_input_data(clt_config_fn: CLTConfig, device: torch.device) -> torch.Tensor:
    """Create identical input data across all ranks."""
    # Use a fixed seed for reproducibility IF testing initialization is sensitive.
    # For forward pass comparison, just ensuring identical data is key.
    # torch.manual_seed(42) # Optional: If needed
    batch_size = 4
    seq_len = 16  # Or just batch_tokens
    input_tensor = torch.randn(batch_size * seq_len, clt_config_fn.d_model, device=device, dtype=torch.float32)

    # Ensure all ranks have the exact same tensor if distributed
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.broadcast(input_tensor, src=0)
        dist.barrier()  # Ensure broadcast is complete

    return input_tensor


# Test 1: Encoder Pre-activations (get_preactivations)
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        {"activation_fn": "topk", "topk_k": 0.2},
    ],
    indirect=True,
)
def test_encoder_preactivations(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    clt_config_fn: CLTConfig,  # use the resolved config
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

    for layer_idx in range(clt_config_fn.num_layers):
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
                multi_out_local_manual.contiguous(), dist.group.WORLD, dim=-1, full_dim_size=clt_config_fn.num_features
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
            if multi_gpu_model:
                multi_gpu_output = multi_gpu_model.get_preactivations(identical_input_data.clone(), layer_idx)
            else:  # Single GPU test run
                multi_gpu_output = single_gpu_model.get_preactivations(identical_input_data.clone(), layer_idx)

            # Sanity checks (as before)
            expected_shape = (identical_input_data.shape[0], clt_config_fn.num_features)
            assert single_gpu_output.shape == expected_shape
            assert multi_gpu_output.shape == expected_shape
            assert single_gpu_output.device == device
            # Note: multi_gpu_output should be on rank's specific device
            assert multi_gpu_output.device == device  # device fixture already resolves to rank's device

        # 3. Compare Outputs (Rank 0)
        # Barrier before next layer (ensures prints are ordered)
        try:
            if RANK == 0:
                print(f"Comparing Encoder Pre-activations for Layer {layer_idx} (Rank 0):")
                single_out = single_gpu_output.to(multi_gpu_output.device)
                multi_out = multi_gpu_output

                print(f"  Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
                # Specific debug for jumprelu, now targeting layer 2 or 3 based on typical failure points
                if clt_config_fn.activation_fn == "jumprelu" and layer_idx in [2, 3]:
                    mismatch_indices = torch.where(torch.abs(single_out - multi_out) > 1e-6)
                    num_mismatches = mismatch_indices[0].numel()
                    print(f"    DEBUG jumprelu L{layer_idx}: Num mismatches by allclose logic: {num_mismatches}")
                    if num_mismatches > 0 and num_mismatches < 20:  # Print details if few mismatches
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Mismatch indices (first one): {tuple(idx[0].item() for idx in mismatch_indices)}"
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Single out at mismatch: {single_out[mismatch_indices[0][0], mismatch_indices[1][0]]}"
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Multi  out at mismatch: {multi_out[mismatch_indices[0][0], mismatch_indices[1][0]]}"
                        )
                        # Check if one is zero and other is not (typical JumpReLU diff)
                        single_is_zero = torch.isclose(
                            single_out[mismatch_indices[0][0], mismatch_indices[1][0]],
                            torch.tensor(0.0, device=single_out.device),
                            atol=1e-7,
                        )
                        multi_is_zero = torch.isclose(
                            multi_out[mismatch_indices[0][0], mismatch_indices[1][0]],
                            torch.tensor(0.0, device=multi_out.device),
                            atol=1e-7,
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Single is zero: {single_is_zero}, Multi is zero: {multi_is_zero}"
                        )

                assert torch.allclose(single_out, multi_out, atol=1e-6), (
                    f"Mismatch in pre-activations for layer {layer_idx} between single and multi-GPU."
                    f"\nMax diff: {(single_out - multi_out).abs().max()}"
                )
                print(f"  Layer {layer_idx} Pre-activation Check PASSED.")
        finally:
            if (
                dist.is_initialized()
            ):  # Removed WORLD_SIZE > 1 check as dist.is_initialized() implies it for barrier usage
                dist.barrier()


# Test 2: Feature Activations (encode)
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "relu"},  # Add relu for completeness here
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        {"activation_fn": "topk", "topk_k": 0.2},
    ],
    indirect=True,
)
def test_feature_activations(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    clt_config_fn: CLTConfig,
    device: torch.device,
):
    """Compare feature activation outputs (after nonlinearity) between single and multi-GPU."""

    # --- Single GPU Execution --- #
    with torch.no_grad():
        single_gpu_outputs = {}
        for layer_idx in range(clt_config_fn.num_layers):
            single_gpu_outputs[layer_idx] = single_gpu_model.encode(identical_input_data.clone(), layer_idx)
            expected_shape = (identical_input_data.shape[0], clt_config_fn.num_features)
            assert single_gpu_outputs[layer_idx].shape == expected_shape
            assert single_gpu_outputs[layer_idx].device == device

    # --- Multi GPU Execution --- #
    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model

    multi_gpu_outputs = {}
    with torch.no_grad():
        for layer_idx in range(clt_config_fn.num_layers):
            multi_gpu_output = multi_gpu_actual_model.encode(identical_input_data.clone(), layer_idx)
            multi_gpu_outputs[layer_idx] = multi_gpu_output
            expected_shape = (identical_input_data.shape[0], clt_config_fn.num_features)
            assert multi_gpu_output.shape == expected_shape
            assert multi_gpu_output.device == device

    # --- Comparison (only on Rank 0) --- #
    try:
        if RANK == 0:
            print(f"\nComparing Feature Activations (Rank 0) for config: {clt_config_fn.activation_fn}")
            for layer_idx in range(clt_config_fn.num_layers):
                single_out = single_gpu_outputs[layer_idx]
                multi_out = multi_gpu_outputs[layer_idx]
                # Ensure multi_out is on the same device as single_out for comparison if it was from single_gpu_model fallback
                multi_out = multi_out.to(single_out.device)

                print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
                # Specific debug for jumprelu, now targeting layer 2 or 3 based on typical failure points
                if clt_config_fn.activation_fn == "jumprelu" and layer_idx in [2, 3]:
                    mismatch_indices = torch.where(torch.abs(single_out - multi_out) > 1e-6)
                    num_mismatches = mismatch_indices[0].numel()
                    print(f"    DEBUG jumprelu L{layer_idx}: Num mismatches by allclose logic: {num_mismatches}")
                    if num_mismatches > 0 and num_mismatches < 20:  # Print details if few mismatches
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Mismatch indices (first one): {tuple(idx[0].item() for idx in mismatch_indices)}"
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Single out at mismatch: {single_out[mismatch_indices[0][0], mismatch_indices[1][0]]}"
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Multi  out at mismatch: {multi_out[mismatch_indices[0][0], mismatch_indices[1][0]]}"
                        )
                        # Check if one is zero and other is not (typical JumpReLU diff)
                        single_is_zero = torch.isclose(
                            single_out[mismatch_indices[0][0], mismatch_indices[1][0]],
                            torch.tensor(0.0, device=single_out.device),
                            atol=1e-7,
                        )
                        multi_is_zero = torch.isclose(
                            multi_out[mismatch_indices[0][0], mismatch_indices[1][0]],
                            torch.tensor(0.0, device=multi_out.device),
                            atol=1e-7,
                        )
                        print(
                            f"    DEBUG jumprelu L{layer_idx}: Single is zero: {single_is_zero}, Multi is zero: {multi_is_zero}"
                        )

                assert torch.allclose(single_out, multi_out, atol=1e-6), (
                    f"Mismatch in feature activations for layer {layer_idx} (config: {clt_config_fn.activation_fn})."
                    f"\nMax diff: {(single_out - multi_out).abs().max()}"
                )
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 3: Decoder Forward Pass (decode)
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        {"activation_fn": "topk", "topk_k": 0.2},
    ],
    indirect=True,
)
def test_decoder_decode(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    clt_config_fn: CLTConfig,
    device: torch.device,
):
    """Compare decoder reconstruction outputs (decode method) between single and multi-GPU."""

    feature_activations = {}
    with torch.no_grad():
        for layer_idx in range(clt_config_fn.num_layers):
            feature_activations[layer_idx] = single_gpu_model.encode(identical_input_data.clone(), layer_idx)
            feature_activations[layer_idx] = feature_activations[layer_idx].to(device)

    single_gpu_reconstructions = {}
    with torch.no_grad():
        for layer_idx in range(clt_config_fn.num_layers):
            current_activations = {k: v.clone() for k, v in feature_activations.items() if k <= layer_idx}
            single_gpu_reconstructions[layer_idx] = single_gpu_model.decode(current_activations, layer_idx)
            expected_shape = (identical_input_data.shape[0], clt_config_fn.d_model)
            assert single_gpu_reconstructions[layer_idx].shape == expected_shape
            assert single_gpu_reconstructions[layer_idx].device == device

    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model
    multi_gpu_reconstructions = {}
    with torch.no_grad():
        for layer_idx in range(clt_config_fn.num_layers):
            current_activations_multi = {k: v.clone() for k, v in feature_activations.items() if k <= layer_idx}
            multi_gpu_output = multi_gpu_actual_model.decode(current_activations_multi, layer_idx)
            multi_gpu_reconstructions[layer_idx] = multi_gpu_output
            expected_shape = (identical_input_data.shape[0], clt_config_fn.d_model)
            assert multi_gpu_output.shape == expected_shape
            assert multi_gpu_output.device == device

    # --- Comparison (only on Rank 0) --- #
    try:
        if RANK == 0:
            print(f"\nComparing Decoder Outputs (Rank 0) for config: {clt_config_fn.activation_fn}")
            for layer_idx in range(clt_config_fn.num_layers):
                single_out = single_gpu_reconstructions[layer_idx]
                multi_out = multi_gpu_reconstructions[layer_idx]
                multi_out = multi_out.to(single_out.device)

                print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
                if single_out.shape != multi_out.shape:
                    pytest.fail(
                        f"Shape mismatch for layer {layer_idx}: Single={single_out.shape}, Multi={multi_out.shape}"
                    )
                assert torch.allclose(single_out, multi_out, atol=1e-5, rtol=1e-4), (
                    f"Mismatch in decoder outputs for layer {layer_idx} (config: {clt_config_fn.activation_fn})."
                    f"\nMax diff: {(single_out - multi_out).abs().max()}"
                )
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 4: Full Forward Pass (forward)
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        {"activation_fn": "topk", "topk_k": 0.2},
        # T4 Degenerate dimensions case:
        {
            "activation_fn": "jumprelu",
            "num_features": 2,
            "d_model": 4,
            "num_layers": 1,
            "batchtopk_k": None,
            "topk_k": None,
        },
    ],
    indirect=True,
)
def test_full_forward_pass(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    clt_config_fn: CLTConfig,
    device: torch.device,
):
    """Compare the full forward pass outputs between single and multi-GPU."""

    input_dict = {}
    for layer_idx in range(clt_config_fn.num_layers):
        input_dict[layer_idx] = identical_input_data.clone()

    single_gpu_outputs = {}
    with torch.no_grad():
        single_gpu_outputs = single_gpu_model(input_dict)
        for layer_idx, output in single_gpu_outputs.items():
            expected_shape = (identical_input_data.shape[0], clt_config_fn.d_model)
            assert output.shape == expected_shape
            assert output.device == device

    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model
    multi_gpu_outputs = {}
    with torch.no_grad():
        input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
        multi_gpu_outputs = multi_gpu_actual_model(input_dict_multi)
        for layer_idx, output in multi_gpu_outputs.items():
            expected_shape = (identical_input_data.shape[0], clt_config_fn.d_model)
            assert output.shape == expected_shape
            assert output.device == device

    try:
        if RANK == 0:
            print(f"\nComparing Full Forward Pass Outputs (Rank 0) for config: {clt_config_fn.activation_fn}")
            assert single_gpu_outputs.keys() == multi_gpu_outputs.keys()
            for layer_idx in single_gpu_outputs.keys():
                single_out = single_gpu_outputs[layer_idx]
                multi_out = multi_gpu_outputs[layer_idx]
                multi_out = multi_out.to(single_out.device)

                print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
                if single_out.shape != multi_out.shape:
                    pytest.fail(
                        f"Shape mismatch forward layer {layer_idx}: Single={single_out.shape}, Multi={multi_out.shape}"
                    )
                assert torch.allclose(single_out, multi_out, atol=1e-5, rtol=1e-4), (
                    f"Mismatch in full forward pass for layer {layer_idx} (config: {clt_config_fn.activation_fn})."
                    f"\nMax diff: {(single_out - multi_out).abs().max()}"
                )
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 5: Reconstruction Loss Calculation
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        # {"activation_fn": "topk", "topk_k": 0.2} # TopK/BatchTopK affect activations, recon loss depends on these.
    ],
    indirect=True,
)
def test_reconstruction_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    clt_config_fn: CLTConfig,
    device: torch.device,
):
    """Compare the reconstruction loss component between single and multi-GPU."""
    input_dict = {}
    target_dict = {}
    for layer_idx in range(clt_config_fn.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : clt_config_fn.d_model]) * 0.5 + input_tensor[:, : clt_config_fn.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    single_gpu_outputs = single_gpu_model(input_dict)
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    single_recon_loss = loss_manager.compute_reconstruction_loss(single_gpu_outputs, target_dict_single)
    assert isinstance(single_recon_loss, torch.Tensor) and single_recon_loss.numel() == 1

    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    multi_gpu_outputs = multi_gpu_actual_model(input_dict_multi)
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    multi_recon_loss = loss_manager.compute_reconstruction_loss(multi_gpu_outputs, target_dict_multi)
    assert isinstance(multi_recon_loss, torch.Tensor) and multi_recon_loss.numel() == 1

    try:
        if RANK == 0:
            print(f"\nComparing Reconstruction Loss (Rank 0) for config: {clt_config_fn.activation_fn}")
            single_loss_val = single_recon_loss.item()
            multi_loss_val = multi_recon_loss.item()
            assert math.isclose(
                single_loss_val, multi_loss_val, rel_tol=1e-5, abs_tol=1e-6
            ), f"Mismatch in recon loss (config {clt_config_fn.activation_fn}): single ({single_loss_val}), multi ({multi_loss_val})."
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 6: Sparsity Loss Calculation (via total loss)
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        # {"activation_fn": "topk", "topk_k": 0.2}
    ],
    indirect=True,
)
def test_sparsity_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    clt_config_fn: CLTConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Compare the sparsity loss component (extracted from total loss) between single and multi-GPU."""
    input_dict = {}
    target_dict = {}
    for layer_idx in range(clt_config_fn.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : clt_config_fn.d_model]) * 0.5 + input_tensor[:, : clt_config_fn.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    _, single_loss_dict = loss_manager.compute_total_loss(
        single_gpu_model,
        input_dict,
        target_dict_single,
        current_step=0,
        total_steps=training_config.training_steps,
    )
    single_sparsity_loss_val = single_loss_dict.get("sparsity", 0.0)

    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    _, multi_loss_dict = loss_manager.compute_total_loss(
        multi_gpu_actual_model,
        input_dict_multi,
        target_dict_multi,
        current_step=0,
        total_steps=training_config.training_steps,
    )
    multi_sparsity_loss_val = multi_loss_dict.get("sparsity", 0.0)

    # --- Comparison (only on Rank 0) --- #
    try:
        if RANK == 0:
            print(f"\nComparing Sparsity Loss (Rank 0) for config: {clt_config_fn.activation_fn}")
            assert math.isclose(
                single_sparsity_loss_val, multi_sparsity_loss_val, rel_tol=1e-4, abs_tol=1e-5
            ), f"Mismatch in sparsity loss (config {clt_config_fn.activation_fn}): single ({single_sparsity_loss_val}), multi ({multi_sparsity_loss_val})."
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 7: Total Loss Calculation
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        # {"activation_fn": "topk", "topk_k": 0.2}
    ],
    indirect=True,
)
def test_total_loss(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    clt_config_fn: CLTConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Compare the total loss value between single and multi-GPU."""
    input_dict = {}
    target_dict = {}
    for layer_idx in range(clt_config_fn.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : clt_config_fn.d_model]) * 0.5 + input_tensor[:, : clt_config_fn.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    single_total_loss, _ = loss_manager.compute_total_loss(
        single_gpu_model, input_dict, target_dict_single, current_step=0, total_steps=training_config.training_steps
    )
    single_total_loss_val = single_total_loss.item()

    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    multi_total_loss, _ = loss_manager.compute_total_loss(
        multi_gpu_actual_model,
        input_dict_multi,
        target_dict_multi,
        current_step=0,
        total_steps=training_config.training_steps,
    )
    multi_total_loss_val = multi_total_loss.item()

    # --- Comparison (only on Rank 0) --- #
    try:
        if RANK == 0:
            print(f"\nComparing Total Loss (Rank 0) for config: {clt_config_fn.activation_fn}")
            assert math.isclose(
                single_total_loss_val, multi_total_loss_val, rel_tol=1e-4, abs_tol=1e-5
            ), f"Mismatch in total loss (config {clt_config_fn.activation_fn}): single ({single_total_loss_val}), multi ({multi_total_loss_val})."
    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():
            dist.barrier()


# Test 8: Gradient Calculation
@pytest.mark.parametrize(
    "clt_config_fn",
    [
        {"activation_fn": "jumprelu"},
        {"activation_fn": "batchtopk", "batchtopk_k": 10},
        {"activation_fn": "topk", "topk_k": 0.2},
    ],
    indirect=True,
)
def test_gradient_calculation(
    single_gpu_model: CrossLayerTranscoder,
    multi_gpu_model: Optional[CrossLayerTranscoder],
    identical_input_data: torch.Tensor,
    loss_manager: LossManager,
    clt_config_fn: CLTConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Compare gradients between single and multi-GPU after backward pass."""

    input_dict = {}
    target_dict = {}
    for layer_idx in range(clt_config_fn.num_layers):
        input_tensor = identical_input_data.clone()
        input_dict[layer_idx] = input_tensor
        target_tensor = (
            torch.randn_like(input_tensor[:, : clt_config_fn.d_model]) * 0.5 + input_tensor[:, : clt_config_fn.d_model]
        )
        target_dict[layer_idx] = target_tensor.to(device)

    # --- Single GPU Backward Pass --- #
    single_gpu_model.train()
    single_gpu_model.zero_grad()
    target_dict_single = {k: v.clone() for k, v in target_dict.items()}
    single_total_loss, _ = loss_manager.compute_total_loss(
        single_gpu_model, input_dict, target_dict_single, current_step=0, total_steps=training_config.training_steps
    )
    single_total_loss.backward()
    single_gpu_grads = {name: p.grad.clone() for name, p in single_gpu_model.named_parameters() if p.grad is not None}
    single_gpu_model.eval()

    # --- Multi GPU Backward Pass --- #
    multi_gpu_actual_model = multi_gpu_model if multi_gpu_model is not None else single_gpu_model

    multi_gpu_actual_model.train()
    multi_gpu_actual_model.zero_grad()
    input_dict_multi = {k: v.clone() for k, v in input_dict.items()}
    target_dict_multi = {k: v.clone() for k, v in target_dict.items()}
    multi_total_loss, _ = loss_manager.compute_total_loss(
        multi_gpu_actual_model,
        input_dict_multi,
        target_dict_multi,
        current_step=0,
        total_steps=training_config.training_steps,
    )
    multi_total_loss.backward()
    if multi_gpu_model is not None and WORLD_SIZE > 1:  # Only average for actual multi_gpu_model
        average_replicated_grads(multi_gpu_model)
    if dist.is_initialized():
        dist.barrier()

    multi_gpu_local_grads = {
        name: p.grad for name, p in multi_gpu_actual_model.named_parameters() if p.grad is not None
    }
    multi_gpu_actual_model.eval()

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()

    # --- Comparison (only on Rank 0, with final barrier for all) --- #
    try:
        if RANK == 0:
            print(f"\nComparing Gradients (Rank 0) for config: {clt_config_fn.activation_fn}")
            assert (
                single_gpu_grads.keys() == multi_gpu_local_grads.keys()
            ), f"Grad keys differ (config {clt_config_fn.activation_fn}): {single_gpu_grads.keys()} vs {multi_gpu_local_grads.keys()}"

        multi_gpu_params_dict = dict(multi_gpu_actual_model.named_parameters())
        gradient_mismatch_detected = False
        mismatch_messages = []  # This list is populated by RANK 0

        # This loop is executed by all ranks to prepare full_multi_grad if needed,
        # but comparisons and mismatch_messages updates are RANK 0 only.
        for name, single_grad_val in single_gpu_grads.items():
            if name not in multi_gpu_local_grads:
                if RANK == 0:
                    msg = f"Warning: Grad for {name} missing in multi-GPU model (config {clt_config_fn.activation_fn})."
                    print(msg)
                    gradient_mismatch_detected = True
                    mismatch_messages.append(msg)
                continue

            local_grad_val = multi_gpu_local_grads[name]
            param = multi_gpu_params_dict[name]
            full_multi_grad = None
            is_sharded = False
            partition_dim = -1

            if "encoders." in name and ".weight" in name:
                is_sharded = True
                partition_dim = 0
            elif "decoders." in name and ".weight" in name:
                is_sharded = True
                partition_dim = 1
            elif "encoders." in name and ".bias_param" in name:
                if (
                    hasattr(multi_gpu_actual_model.encoders[int(name.split(".")[1])], "bias_param")
                    and param is not None
                    and param.numel() > 0
                ):
                    is_sharded = True
                    partition_dim = 0

            if is_sharded and multi_gpu_model is not None and WORLD_SIZE > 1:
                # All ranks must call gather_sharded_gradient if it involves collectives (dist.all_gather)
                full_multi_grad = gather_sharded_gradient(local_grad_val, param, single_grad_val.shape, partition_dim)
            elif not is_sharded:  # Replicated param or single GPU run
                full_multi_grad = local_grad_val
            elif is_sharded and (multi_gpu_model is None or WORLD_SIZE == 1):  # Sharded but running in single GPU mode
                full_multi_grad = local_grad_val

            if RANK == 0:  # Assertions and mismatch reporting only on Rank 0
                print(f"  Comparing grad for: {name} (shape {single_grad_val.shape})")
                single_grad_to_compare = single_grad_val.to(device)

                if full_multi_grad is None and is_sharded and WORLD_SIZE > 1:
                    gradient_mismatch_detected = True
                    mismatch_messages.append(
                        f"Failed to obtain full_multi_grad for sharded {name} on Rank 0 (config {clt_config_fn.activation_fn})."
                    )
                    continue

                multi_grad_to_compare = full_multi_grad.to(device) if full_multi_grad is not None else None

                if multi_grad_to_compare is None:
                    gradient_mismatch_detected = True
                    mismatch_messages.append(
                        f"Multi grad for {name} is None on Rank 0 (config {clt_config_fn.activation_fn})."
                    )
                    continue

                atol, rtol = (1e-4, 1e-3) if is_sharded else (1e-4, 1e-4)
                if name == "log_threshold":
                    atol, rtol = (1e-5, 1e-4)

                if not torch.allclose(single_grad_to_compare, multi_grad_to_compare, atol=atol, rtol=rtol):
                    gradient_mismatch_detected = True
                    diff = (single_grad_to_compare - multi_grad_to_compare).abs().max()
                    mismatch_messages.append(
                        f"Mismatch for '{name}' (config {clt_config_fn.activation_fn}). Max diff: {diff}. Type: {'Sharded' if is_sharded else 'Replicated/Other'}"
                    )

        # Actual failure point for RANK 0 if mismatches were detected
        if RANK == 0 and gradient_mismatch_detected:
            pytest.fail("\n".join(mismatch_messages))

    finally:
        if WORLD_SIZE > 1 and dist.is_initialized():  # Ensure non-rank0 pass if no errors on rank0
            dist.barrier()


# --- Test T1: Replicated Parameter Sync after JumpReLU conversion ---


# Dummy data iterator for estimate_theta_posthoc
def _dummy_data_iterator(num_batches, batch_size_tokens, d_model, num_layers, device, dtype):
    for _ in range(num_batches):
        inputs_dict = {}
        for i in range(num_layers):
            # Shape: [batch_size_tokens, d_model]
            inputs_dict[i] = torch.randn(batch_size_tokens, d_model, device=device, dtype=dtype)
        # estimate_theta_posthoc only uses inputs from the iterator
        yield inputs_dict, None  # targets are not used by the parts of estimate_theta we call


@pytest.mark.parametrize(
    "clt_config_fn",
    [{"activation_fn": "batchtopk", "batchtopk_k": 10, "num_features": 64, "d_model": 32, "num_layers": 2}],
    indirect=True,
)
def test_replicated_param_sync_after_conversion(
    clt_config_fn: CLTConfig,  # Starts as batchtopk
    training_config: TrainingConfig,  # For optimizer
    loss_manager: LossManager,
    device: torch.device,
):
    """
    Tests T1: Replicated parameter synchronization after BatchTopK -> JumpReLU conversion.
    Checks log_threshold (2D) and RowParallelLinear biases.
    """
    # 1. Create models: single_gpu for reference, multi_gpu for distributed testing
    # Re-create models here to ensure they use the specific "batchtopk" config from parametrize
    torch.manual_seed(42 + RANK)  # Ensure models start same across ranks before potential divergence

    _single_model = CrossLayerTranscoder(clt_config_fn, process_group=None, device=device)
    _single_model.eval()
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()

    _multi_model: Optional[CrossLayerTranscoder] = None
    if WORLD_SIZE > 1 and dist.is_initialized():
        _multi_model = CrossLayerTranscoder(clt_config_fn, process_group=dist.group.WORLD, device=device)
        _multi_model.eval()
        # Copy weights from _single_model to _multi_model
        single_params_dict = dict(_single_model.named_parameters())
        with torch.no_grad():
            for name, multi_param in _multi_model.named_parameters():
                if name not in single_params_dict:
                    if RANK == 0:
                        print(f"Warning T1: Param {name} in multi_model not in single_model, skipping copy.")
                    continue

                single_param_data = single_params_dict[name].data.to(device)

                if "encoders." in name:
                    layer_idx_str = name.split(".")[1]
                    if ".weight" in name:
                        scatter_full_parameter(single_param_data, multi_param, 0)
                    elif ".bias_param" in name:
                        # Encoder bias is sharded if it exists
                        if _multi_model.encoders[int(layer_idx_str)].bias:  # Check if bias was created
                            scatter_full_parameter(single_param_data, multi_param, 0)
                elif "decoders." in name:
                    decoder_key = name.split(".")[1]  # This is "src->tgt"
                    if ".weight" in name:
                        scatter_full_parameter(single_param_data, multi_param, 1)
                    elif ".bias_param" in name:
                        # Decoder bias is replicated
                        if _multi_model.decoders[decoder_key].bias:  # Check if bias was created
                            multi_param.data.copy_(single_param_data)
                # log_threshold is not copied here as it's for batchtopk init
                # Other parameters (if any) are not handled by this specific copy logic

        dist.barrier()

    model_to_test = _multi_model if WORLD_SIZE > 1 and _multi_model is not None else _single_model

    # 2. Convert to JumpReLU
    if clt_config_fn.activation_fn == "batchtopk":
        if RANK == 0:
            print("\nConverting model to JumpReLU for T1 test...")
        data_iter = _dummy_data_iterator(
            num_batches=2,  # Small number of batches for estimation
            batch_size_tokens=training_config.train_batch_size_tokens // 4,  # Smaller batch
            d_model=clt_config_fn.d_model,
            num_layers=clt_config_fn.num_layers,
            device=device,
            dtype=model_to_test.dtype,  # Match model dtype
        )
        model_to_test.estimate_theta_posthoc(data_iter, num_batches=2, device=device)
        # convert_to_jumprelu_inplace is called by estimate_theta_posthoc
        assert model_to_test.config.activation_fn == "jumprelu"
        assert hasattr(model_to_test, "log_threshold")
        # mark_replicated should have been called inside convert_to_jumprelu_inplace
        assert getattr(model_to_test.log_threshold, "_is_replicated", False) is True
        if RANK == 0:
            print("Model converted. log_threshold shape:", model_to_test.log_threshold.shape)

    # 3. Optimizer (AdamW is fine)
    optimizer = torch.optim.AdamW(model_to_test.parameters(), lr=1e-3)

    # 4. Dummy input and target for a training step
    input_dict = {}
    target_dict = {}
    for i in range(clt_config_fn.num_layers):
        input_dict[i] = torch.randn(
            training_config.train_batch_size_tokens // 4,
            clt_config_fn.d_model,
            device=device,
            dtype=model_to_test.dtype,
        )
        target_dict[i] = torch.randn(
            training_config.train_batch_size_tokens // 4,
            clt_config_fn.d_model,
            device=device,
            dtype=model_to_test.dtype,
        )

    # 5. Forward/Backward/Optimize step
    model_to_test.train()
    optimizer.zero_grad()
    loss, _ = loss_manager.compute_total_loss(model_to_test, input_dict, target_dict, 0, 1)
    loss.backward()

    if WORLD_SIZE > 1 and dist.is_initialized() and _multi_model is not None:
        average_replicated_grads(_multi_model)  # Apply our fixed averaging

    optimizer.step()
    model_to_test.eval()

    if WORLD_SIZE <= 1:  # Test doesn't apply to single GPU for param sync
        if RANK == 0:
            print("Skipping T1 param sync checks for WORLD_SIZE=1")
        return

    # 6. Check sync for log_threshold and decoder biases
    mismatch_messages_t1 = []
    # Check log_threshold
    if hasattr(model_to_test, "log_threshold") and model_to_test.log_threshold is not None:
        log_thresh = model_to_test.log_threshold.data.clone()
        gathered_log_threshs = [torch.empty_like(log_thresh) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_log_threshs, log_thresh)
        if RANK == 0:
            for i in range(1, WORLD_SIZE):
                if not torch.allclose(gathered_log_threshs[0], gathered_log_threshs[i]):
                    diff = (gathered_log_threshs[0] - gathered_log_threshs[i]).abs().max()
                    mismatch_messages_t1.append(
                        f"T1: log_threshold mismatch between rank 0 and rank {i}. Max diff: {diff}"
                    )
                    break
            if not mismatch_messages_t1:
                print("T1: log_threshold synced after step.")

    # Check decoder biases (RowParallelLinear, should be marked replicated)
    for name, param in model_to_test.named_parameters():
        if "decoders." in name and ".bias_param" in name and getattr(param, "_is_replicated", False):
            bias_data = param.data.clone()
            gathered_biases = [torch.empty_like(bias_data) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered_biases, bias_data)
            if RANK == 0:
                for i in range(1, WORLD_SIZE):
                    if not torch.allclose(gathered_biases[0], gathered_biases[i]):
                        diff = (gathered_biases[0] - gathered_biases[i]).abs().max()
                        mismatch_messages_t1.append(f"T1: Decoder bias {name} mismatch rank 0 vs {i}. Max diff: {diff}")
                        break
                if not any(name in msg for msg in mismatch_messages_t1):
                    print(f"T1: Decoder bias {name} synced.")

    # --- Final assertions (Rank 0) and cleanup --- #
    try:
        if RANK == 0 and mismatch_messages_t1:
            pytest.fail("\n".join(mismatch_messages_t1))
    finally:
        if dist.is_initialized():  # Check before calling barrier
            dist.barrier()


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


@pytest.mark.parametrize(
    "clt_config_fn",
    [{"activation_fn": "jumprelu", "d_model": 32, "num_layers": 4}],  # Provide a default config
    indirect=True,
)
def test_row_parallel_linear_forward(
    clt_config_fn: CLTConfig,  # Changed from base_config
    device: torch.device,
):
    """Test RowParallelLinear forward pass against nn.Linear."""
    if WORLD_SIZE <= 1:
        pytest.skip("Skipping RowParallelLinear test (WORLD_SIZE <= 1)")

    # Note: For RowParallel, input features are sharded.
    # Output features remain the full dimension.
    in_features = 64  # Feature dim (sharded)
    out_features = 32  # Model dim (full) - Matches clt_config_fn.d_model
    if clt_config_fn.d_model != out_features:
        # Adjust if clt_config_fn d_model is different, or ensure it matches test setup
        # For this test, let's use a fixed out_features and ensure config matches, or make it flexible.
        # The parametrize for clt_config_fn sets d_model=32, so this is consistent.
        pass

    batch_tokens = 128
    seed = 42 + RANK

    test_device = device
    if RANK == 0:
        print(f"Rank {RANK}: Running RPL test on device: {test_device}")

    # 1. Create standard nn.Linear layer (broadcast weights)
    torch.manual_seed(seed)
    single_layer = nn.Linear(in_features, out_features, bias=True).to(test_device)
    if dist.is_initialized():
        dist.broadcast(single_layer.weight.data, src=0)
        dist.broadcast(single_layer.bias.data, src=0)
        dist.barrier()
    if RANK == 0:
        print(f"Rank {RANK}: Single nn.Linear layer created and weights broadcasted.")

    # 2. Create RowParallelLinear layer
    torch.manual_seed(seed)
    multi_layer = RowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        process_group=dist.group.WORLD,
        input_is_parallel=False,  # Mimic usage in decode - layer handles split
        d_model_for_init=clt_config_fn.d_model,  # Use d_model from config
        num_layers_for_init=clt_config_fn.num_layers,  # Use num_layers from config
        device=test_device,
    )
    multi_layer.eval()
    if RANK == 0:
        print(f"Rank {RANK}: RowParallelLinear created.")

    # 3. Copy weights/bias from single_layer to multi_layer
    if RANK == 0:
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


# --- Synchronize ranks between individual tests --- #
# Ensures that all ranks finish a test before any starts the next one.
@pytest.fixture(scope="function", autouse=True)
def sync_barrier_between_tests():
    """Barrier after each test function to keep ranks perfectly in lock-step.

    Without this, one rank can finish a test and start creating the next
    fixture (which includes new collectives) while another rank is still in
    the previous test.  That race leads to mismatched collective sequences
    and NCCL watchdog timeouts.
    """
    yield
    if dist.is_initialized():
        dist.barrier()
