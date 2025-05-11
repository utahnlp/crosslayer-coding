import torch
import torch.distributed as dist
import pytest
import os
import math
from typing import Optional, Union
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

    if WORLD_SIZE > 1 and dist.is_initialized():
        with torch.no_grad():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        dist.barrier()
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

    # --- Copy weights from single_gpu_model to multi_gpu_model shards --- #
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
    if RANK == 0:
        print(f"\nComparing Feature Activations (Rank 0) for config: {clt_config_fn.activation_fn}")
        for layer_idx in range(clt_config_fn.num_layers):
            single_out = single_gpu_outputs[layer_idx]
            multi_out = multi_gpu_outputs[layer_idx]
            # Ensure multi_out is on the same device as single_out for comparison if it was from single_gpu_model fallback
            multi_out = multi_out.to(single_out.device)

            print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            assert torch.allclose(single_out, multi_out, atol=1e-6), (
                f"Mismatch in feature activations for layer {layer_idx} (config: {clt_config_fn.activation_fn})."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )
            # For topk variants, check that non-selected are zero
            if clt_config_fn.activation_fn == "topk" or clt_config_fn.activation_fn == "batchtopk":
                # Calculate the number of non-zeros we expect
                k_val_resolved: Optional[Union[float, int]] = None
                if clt_config_fn.activation_fn == "topk":
                    k_val_resolved = clt_config_fn.topk_k
                elif clt_config_fn.activation_fn == "batchtopk":
                    k_val_resolved = clt_config_fn.batchtopk_k

                num_expected_non_zero_per_token: int = 0
                if k_val_resolved is not None:
                    if 0 < k_val_resolved < 1:  # k_val_resolved is float here
                        num_expected_non_zero_per_token = math.ceil(k_val_resolved * clt_config_fn.num_features)
                    elif k_val_resolved >= 1:  # k_val_resolved is float or int here
                        num_expected_non_zero_per_token = int(k_val_resolved)

                if (
                    num_expected_non_zero_per_token > 0
                    and num_expected_non_zero_per_token <= clt_config_fn.num_features
                ):
                    if clt_config_fn.activation_fn == "topk":
                        # For TokenTopK, count non-zeros per token
                        non_zeros_per_token = (multi_out != 0).float().sum(dim=-1)
                        assert torch.all(
                            non_zeros_per_token <= num_expected_non_zero_per_token
                        ), f"Layer {layer_idx} (topk): Expected at most {num_expected_non_zero_per_token} non-zeros per token, found more."
                    elif clt_config_fn.activation_fn == "batchtopk":
                        # For BatchTopK, count total non-zeros across batch for this layer's output
                        total_non_zeros = (multi_out != 0).float().sum()
                        max_expected_total_non_zeros = (
                            num_expected_non_zero_per_token * multi_out.shape[0]
                        )  # k_per_token * B_tokens
                        assert (
                            total_non_zeros <= max_expected_total_non_zeros
                        ), f"Layer {layer_idx} (batchtopk): Expected at most {max_expected_total_non_zeros} total non-zeros, found {total_non_zeros}."

    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()  # Ensure all ranks complete before next test item


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

    if RANK == 0:
        print(f"\nComparing Decoder Outputs (Rank 0) for config: {clt_config_fn.activation_fn}")
        for layer_idx in range(clt_config_fn.num_layers):
            single_out = single_gpu_reconstructions[layer_idx]
            multi_out = multi_gpu_reconstructions[layer_idx]
            multi_out = multi_out.to(single_out.device)

            print(f"  Layer {layer_idx}: Single GPU shape={single_out.shape}, Multi GPU shape={multi_out.shape}")
            if single_out.shape != multi_out.shape:
                pytest.fail(f"Shape mismatch for layer {layer_idx}: Single={single_out.shape}, Multi={multi_out.shape}")
            assert torch.allclose(single_out, multi_out, atol=1e-5, rtol=1e-4), (
                f"Mismatch in decoder outputs for layer {layer_idx} (config: {clt_config_fn.activation_fn})."
                f"\nMax diff: {(single_out - multi_out).abs().max()}"
            )
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

    if RANK == 0:
        print(f"\nComparing Reconstruction Loss (Rank 0) for config: {clt_config_fn.activation_fn}")
        single_loss_val = single_recon_loss.item()
        multi_loss_val = multi_recon_loss.item()
        assert math.isclose(
            single_loss_val, multi_loss_val, rel_tol=1e-5, abs_tol=1e-6
        ), f"Mismatch in recon loss (config {clt_config_fn.activation_fn}): single ({single_loss_val}), multi ({multi_loss_val})."
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
