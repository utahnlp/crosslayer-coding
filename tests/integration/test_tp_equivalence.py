import torch
import torch.distributed as dist
import pytest
import os
import math
from typing import Optional, Dict, Any, cast
import torch.multiprocessing as mp

import torch.nn as nn

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.losses import LossManager
from clt.models.parallel import ColumnParallelLinear, RowParallelLinear


# --- Test Setup ---
# Use multiprocessing to spawn processes for distributed tests
# This is more robust than relying on environment variables alone for pytest
def setup_distributed_environment(rank: int, world_size: int, port: str):
    """Initializes the distributed process group for a test worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # Set device for this process
    # No CUDA support on this machine, so we use CPU. If CUDA were available:
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(rank)


def cleanup_distributed_environment():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def distributed_test_runner(rank: int, world_size: int, port: str, test_fn, *args):
    """A wrapper to run a distributed test function in a spawned process."""
    setup_distributed_environment(rank, world_size, port)
    try:
        # Pass rank and world_size to the actual test function
        test_fn(rank, world_size, *args)
    finally:
        cleanup_distributed_environment()


# --- Helper Functions for Equivalence Testing ---


def scatter_full_parameter(
    full_param: torch.Tensor, model_param: nn.Parameter, partition_dim: int, world_size: int, rank: int
):
    """
    Splits a full parameter tensor and loads the correct shard onto the current rank.
    Handles the padding required for uniform sharding.
    """
    if not dist.is_initialized():
        if model_param.shape == full_param.shape:
            model_param.data.copy_(full_param)
        return

    full_dim_size = full_param.size(partition_dim)
    # Calculate padded size for uniform distribution
    local_dim_padded = math.ceil(full_dim_size / world_size)

    start_index = rank * local_dim_padded
    # The actual size of the slice might be smaller than the padded size for the last rank
    local_dim_actual = model_param.size(partition_dim)
    end_index = start_index + local_dim_actual

    # Ensure we don't go past the original tensor's boundary
    end_index = min(end_index, full_dim_size)
    actual_slice_size = max(0, end_index - start_index)

    if actual_slice_size > 0:
        indices = [slice(None)] * full_param.dim()
        indices[partition_dim] = slice(start_index, end_index)
        param_shard = full_param[tuple(indices)].clone()

        # If the model's parameter is larger due to padding, we need to pad the shard
        pad_amount = model_param.size(partition_dim) - param_shard.size(partition_dim)
        if pad_amount > 0:
            pad_dims = [0, 0] * model_param.dim()
            # F.pad takes pads in reverse order of dimensions
            pad_idx = model_param.dim() - 1 - partition_dim
            pad_dims[2 * pad_idx + 1] = pad_amount  # Pad on the right
            param_shard = torch.nn.functional.pad(param_shard, tuple(pad_dims))

        if model_param.shape == param_shard.shape:
            model_param.data.copy_(param_shard)
        else:
            raise ValueError(
                f"Shape mismatch during scatter on rank {rank}: {model_param.shape} vs {param_shard.shape}"
            )


def gather_sharded_gradient(local_grad: torch.Tensor, full_shape: tuple, partition_dim: int) -> Optional[torch.Tensor]:
    """Gathers gradient slices from all ranks and reconstructs the full gradient tensor."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_grad.clone()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # All ranks need a tensor of the same size for all_gather
    gathered_grads_list = [torch.empty_like(local_grad) for _ in range(world_size)]
    dist.all_gather(gathered_grads_list, local_grad.contiguous())

    if rank == 0:
        # Reconstruct the full gradient on rank 0
        full_grad = torch.cat(gathered_grads_list, dim=partition_dim)

        # Truncate any padding that was added for uniform sharding
        if full_grad.size(partition_dim) > full_shape[partition_dim]:
            slicing_indices = [slice(None)] * full_grad.dim()
            slicing_indices[partition_dim] = slice(0, full_shape[partition_dim])
            full_grad = full_grad[tuple(slicing_indices)]

        if full_grad.shape != full_shape:
            raise ValueError(f"Reconstructed gradient shape {full_grad.shape} != expected {full_shape}")
        return full_grad.contiguous()
    else:
        return None


def average_replicated_grads(model: CrossLayerTranscoder):
    """Averages gradients for parameters marked as replicated across all ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    world_size = float(dist.get_world_size())
    for p in model.parameters():
        if p.grad is not None and getattr(p, "_is_replicated", False):
            # Sum gradients from all ranks
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            # Divide by world size to get the average
            if world_size > 0:
                p.grad.data.div_(world_size)


# --- Test Fixture Setup ---


class EquivalenceTester:
    """A class to hold state and methods for a single equivalence test run."""

    def __init__(self, rank: int, world_size: int, clt_config_params: Dict[str, Any]):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")  # Using CPU for this test
        self.clt_config = self._create_clt_config(clt_config_params)
        self.training_config = self._create_training_config()
        self.loss_manager = LossManager(self.training_config)

        # Create models
        self.single_model = self._create_single_model()
        self.tp_model = self._create_tp_model()

        # Sync weights from single to TP model
        self._sync_weights()

    def _create_clt_config(self, params: Dict[str, Any]) -> CLTConfig:
        return CLTConfig(
            d_model=params.get("d_model", 32),
            num_features=params.get("num_features", 64),
            num_layers=params.get("num_layers", 4),
            activation_fn=params.get("activation_fn", "relu"),
            jumprelu_threshold=params.get("jumprelu_threshold", 0.01),
            batchtopk_k=params.get("batchtopk_k", None),
            topk_k=params.get("topk_k", None),
            clt_dtype="float32",
        )

    def _create_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            learning_rate=1e-4,
            training_steps=100,
            train_batch_size_tokens=1024,
            activation_source="local_manifest",
            activation_path="dummy",
            sparsity_lambda=0.01,
            preactivation_coef=0.0,
            normalization_method="none",
        )

    def _create_single_model(self) -> CrossLayerTranscoder:
        # Ensure deterministic init for the reference model
        torch.manual_seed(42)
        model = CrossLayerTranscoder(self.clt_config, process_group=None, device=self.device)
        model.eval()
        # Broadcast from rank 0 to ensure all processes have the identical reference model
        if dist.is_initialized():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        return model

    def _create_tp_model(self) -> CrossLayerTranscoder:
        # Use a different seed for the TP model to ensure its initial state is different before syncing
        torch.manual_seed(1337 + self.rank)
        model = CrossLayerTranscoder(self.clt_config, process_group=dist.group.WORLD, device=self.device)
        model.eval()
        return model

    def _sync_weights(self):
        """Copy weights from the single-GPU model to the sharded tensor-parallel model."""
        with torch.no_grad():
            single_params = dict(self.single_model.named_parameters())
            for name, tp_param in self.tp_model.named_parameters():
                if name not in single_params:
                    raise ValueError(f"Parameter {name} not found in single-GPU model.")

                full_param = single_params[name].data

                if "encoder_module.encoders" in name and ".weight" in name:
                    scatter_full_parameter(
                        full_param, tp_param, partition_dim=0, world_size=self.world_size, rank=self.rank
                    )
                elif "decoder_module.decoders" in name and ".weight" in name:
                    scatter_full_parameter(
                        full_param, tp_param, partition_dim=1, world_size=self.world_size, rank=self.rank
                    )
                elif "encoder_module.encoders" in name and ".bias_param" in name:
                    scatter_full_parameter(
                        full_param, tp_param, partition_dim=0, world_size=self.world_size, rank=self.rank
                    )
                elif "decoder_module.decoders" in name and ".bias_param" in name:
                    # Decoder bias is replicated
                    tp_param.data.copy_(full_param)
                elif "theta_manager.log_threshold" in name:
                    # log_threshold is replicated
                    tp_param.data.copy_(full_param)
                else:
                    raise ValueError(f"Unknown parameter sharding strategy for {name}")

        if dist.is_initialized():
            dist.barrier()

        # Add verification step
        if self.rank == 0:
            print("\nVerifying synced weights on Rank 0...")
        self._verify_synced_weights()

    def _verify_synced_weights(self):
        """Compare single model weights with reconstructed TP model weights."""
        single_params = dict(self.single_model.named_parameters())
        tp_params = dict(self.tp_model.named_parameters())

        for name, single_param in single_params.items():
            if name not in tp_params:
                continue

            tp_param = tp_params[name]

            # For sharded params, we need to gather them first
            is_sharded = not getattr(tp_param, "_is_replicated", False)
            full_tp_param = None

            if is_sharded:
                partition_dim = 0 if "encoder" in name else 1

                # To verify, we need to gather the shards from all ranks
                world_size = dist.get_world_size()
                gathered_shards = [torch.empty_like(tp_param.data) for _ in range(world_size)]
                dist.all_gather(gathered_shards, tp_param.data.contiguous())

                if self.rank == 0:
                    full_tp_tensor = torch.cat(gathered_shards, dim=partition_dim)
                    # Trim padding
                    if full_tp_tensor.size(partition_dim) > single_param.size(partition_dim):
                        slicing_indices = [slice(None)] * full_tp_tensor.dim()
                        slicing_indices[partition_dim] = slice(0, single_param.size(partition_dim))
                        full_tp_param = full_tp_tensor[tuple(slicing_indices)]
                    else:
                        full_tp_param = full_tp_tensor
            else:  # Replicated
                full_tp_param = tp_param.data

            if self.rank == 0:
                assert full_tp_param is not None, f"Verification failed: full_tp_param is None for {name}"
                if not torch.allclose(single_param.data, full_tp_param, atol=1e-7):
                    print(f" MISMATCH in {name}")
                    print(f"  Single param sample: {single_param.data.flatten()[:5]}")
                    print(f"  TP param sample:     {full_tp_param.flatten()[:5]}")
                    print(f"  Max abs diff: {(single_param.data - full_tp_param).abs().max()}")
                else:
                    print(f"  ✅ {name} synced correctly.")

        if dist.is_initialized():
            dist.barrier()

    def get_test_data(self) -> Dict[int, torch.Tensor]:
        """Generate identical input data across all ranks."""
        torch.manual_seed(123)
        input_data = torch.randn(128, self.clt_config.d_model, device=self.device)
        if dist.is_initialized():
            dist.broadcast(input_data, src=0)

        return {i: input_data.clone() for i in range(self.clt_config.num_layers)}

    def run_column_parallel_verification(self):
        """Isolates and verifies a ColumnParallelLinear layer."""
        if self.rank == 0:
            print("\n--- Verifying ColumnParallelLinear ---")

        # Get the first encoder layer from both models
        single_encoder_module = self.single_model.encoder_module.encoders[0]
        tp_encoder_module = self.tp_model.encoder_module.encoders[0]

        # Explicitly cast to the correct type to help the linter
        single_encoder = cast(ColumnParallelLinear, single_encoder_module)
        tp_encoder = cast(ColumnParallelLinear, tp_encoder_module)

        # Get a test input
        input_data = self.get_test_data()[0]  # Use data for layer 0

        # Create a standard nn.Linear layer for reference
        reference_linear = nn.Linear(
            in_features=single_encoder.full_in_features,
            out_features=single_encoder.full_out_features,
            bias=True,  # Assuming bias is true for encoders
            device=self.device,
        )
        # Manually set its weights to be identical to the single model's encoder
        reference_linear.weight.data.copy_(single_encoder.weight.data)
        if single_encoder.bias_param is not None:
            reference_linear.bias.data.copy_(single_encoder.bias_param.data)

        # 1. Run forward pass on the reference nn.Linear
        ref_output = reference_linear(input_data.clone())

        # 2. Run forward pass on the TP ColumnParallelLinear layer
        tp_output = tp_encoder(input_data.clone())

        # 3. Compare outputs on rank 0
        if self.rank == 0:
            max_abs_diff = (ref_output - tp_output).abs().max()
            print(f"Reference output sample: {ref_output.flatten()[:5]}")
            print(f"TP output sample:        {tp_output.flatten()[:5]}")
            print(f"Max absolute difference: {max_abs_diff}")
            assert torch.allclose(
                ref_output, tp_output, atol=1e-6
            ), "ColumnParallelLinear output does not match reference nn.Linear."
            print("✅ ColumnParallelLinear verification passed.")

        if dist.is_initialized():
            dist.barrier()

    def run_row_parallel_verification(self):
        """Isolates and verifies a RowParallelLinear layer."""
        if self.rank == 0:
            print("\n--- Verifying RowParallelLinear ---")

        # Get the first decoder layer from both models
        single_decoder_module = self.single_model.decoder_module.decoders["0->0"]
        tp_decoder_module = self.tp_model.decoder_module.decoders["0->0"]

        single_decoder = cast(RowParallelLinear, single_decoder_module)
        tp_decoder = cast(RowParallelLinear, tp_decoder_module)

        # Create a test input (should be the size of num_features)
        torch.manual_seed(456)
        input_data = torch.randn(128, self.clt_config.num_features, device=self.device)
        if dist.is_initialized():
            dist.broadcast(input_data, src=0)

        # Create a standard nn.Linear layer for reference
        reference_linear = nn.Linear(
            in_features=single_decoder.full_in_features,
            out_features=single_decoder.full_out_features,
            bias=True,
            device=self.device,
        )
        reference_linear.weight.data.copy_(single_decoder.weight.data)
        if single_decoder.bias_param is not None:
            reference_linear.bias.data.copy_(single_decoder.bias_param.data)

        # 1. Run forward pass on the reference nn.Linear
        ref_output = reference_linear(input_data.clone())

        # 2. Run forward pass on the TP RowParallelLinear layer
        # It expects `input_is_parallel=False` in our setup
        tp_output = tp_decoder(input_data.clone())

        # 3. Compare outputs on rank 0
        if self.rank == 0:
            max_abs_diff = (ref_output - tp_output).abs().max()
            print(f"Reference output sample: {ref_output.flatten()[:5]}")
            print(f"TP output sample:        {tp_output.flatten()[:5]}")
            print(f"Max absolute difference: {max_abs_diff}")
            assert torch.allclose(
                ref_output, tp_output, atol=1e-5
            ), "RowParallelLinear output does not match reference nn.Linear."
            print("✅ RowParallelLinear verification passed.")

        if dist.is_initialized():
            dist.barrier()

    def run_forward_pass_test(self):
        """Test equivalence of the full forward pass."""
        test_data = self.get_test_data()

        with torch.no_grad():
            single_out = self.single_model(test_data)
            tp_out = self.tp_model({k: v.clone() for k, v in test_data.items()})

        if self.rank == 0:
            assert single_out.keys() == tp_out.keys()
            for layer_idx in single_out:
                s_out = single_out[layer_idx]
                t_out = tp_out[layer_idx]
                max_abs_diff = (s_out - t_out).abs().max()
                max_rel_diff = ((s_out - t_out) / s_out.abs().clamp(min=1e-9)).abs().max()

                print(f"\n--- Layer {layer_idx} ---")
                print(f"Single model output sample: {s_out.flatten()[:5]}")
                print(f"TP model output sample:     {t_out.flatten()[:5]}")
                print(f"Max absolute difference: {max_abs_diff}")
                print(f"Max relative difference: {max_rel_diff}")

                assert torch.allclose(s_out, t_out, atol=1e-6), f"Forward pass mismatch on layer {layer_idx}"
            print("✅ Forward pass test passed.")

        if dist.is_initialized():
            dist.barrier()

    def run_loss_calculation_test(self):
        """Test equivalence of loss calculation."""
        test_data = self.get_test_data()
        target_data = {k: torch.randn_like(v) for k, v in test_data.items()}

        _, single_loss_dict = self.loss_manager.compute_total_loss(self.single_model, test_data, target_data, 0, 100)
        _, tp_loss_dict = self.loss_manager.compute_total_loss(
            self.tp_model,
            {k: v.clone() for k, v in test_data.items()},
            {k: v.clone() for k, v in target_data.items()},
            0,
            100,
        )

        if self.rank == 0:
            assert single_loss_dict.keys() == tp_loss_dict.keys()
            for key in single_loss_dict:
                assert math.isclose(
                    single_loss_dict[key], tp_loss_dict[key], rel_tol=1e-5
                ), f"Loss component '{key}' mismatch: single={single_loss_dict[key]}, tp={tp_loss_dict[key]}"
            print("✅ Loss calculation test passed.")

        if dist.is_initialized():
            dist.barrier()

    def run_gradient_test(self):
        """Test equivalence of gradients after backward pass."""
        # Setup models for training
        self.single_model.train()
        self.tp_model.train()
        self.single_model.zero_grad()
        self.tp_model.zero_grad()

        test_data = self.get_test_data()
        target_data = {k: torch.randn_like(v) for k, v in test_data.items()}

        # Single model backward
        single_loss, _ = self.loss_manager.compute_total_loss(self.single_model, test_data, target_data, 0, 100)
        single_loss.backward()

        # TP model backward
        tp_loss, _ = self.loss_manager.compute_total_loss(
            self.tp_model,
            {k: v.clone() for k, v in test_data.items()},
            {k: v.clone() for k, v in target_data.items()},
            0,
            100,
        )
        tp_loss.backward()
        average_replicated_grads(self.tp_model)  # Manually average replicated grads

        # Compare gradients
        single_grads = {name: p.grad for name, p in self.single_model.named_parameters() if p.grad is not None}
        tp_local_grads = {name: p.grad for name, p in self.tp_model.named_parameters() if p.grad is not None}

        if self.rank == 0:
            print(f"Comparing {len(single_grads)} gradients...")

        for name, single_grad in single_grads.items():
            assert name in tp_local_grads, f"Gradient for '{name}' missing in TP model."

            local_grad = tp_local_grads[name]
            is_sharded = not getattr(self.tp_model.get_parameter(name), "_is_replicated", False)

            full_tp_grad = None
            if is_sharded:
                partition_dim = 0 if "encoder" in name else 1
                full_tp_grad = gather_sharded_gradient(local_grad, single_grad.shape, partition_dim)
            else:  # Replicated
                full_tp_grad = local_grad

            if self.rank == 0:
                assert full_tp_grad is not None

                # Use a slightly higher tolerance for the jumprelu threshold gradient
                # due to minor floating point differences in its custom autograd function.
                atol = 1e-4 if "log_threshold" in name else 1e-5

                if not torch.allclose(single_grad, full_tp_grad, atol=atol):
                    raise AssertionError(
                        f"Gradient mismatch for '{name}'. Max diff: {(single_grad - full_tp_grad).abs().max()}"
                    )

        if self.rank == 0:
            print("✅ Gradient test passed.")

        if dist.is_initialized():
            dist.barrier()


# --- Pytest Test Functions ---

# Use a common set of configurations to test against
TEST_CONFIGS = [
    {"activation_fn": "relu"},
    {"activation_fn": "jumprelu"},
    {"activation_fn": "batchtopk", "batchtopk_k": 16},
    {"activation_fn": "topk", "topk_k": 0.25},
]


def _run_all_tests_for_config(rank: int, world_size: int, config_params: Dict[str, Any]):
    """The actual test logic run by each process."""
    tester = EquivalenceTester(rank, world_size, config_params)
    tester.run_column_parallel_verification()
    tester.run_row_parallel_verification()
    tester.run_forward_pass_test()
    tester.run_loss_calculation_test()
    tester.run_gradient_test()


@pytest.mark.parametrize("config_params", TEST_CONFIGS)
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_tp_equivalence(config_params: Dict[str, Any]):
    """
    Spawns distributed processes to test tensor parallel model equivalence.
    """
    world_size = 2  # Test with 2 GPUs/processes
    port = "29501"  # Port for this test run

    # Using torch.multiprocessing.spawn to run the test in parallel
    mp.spawn(  # type: ignore
        distributed_test_runner,
        args=(world_size, port, _run_all_tests_for_config, config_params),
        nprocs=world_size,
        join=True,
    )
