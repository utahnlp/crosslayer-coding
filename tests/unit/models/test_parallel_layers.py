import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from clt.models.parallel import ColumnParallelLinear, RowParallelLinear


def setup_distributed_environment(rank, world_size, port="12355"):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed_environment():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def distributed_test_runner(rank, world_size, test_fn, *args):
    """A wrapper to run a distributed test function."""
    setup_distributed_environment(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        cleanup_distributed_environment()


# --- Non-Distributed Tests (World Size = 1) ---


class TestParallelLayersNonDistributed:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_column_parallel_linear_forward(self, device):
        """Test ColumnParallelLinear forward pass without distribution."""
        layer = ColumnParallelLinear(in_features=10, out_features=20, process_group=None, device=device)
        input_tensor = torch.randn(5, 10, device=device)
        output = layer(input_tensor)
        assert output.shape == (5, 20)
        assert output.device.type == device.type

    def test_row_parallel_linear_forward(self, device):
        """Test RowParallelLinear forward pass without distribution."""
        layer = RowParallelLinear(
            in_features=10,
            out_features=20,
            process_group=None,
            d_model_for_init=20,
            num_layers_for_init=1,
            device=device,
        )
        input_tensor = torch.randn(5, 10, device=device)
        output = layer(input_tensor)
        assert output.shape == (5, 20)
        assert output.device.type == device.type


# --- Distributed Tests (World Size = 2) ---


# This function will be run in each process
def _test_column_parallel_distributed_forward(rank, world_size):
    device = torch.device("cpu")
    in_features, out_features, batch_size = 10, 20, 5

    # Each rank has the same seed for weights
    torch.manual_seed(42)
    layer = ColumnParallelLinear(in_features, out_features, process_group=dist.group.WORLD, device=device)

    # Each rank gets the full input tensor
    torch.manual_seed(123)
    input_tensor = torch.randn(batch_size, in_features, device=device)

    output = layer(input_tensor)

    # Output should be gathered and identical on both ranks
    assert output.shape == (batch_size, out_features)

    # A more robust check would involve gathering the full weight matrix.
    # This requires gathering the full weight matrix manually.
    weight_slices = [torch.zeros_like(layer.weight) for _ in range(world_size)]
    dist.all_gather(weight_slices, layer.weight.data)
    full_weight = torch.cat(weight_slices, dim=0)

    full_bias = torch.zeros(out_features, device=device)
    if layer.bias_param is not None:
        bias_slices = [torch.zeros_like(layer.bias_param) for _ in range(world_size)]
        dist.all_gather(bias_slices, layer.bias_param.data)
        full_bias = torch.cat(bias_slices, dim=0)

    manual_output = torch.matmul(input_tensor, full_weight.t()) + full_bias
    torch.testing.assert_close(output, manual_output, rtol=1e-4, atol=1e-5)


def _test_row_parallel_distributed_forward(rank, world_size):
    device = torch.device("cpu")
    in_features, out_features, batch_size = 10, 20, 5

    torch.manual_seed(42)
    layer = RowParallelLinear(
        in_features,
        out_features,
        process_group=dist.group.WORLD,
        d_model_for_init=out_features,
        num_layers_for_init=1,
        device=device,
    )

    torch.manual_seed(123)
    full_input = torch.randn(batch_size, in_features, device=device)

    # Manually split input for each rank
    local_in_features = layer.local_in_features
    # We manually create the slice that this rank is supposed to receive.
    # The `input_is_parallel` flag on the layer is True by default.
    input_slice = full_input[:, rank * local_in_features : (rank + 1) * local_in_features]

    # To correctly test against a reference, we need to gather the sharded weights
    # into a full weight matrix on each rank.
    weight_slices = [torch.zeros_like(layer.weight) for _ in range(world_size)]
    dist.all_gather(weight_slices, layer.weight.data)
    full_weight_manual = torch.cat(weight_slices, dim=1)

    # The bias in RowParallelLinear is replicated, not sharded.
    # The reference calculation should use the layer's actual bias.
    manual_output = torch.matmul(full_input, full_weight_manual.t())
    if layer.bias_param is not None:
        manual_output += layer.bias_param

    # Pass the pre-sharded input slice to the layer
    output = layer(input_slice)

    torch.testing.assert_close(output, manual_output, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
class TestParallelLayersDistributed:
    def test_column_parallel(self):
        world_size = 2
        mp.spawn(
            distributed_test_runner,
            args=(world_size, _test_column_parallel_distributed_forward),
            nprocs=world_size,
            join=True,
        )

    def test_row_parallel(self):
        world_size = 2
        mp.spawn(
            distributed_test_runner,
            args=(world_size, _test_row_parallel_distributed_forward),
            nprocs=world_size,
            join=True,
        )
