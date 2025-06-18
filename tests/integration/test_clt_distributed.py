import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from typing import cast

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def setup_distributed_environment(rank, world_size, port="12356"):
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


# --- Test Functions (to be run in separate processes) ---


def _test_forward_pass_distributed(rank, world_size):
    """
    Tests that the forward pass produces identical results on all ranks.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)  # Ensure same model initialization

    clt_config = CLTConfig(num_layers=2, d_model=8, num_features=16, activation_fn="relu")
    model = CrossLayerTranscoder(config=clt_config, process_group=dist.group.WORLD, device=device)

    # All ranks get the same input
    torch.manual_seed(123)
    sample_inputs = {
        0: torch.randn(20, clt_config.d_model, device=device),
        1: torch.randn(20, clt_config.d_model, device=device),
    }

    reconstructions = model.forward(sample_inputs)
    loss = torch.mean(reconstructions[0])  # A simple, deterministic loss

    # Gather the loss from all ranks
    loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(loss_list, loss)

    # The loss, and therefore the forward pass result, should be identical on all ranks
    for other_loss in loss_list:
        assert torch.allclose(loss, other_loss), "Forward pass results (losses) differ across ranks"


def _test_sharded_gradient(rank, world_size):
    """
    Tests that sharded parameters receive different gradients on each rank.
    """
    device = torch.device("cpu")
    # Use rank-specific seed for weight initialization to ensure different weights
    torch.manual_seed(42 + rank)

    clt_config = CLTConfig(num_layers=2, d_model=8, num_features=16, activation_fn="relu")
    model = CrossLayerTranscoder(config=clt_config, process_group=dist.group.WORLD, device=device)

    # All ranks get the same input
    torch.manual_seed(123)
    sample_inputs = {0: torch.randn(5, clt_config.d_model, device=device)}

    # Forward pass
    reconstructions = model.forward(sample_inputs)

    # Create a loss that depends on the actual output values
    # This will produce different gradients for different weight values
    target = torch.randn_like(reconstructions[0])
    loss = torch.nn.functional.mse_loss(reconstructions[0], target)

    # Backward pass
    loss.backward()

    # Test gradients of a SHARDED parameter (e.g., Encoder weights)
    sharded_grad_optional = model.encoder_module.encoders[0].weight.grad
    assert sharded_grad_optional is not None, "Gradient for sharded parameter should exist"
    sharded_grad = cast(torch.Tensor, sharded_grad_optional)

    # Gather all gradients to compare
    grad_list = [torch.zeros_like(sharded_grad) for _ in range(world_size)]
    dist.all_gather(grad_list, sharded_grad)

    # The gradients for a sharded parameter should be DIFFERENT on each rank
    # because each rank has different weights and computes different outputs
    assert not torch.allclose(
        grad_list[0], grad_list[1], rtol=1e-5, atol=1e-8
    ), "Gradients for sharded parameters should be different across ranks"


# --- Pytest Test Class ---


@pytest.mark.integration
@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
class TestCLTDistributed:
    def test_forward_pass(self):
        world_size = 2
        mp.spawn(  # type: ignore[attr-defined]
            distributed_test_runner,
            args=(world_size, _test_forward_pass_distributed),
            nprocs=world_size,
            join=True,
        )

    def test_gradient_sharding(self):
        world_size = 2
        mp.spawn(  # type: ignore[attr-defined]
            distributed_test_runner,
            args=(world_size, _test_sharded_gradient),
            nprocs=world_size,
            join=True,
        )
