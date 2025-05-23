import torch
import torch.distributed as dist
import os
from typing import Dict, Optional, Literal

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


# Helper to initialize distributed environment for the test
def setup_distributed_test(rank, world_size, master_port="12355"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)  # Set device for this process


# Helper to cleanup distributed environment
def cleanup_distributed_test():
    dist.destroy_process_group()


def run_forward_pass_test(
    rank, world_size, activation_fn: Literal["jumprelu", "relu", "batchtopk", "topk"], batchtopk_k: Optional[int] = None
):
    setup_distributed_test(rank, world_size)

    d_model = 64  # Small d_model for testing
    num_features_per_layer = d_model * 2
    num_layers = 2  # Small number of layers
    batch_size = 4
    seq_len = 8
    batch_tokens = batch_size * seq_len

    clt_config = CLTConfig(
        d_model=d_model,
        num_features=num_features_per_layer,
        num_layers=num_layers,
        activation_fn=activation_fn,
        batchtopk_k=batchtopk_k,
        # jumprelu_threshold is only relevant if activation_fn is jumprelu
        jumprelu_threshold=0.01 if activation_fn == "jumprelu" else 0.0,
    )

    # Determine device for the model based on availability and rank
    current_device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Instantiate model - process_group is automatically handled by CrossLayerTranscoder init if dist is initialized
    model = CrossLayerTranscoder(config=clt_config, process_group=None, device=current_device)  # PG is WORLD implicitly
    model.to(current_device)
    model.eval()  # Set to eval mode

    # Create identical dummy input data on all ranks
    # (batch_tokens, d_model)
    dummy_inputs: Dict[int, torch.Tensor] = {}
    for i in range(num_layers):
        # Ensure identical tensor across ranks using a fixed seed before creating tensor
        torch.manual_seed(42 + i)  # Same seed for each layer across ranks
        dummy_inputs[i] = torch.randn(batch_tokens, d_model, device=current_device, dtype=model.dtype)

    # Perform forward pass
    reconstructions = model.forward(dummy_inputs)

    # Assertions
    assert isinstance(reconstructions, dict)
    assert len(reconstructions) == num_layers

    # Gather all reconstruction tensors to rank 0 for comparison (if more than 1 GPU)
    # Or, more simply, each rank asserts its output is identical to a tensor broadcast from rank 0
    for layer_idx in range(num_layers):
        output_tensor = reconstructions[layer_idx]
        assert output_tensor.shape == (batch_tokens, d_model)
        assert output_tensor.device == current_device
        assert output_tensor.dtype == model.dtype

        # All-reduce the sum of the tensor and sum of squares. If identical, these will be world_size * val.
        # This is a robust way to check for numerical identity across ranks.
        sum_val = output_tensor.sum()
        sum_sq_val = (output_tensor**2).sum()

        gathered_sum_list = [torch.zeros_like(sum_val) for _ in range(world_size)]
        gathered_sum_sq_list = [torch.zeros_like(sum_sq_val) for _ in range(world_size)]

        if world_size > 1:
            dist.all_gather(gathered_sum_list, sum_val)
            dist.all_gather(gathered_sum_sq_list, sum_sq_val)
        else:
            gathered_sum_list = [sum_val]
            gathered_sum_sq_list = [sum_sq_val]

        # Check if all gathered sums and sum_sq are close to each other
        for i in range(1, world_size):
            assert torch.allclose(
                gathered_sum_list[0], gathered_sum_list[i]
            ), f"Rank {rank} Layer {layer_idx} sum mismatch: {gathered_sum_list[0]} vs {gathered_sum_list[i]} (rank {i}) for act_fn {activation_fn}"
            assert torch.allclose(
                gathered_sum_sq_list[0], gathered_sum_sq_list[i]
            ), f"Rank {rank} Layer {layer_idx} sum_sq mismatch: {gathered_sum_sq_list[0]} vs {gathered_sum_sq_list[i]} (rank {i}) for act_fn {activation_fn}"

    if rank == 0:
        print(f"Distributed forward test PASSED for rank {rank}, activation_fn='{activation_fn}'")

    cleanup_distributed_test()


# Main test execution controlled by torchrun
if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    print(f"Starting distributed test on rank {rank} of {world_size}")

    # Test with ReLU
    print(f"Rank {rank}: Running test for ReLU")
    run_forward_pass_test(rank, world_size, activation_fn="relu")
    if world_size > 1:
        dist.barrier()  # Ensure test finishes before next one

    # Test with BatchTopK
    print(f"Rank {rank}: Running test for BatchTopK")
    run_forward_pass_test(rank, world_size, activation_fn="batchtopk", batchtopk_k=10)
    if world_size > 1:
        dist.barrier()

    # Add more activation functions to test if needed, e.g., jumprelu
    # print(f"Rank {rank}: Running test for JumpReLU")
    # run_forward_pass_test(rank, world_size, activation_fn="jumprelu")
    # if world_size > 1: dist.barrier()

    if rank == 0:
        print("All distributed forward tests completed.")
