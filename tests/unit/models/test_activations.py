import torch

# import pytest # DELETED: Not strictly needed for these torch.asserts
from clt.models.activations import BatchTopK


def test_batchtopk_forward_global_k():
    """Test BatchTopK forward pass with global k selection."""
    # Input tensor: 2 samples (tokens), 10 features each
    # Batch size B = 2, Total features F_total = 10
    x = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 1.0],  # Token 1
            [1.1, 0.15, 1.2, 0.25, 1.3, 0.35, 1.4, 0.45, 1.5, 0.05],  # Token 2
        ],
        dtype=torch.float32,
    )

    # Case 1: k_per_token = 1. Should keep 1*2 = 2 features globally.
    # Expected: 1.5 (from token 2, index 8) and 1.4 (from token 2, index 6)
    k1 = 1
    output1 = BatchTopK.apply(x, float(k1), True, None)
    # assert output1[0].eq(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).all() # This was too broad, depends on specific top-k
    # assert output1[1].eq(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.3, 0.0, 1.4, 0.0, 1.5, 0.0])).all() # This was too broad
    assert torch.count_nonzero(output1) == k1 * x.size(0)
    # More specific check for values (will need to adjust expected based on actual top-k logic)
    # For k_total_batch = 2, top values in flattened x are 1.5 (idx 18), 1.4 (idx 16)
    # So, output1 should have non-zero at x[1,8] and x[1,6]
    # expected_output1 = torch.tensor([ # DELETED: This was for per-token topk
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 1.5, 0.0] # Assuming 1.5 and 1.4 are top 2
    # ], dtype=torch.float32)
    expected_output1_corrected = torch.zeros_like(x)
    expected_output1_corrected[1, 8] = 1.5
    expected_output1_corrected[1, 6] = 1.4
    assert torch.allclose(output1, expected_output1_corrected)

    # Case 2: k_per_token = 3. Should keep 3*2 = 6 features globally.
    k2 = 3
    output2 = BatchTopK.apply(x, float(k2), True, None)
    assert torch.count_nonzero(output2) == k2 * x.size(0)
    # Top 6 values: 1.5, 1.4, 1.3, 1.2, 1.1 (from token 2) and 1.0 (from token 1)
    # Indices in flattened x: 1.5 (18), 1.4 (16), 1.3 (14), 1.2 (12), 1.1 (10), 1.0 (9)
    expected_output2 = torch.zeros_like(x)
    expected_output2[1, 8] = 1.5
    expected_output2[1, 6] = 1.4
    expected_output2[1, 4] = 1.3
    expected_output2[1, 2] = 1.2
    expected_output2[1, 0] = 1.1
    expected_output2[0, 9] = 1.0
    assert torch.allclose(output2, expected_output2)

    # Case 3: k_per_token = 0. Should keep 0 features globally.
    k3 = 0
    output3 = BatchTopK.apply(x, float(k3), True, None)
    assert torch.count_nonzero(output3) == 0
    assert torch.allclose(output3, torch.zeros_like(x))

    # Case 4: k_per_token such that k_total_batch > F_total_batch (all features kept)
    k4 = 25  # k_per_token * B = 50, F_total_batch = 20
    output4 = BatchTopK.apply(x, float(k4), True, None)
    assert torch.count_nonzero(output4) == x.numel()
    assert torch.allclose(output4, x)

    # Case 5: Empty input tensor
    x_empty = torch.empty((0, 5), dtype=torch.float32)
    output_empty = BatchTopK.apply(x_empty, float(k1), True, None)
    assert output_empty.numel() == 0
    assert output_empty.shape == x_empty.shape

    # Case 6: Using x_for_ranking
    x_rank = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Token 1 - 10.0 is highest
            [0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Token 2 - 20.0 is second highest
        ],
        dtype=torch.float32,
    )
    # k_per_token = 1, so k_total_batch = 2.
    # Ranking tensor has 20.0 at (1,2) and 10.0 at (0,0) as top 2.
    # Values should come from original x at these positions.
    output_x_rank = BatchTopK.apply(x, float(k1), True, x_rank)
    expected_output_x_rank = torch.zeros_like(x)
    expected_output_x_rank[1, 2] = x[1, 2]  # Value is 1.2
    expected_output_x_rank[0, 0] = x[0, 0]  # Value is 0.1
    assert torch.allclose(output_x_rank, expected_output_x_rank)
    assert torch.count_nonzero(output_x_rank) == k1 * x.size(0)


def test_batchtopk_backward_ste():
    """Test BatchTopK backward pass with straight-through estimator."""
    x = torch.randn(2, 5, requires_grad=True)
    k = 2  # k_per_token

    # Forward pass
    output = BatchTopK.apply(x, float(k), True, None)

    # Create a gradient for the output
    grad_output = torch.randn_like(output)

    # Backward pass
    output.backward(grad_output)

    # Expected gradient: grad_output where mask is True, 0 otherwise
    # The mask is (output != 0)
    mask = (output != 0).to(grad_output.dtype)
    expected_grad_input = grad_output * mask

    assert x.grad is not None
    assert torch.allclose(x.grad, expected_grad_input)


# It might be useful to test the non-STE case if a different backward pass is implemented in the future.
# For now, non-STE backward is the same as STE.
def test_batchtopk_backward_no_ste():
    """Test BatchTopK backward pass with straight_through=False."""
    x = torch.randn(2, 5, requires_grad=True)
    k = 2  # k_per_token

    # Forward pass
    output = BatchTopK.apply(x, float(k), False, None)  # straight_through = False

    # Create a gradient for the output
    grad_output = torch.randn_like(output)

    # Backward pass
    output.backward(grad_output)

    # Expected gradient for current implementation (same as STE):
    mask = (output != 0).to(grad_output.dtype)
    expected_grad_input = grad_output * mask

    assert x.grad is not None
    assert torch.allclose(x.grad, expected_grad_input)
