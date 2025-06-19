import torch
import pytest
from typing import cast

from clt.models.activations import BatchTopK, TokenTopK, JumpReLU

# --- BatchTopK Tests ---


def test_batchtopk_compute_mask_basic():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.5, 3.5]])  # Batch 2, Features 4
    k_per_token = 2
    mask = BatchTopK._compute_mask(x, k_per_token)
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == k_per_token * x.size(0)  # k_total_batch = 2 * 2 = 4

    # Check if top k elements per batch are selected (overall)
    # Expected: 4.0, 3.0 from row 1; 5.0, 3.5 from row 2.
    # Mask should pick [F, T, T, T] for row 1 (if 3,4 selected) and [T, F, F, T] for row 2 (if 5, 3.5 selected)
    # Flattened x: [1,2,3,4,5,1,2.5,3.5], top 4: [4,5,3.5,3]
    # Indices: 3, 4, 7, 2
    expected_mask_flat = torch.zeros_like(x.view(-1), dtype=torch.bool)
    expected_mask_flat[torch.tensor([3, 4, 7, 2])] = True
    assert torch.equal(mask.view(-1), expected_mask_flat)


def test_batchtopk_compute_mask_k_zero():
    x = torch.randn(2, 4)
    k_per_token = 0
    mask = BatchTopK._compute_mask(x, k_per_token)
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 0


def test_batchtopk_compute_mask_k_full():
    x = torch.randn(2, 4)
    k_per_token = 4
    mask = BatchTopK._compute_mask(x, k_per_token)
    assert mask.sum().item() == x.numel()
    assert torch.all(mask)


def test_batchtopk_compute_mask_k_more_than_features():
    x = torch.randn(2, 4)
    k_per_token = 5
    mask = BatchTopK._compute_mask(x, k_per_token)  # k_total_batch = min(5*2, 2*4) = 8
    assert mask.sum().item() == x.numel()
    assert torch.all(mask)


def test_batchtopk_compute_mask_empty_input():
    x = torch.empty(0, 4)
    k_per_token = 2
    mask = BatchTopK._compute_mask(x, k_per_token)
    assert mask.shape == x.shape
    assert mask.sum().item() == 0

    x2 = torch.empty(2, 0)
    mask2 = BatchTopK._compute_mask(x2, k_per_token)
    assert mask2.shape == x2.shape
    assert mask2.sum().item() == 0


def test_batchtopk_forward_basic():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.5, 3.5]], dtype=torch.float32, requires_grad=True)
    k_float = 2.0  # Treated as int for k_per_token
    straight_through = True

    output = cast(torch.Tensor, BatchTopK.apply(x, k_float, straight_through, None))

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    # Based on test_batchtopk_compute_mask_basic: values 1.0 and 2.5 should be zeroed
    # x_flat indices for 1.0 and 2.5 are 0 and 6
    # mask_flat indices 3,4,7,2 -> original values are 4,5,3.5,3
    # Values at indices 0, 1, 5, 6 should be zeroed
    # Original x: [[1,2,3,4], [5,1,2.5,3.5]]
    # Mask:       [[F,T,T,T], [T,F,F,T]] if x used for ranking (no, this is batch not token)
    # Mask (global): mask_flat[3]=T(4.0), mask_flat[4]=T(5.0), mask_flat[7]=T(3.5), mask_flat[2]=T(3.0)
    # Output should be: [[0,0,3,4], [5,0,0,3.5]]
    # expected_output = torch.tensor([[0.0, 0.0, 3.0, 4.0], [5.0, 0.0, 0.0, 3.5]], dtype=torch.float32)
    # Rerun logic from _compute_mask to confirm mask for this x and k=2
    true_mask = torch.zeros_like(x, dtype=torch.bool)
    true_mask_flat = true_mask.view(-1)
    true_mask_flat[torch.tensor([4, 3, 7, 2])] = True  # Corresponds to 5,4,3.5,3

    assert torch.allclose(output, x * true_mask.to(x.dtype))


def test_batchtopk_forward_with_x_for_ranking():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    x_for_ranking = torch.tensor([[0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    k_float = 2.0
    straight_through = True
    # Ranking based on x_for_ranking, k=2*2=4. Should pick 8,7,6,5 (all from second row of x_for_ranking)
    # This means mask is all False for first row of x, all True for second row of x.
    # Output should be: [[0,0,0,0],[0,0,0,0]] because x_for_ranking makes mask select 2nd row of x, which is all 0
    # Correction: the mask applies to x. So if x_for_ranking leads to selecting indices in the second row,
    # then the *values* from the second row of *x* should be preserved.
    # x_for_ranking flat = [0,0,0,0,5,6,7,8]. Top 4: 8,7,6,5. Indices: 7,6,5,4
    # Mask applied to x: x elements at these flat indices are kept.
    # x_flat = [1,2,3,4,0,0,0,0].
    # Mask: [[F,F,F,F], [T,T,T,T]]
    # Output: [[0,0,0,0], [0,0,0,0]] - This is correct.

    output = cast(torch.Tensor, BatchTopK.apply(x, k_float, straight_through, x_for_ranking))
    expected_mask = torch.tensor([[False, False, False, False], [True, True, True, True]])
    expected_output = x * expected_mask.to(x.dtype)
    assert torch.allclose(output, expected_output)


def test_batchtopk_backward_ste():
    x = torch.randn(2, 4, requires_grad=True)
    k_float = 2.0
    straight_through = True

    # Forward pass to save context
    output = cast(torch.Tensor, BatchTopK.apply(x, k_float, straight_through, None))

    # Dummy gradient from subsequent layer
    grad_output = torch.randn_like(output)

    # Backward pass
    output.backward(grad_output)

    # Check gradients
    assert x.grad is not None

    # Expected gradient for STE: grad_output * mask
    # Recompute mask
    mask = BatchTopK._compute_mask(x.data, int(k_float), None)
    expected_grad_x = grad_output * mask.to(x.dtype)

    assert torch.allclose(x.grad, expected_grad_x)


def test_batchtopk_backward_non_ste_placeholder():
    # Currently, non-STE backward behaves like STE. This test reflects that.
    x = torch.randn(2, 4, requires_grad=True)
    k_float = 2.0
    straight_through = False  # Non-STE

    output = cast(torch.Tensor, BatchTopK.apply(x, k_float, straight_through, None))
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    assert x.grad is not None
    mask = BatchTopK._compute_mask(x.data, int(k_float), None)
    expected_grad_x = grad_output * mask.to(x.dtype)  # Same as STE
    assert torch.allclose(x.grad, expected_grad_x)


# --- TokenTopK Tests ---


def test_tokentopk_compute_mask_basic_fraction_k():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.5, 3.5]])  # Batch 2, Features 4
    k_float = 0.5  # Keep 50% of features per token (0.5 * 4 = 2)
    mask = TokenTopK._compute_mask(x, k_float)
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 2 * x.size(0)  # 2 features per token * 2 tokens = 4

    # Row 0: [1,2,3,4], top 2 are 3,4. Mask: [F,F,T,T]
    # Row 1: [5,1,2.5,3.5], top 2 are 5,3.5. Mask: [T,F,F,T]
    expected_mask = torch.tensor([[False, False, True, True], [True, False, False, True]])
    assert torch.equal(mask, expected_mask)


def test_tokentopk_compute_mask_basic_int_k():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.5, 3.5]])
    k_int = 1  # Keep 1 feature per token
    mask = TokenTopK._compute_mask(x, float(k_int))
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == k_int * x.size(0)

    # Row 0: [1,2,3,4], top 1 is 4. Mask: [F,F,F,T]
    # Row 1: [5,1,2.5,3.5], top 1 is 5. Mask: [T,F,F,F]
    expected_mask = torch.tensor([[False, False, False, True], [True, False, False, False]])
    assert torch.equal(mask, expected_mask)


def test_tokentopk_compute_mask_k_zero():
    x = torch.randn(2, 4)
    k_float = 0.0
    mask = TokenTopK._compute_mask(x, k_float)
    assert mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert mask.sum().item() == 0


def test_tokentopk_compute_mask_k_negative():
    x = torch.randn(2, 4)
    k_float = -0.5
    mask = TokenTopK._compute_mask(x, k_float)
    assert mask.sum().item() == 0


def test_tokentopk_compute_mask_k_fraction_ceil():
    # k_float * F_total = 0.6 * 5 = 3.0, ceil(3.0) = 3
    x = torch.randn(2, 5)
    k_float = 0.6
    mask = TokenTopK._compute_mask(x, k_float)
    assert mask.sum(dim=1).tolist() == [3, 3]

    # k_float * F_total = 0.5 * 5 = 2.5, ceil(2.5) = 3
    x2 = torch.randn(2, 5)
    k_float2 = 0.5
    mask2 = TokenTopK._compute_mask(x2, k_float2)
    assert mask2.sum(dim=1).tolist() == [3, 3]


def test_tokentopk_compute_mask_k_full_fraction():
    x = torch.randn(2, 4)
    k_float = 1.0  # According to TokenTopK logic, this means k_per_token = int(1.0) = 1
    mask = TokenTopK._compute_mask(x, k_float)
    # Expected sum is 1 (k_per_token) * 2 (num_tokens) = 2
    assert mask.sum().item() == 1 * x.size(0)
    # This assert torch.all(mask) will now fail, as only 1 element per row is true.
    # We should check that each row sums to 1.
    assert torch.all(mask.sum(dim=1) == 1)


def test_tokentopk_compute_mask_k_full_int():
    x = torch.randn(2, 4)
    k_int = 4
    mask = TokenTopK._compute_mask(x, float(k_int))
    assert mask.sum().item() == x.numel()
    assert torch.all(mask)


def test_tokentopk_compute_mask_k_more_than_features_int():
    x = torch.randn(2, 4)
    k_int = 5
    mask = TokenTopK._compute_mask(x, float(k_int))
    assert mask.sum().item() == x.numel()
    assert torch.all(mask)


def test_tokentopk_compute_mask_empty_input():
    x = torch.empty(0, 4)
    k_float = 0.5
    mask = TokenTopK._compute_mask(x, k_float)
    assert mask.shape == x.shape
    assert mask.sum().item() == 0

    x2 = torch.empty(2, 0)
    mask2 = TokenTopK._compute_mask(x2, k_float)
    assert mask2.shape == x2.shape
    assert mask2.sum().item() == 0


def test_tokentopk_forward_basic():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 1.0, 2.5, 3.5]], dtype=torch.float32, requires_grad=True)
    k_float = 0.5  # keep 2 per token
    straight_through = True

    output = cast(torch.Tensor, TokenTopK.apply(x, k_float, straight_through, None))

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    expected_mask = torch.tensor([[False, False, True, True], [True, False, False, True]])
    expected_output = x * expected_mask.to(x.dtype)
    assert torch.allclose(output, expected_output)


def test_tokentopk_forward_with_x_for_ranking():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]], dtype=torch.float32, requires_grad=True)
    x_for_ranking = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32
    )  # Ranks inverted for row 0, normal for row 1
    k_float = 0.5  # keep 2
    straight_through = True

    # Row 0 x_for_ranking: top 2 are 4,3 (indices 0,1). Mask on x: [T,T,F,F]. Output from x: [1,2,0,0]
    # Row 1 x_for_ranking: top 2 are 3,4 (indices 2,3). Mask on x: [F,F,T,T]. Output from x: [0,0,10,10]
    output = cast(torch.Tensor, TokenTopK.apply(x, k_float, straight_through, x_for_ranking))
    expected_mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
    expected_output = x * expected_mask.to(x.dtype)
    assert torch.allclose(output, expected_output)


def test_tokentopk_backward_ste():
    x = torch.randn(2, 4, requires_grad=True)
    k_float = 0.5
    straight_through = True

    output = cast(torch.Tensor, TokenTopK.apply(x, k_float, straight_through, None))
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    assert x.grad is not None
    mask = TokenTopK._compute_mask(x.data, k_float, None)
    expected_grad_x = grad_output * mask.to(x.dtype)
    assert torch.allclose(x.grad, expected_grad_x)


# --- JumpReLU Tests ---


@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_jumprelu_forward(bandwidth):
    input_tensor = torch.tensor([-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0], requires_grad=True)
    threshold = torch.tensor([0.5])  # Threshold is 0.5

    # Expected: input values >= 0.5 pass, others zeroed.
    # [-2, -1, -0.4, 0, 0.4, 1, 2] with threshold 0.5 -> [0,0,0,0,0,1,2]
    expected_output = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])

    output = cast(torch.Tensor, JumpReLU.apply(input_tensor, threshold, bandwidth))
    assert torch.allclose(output, expected_output)
    assert output.dtype == input_tensor.dtype


@pytest.mark.parametrize("bandwidth", [1.0, 2.0])  # Bandwidth must be > 0
@pytest.mark.parametrize(
    "input_val, threshold_val, expected_grad_input_factor, expected_grad_thresh_factor",
    [
        # Case 1: input < threshold, outside bandwidth/2 on the left
        (0.0, 1.0, 0.0, 0.0),  # grad_input is STE (0), grad_thresh is 0
        # Case 2: input < threshold, inside bandwidth/2 on the left (input - threshold = -0.2, abs = 0.2 <= bandwidth/2)
        (0.8, 1.0, 0.0, -0.8),  # grad_input is STE (0), grad_thresh is -input/bandwidth (if bandwidth=1, -0.8/1 = -0.8)
        # Case 3: input == threshold (center of bandwidth)
        (1.0, 1.0, 1.0, -1.0),  # grad_input is STE (1), grad_thresh is -input/bandwidth (if bandwidth=1, -1.0/1 = -1.0)
        # Case 4: input > threshold, inside bandwidth/2 on the right (input - threshold = 0.2, abs = 0.2 <= bandwidth/2)
        (1.2, 1.0, 1.0, -1.2),  # grad_input is STE (1), grad_thresh is -input/bandwidth (if bandwidth=1, -1.2/1 = -1.2)
        # Case 5: input > threshold, outside bandwidth/2 on the right
        (2.0, 1.0, 1.0, 0.0),  # grad_input is STE (1), grad_thresh is 0
        # Case 6: Another example for input < threshold, inside bandwidth
        (-0.2, 0.0, 0.0, 0.2),  # input - thresh = -0.2. grad_thresh = -(-0.2)/bandwidth = 0.2/bandwidth
        # Case 7: input significantly less than threshold
        (-5.0, 0.0, 0.0, 0.0),
        # Case 8: input significantly greater than threshold
        (5.0, 0.0, 1.0, 0.0),
    ],
)
def test_jumprelu_backward_detailed(
    bandwidth, input_val, threshold_val, expected_grad_input_factor, expected_grad_thresh_factor
):
    input_t = torch.tensor([input_val], dtype=torch.float64, requires_grad=True)
    # Threshold must also require grad for its grad to be computed and non-None
    threshold_t = torch.tensor([threshold_val], dtype=torch.float64, requires_grad=True)
    grad_output = torch.tensor([1.0], dtype=torch.float64)  # Assume upstream grad is 1 for simplicity

    # Forward pass
    output = cast(torch.Tensor, JumpReLU.apply(input_t, threshold_t, bandwidth))
    # Backward pass
    output.backward(grad_output)

    assert input_t.grad is not None
    assert threshold_t.grad is not None

    # Check grad_input (STE part)
    # ste_mask = (input_t >= threshold_t).float()
    # expected_grad_input = grad_output * ste_mask
    # Using expected_grad_input_factor which is effectively the ste_mask for grad_output=1
    assert torch.allclose(
        input_t.grad, torch.tensor([expected_grad_input_factor * grad_output.item()], dtype=torch.float64)
    ), f"Input: {input_val}, Thresh: {threshold_val}, BW: {bandwidth}\nGrad_input: {input_t.grad.item()}, Expected_factor: {expected_grad_input_factor}"

    # Check grad_threshold
    # is_near_threshold = torch.abs(input_t - threshold_t) <= (bandwidth / 2.0)
    # local_grad_theta = (-input_t / bandwidth) * is_near_threshold.float()
    # expected_grad_threshold = grad_output * local_grad_theta
    # Using expected_grad_thresh_factor which is effectively local_grad_theta for grad_output=1
    # Note: the formula in JumpReLU.backward is (-input / bandwidth) * is_near_threshold
    # So if expected_grad_thresh_factor is -0.8, this implies is_near_threshold=True, and factor = -input_val/bandwidth
    # If expected_grad_thresh_factor is 0.0, implies is_near_threshold=False
    # let's re-evaluate the factor based on the original formula components:
    is_near = abs(input_val - threshold_val) <= (bandwidth / 2.0)
    if is_near:
        true_expected_factor_for_grad_thresh = -input_val / bandwidth
    else:
        true_expected_factor_for_grad_thresh = 0.0

    assert torch.allclose(
        threshold_t.grad, torch.tensor([true_expected_factor_for_grad_thresh * grad_output.item()], dtype=torch.float64)
    ), f"Input: {input_val}, Thresh: {threshold_val}, BW: {bandwidth}\nGrad_thresh: {threshold_t.grad.item()}, Expected_factor_calc: {true_expected_factor_for_grad_thresh}"


def test_jumprelu_backward_grad_flags():
    bandwidth = 1.0
    # Case 1: Only input requires grad
    inp1 = torch.tensor([1.5], requires_grad=True)
    thr1 = torch.tensor([0.5], requires_grad=False)
    out1 = cast(torch.Tensor, JumpReLU.apply(inp1, thr1, bandwidth))
    out1.backward(torch.tensor([1.0]))
    assert inp1.grad is not None
    assert thr1.grad is None

    # Case 2: Only threshold requires grad
    inp2 = torch.tensor([1.5], requires_grad=False)
    thr2 = torch.tensor([0.5], requires_grad=True)
    out2 = cast(torch.Tensor, JumpReLU.apply(inp2, thr2, bandwidth))
    out2.backward(torch.tensor([1.0]))
    assert inp2.grad is None
    assert thr2.grad is not None

    # Case 3: Neither requires grad (should not error, grads just won't be populated)
    inp3 = torch.tensor([1.5], requires_grad=False)
    thr3 = torch.tensor([0.5], requires_grad=False)
    # out3 = JumpReLU.apply(inp3, thr3, bandwidth)
    # .backward() would error if no input needs grad and it's called.
    # We are testing autograd.Function behavior, which populates ctx.needs_input_grad
    # For this, we simulate the call to backward directly

    class MockContext:
        def __init__(self, needs_input_grad_list, saved_tensors_tuple, bw):
            self.needs_input_grad = needs_input_grad_list
            self.saved_tensors = saved_tensors_tuple
            self.bandwidth = bw

    ctx_mock_no_input_grad = MockContext([False, True, False], (inp3.detach(), thr3.detach()), bandwidth)
    grad_in_sim, grad_thr_sim, _ = JumpReLU.backward(ctx_mock_no_input_grad, torch.tensor([1.0]))
    assert grad_in_sim is None
    assert grad_thr_sim is not None  # grad_thresh calculation part should still run if thr needs grad

    ctx_mock_no_thresh_grad = MockContext([True, False, False], (inp3.detach(), thr3.detach()), bandwidth)
    grad_in_sim2, grad_thr_sim2, _ = JumpReLU.backward(ctx_mock_no_thresh_grad, torch.tensor([1.0]))
    assert grad_in_sim2 is not None
    assert grad_thr_sim2 is None

    ctx_mock_no_grads_needed = MockContext([False, False, False], (inp3.detach(), thr3.detach()), bandwidth)
    grad_in_sim3, grad_thr_sim3, _ = JumpReLU.backward(ctx_mock_no_grads_needed, torch.tensor([1.0]))
    assert grad_in_sim3 is None
    assert grad_thr_sim3 is None
