import torch
from typing import Optional, Tuple


class BatchTopK(torch.autograd.Function):
    @staticmethod
    def _compute_mask(x: torch.Tensor, k_per_token: int, x_for_ranking: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper static method to compute the BatchTopK mask."""
        B = x.size(0)
        if k_per_token <= 0:
            return torch.zeros_like(x, dtype=torch.bool)

        F_total_batch = x.numel()
        if F_total_batch == 0:
            return torch.zeros_like(x, dtype=torch.bool)

        k_total_batch = min(k_per_token * B, F_total_batch)

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x
        x_flat = x.reshape(-1)
        ranking_flat = ranking_tensor_to_use.reshape(-1)

        if k_total_batch > 0:
            _, flat_indices = torch.topk(ranking_flat, k_total_batch, sorted=False)
            mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
            mask_flat[flat_indices] = True
            mask = mask_flat.view_as(x)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)

        return mask

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, k: float, straight_through: bool, x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies BatchTopK activation.

        Args:
            x: Input tensor of shape [B, F_total] or [B*S, F_total].
            k: The number of top elements to keep. If k < 1, it's treated as a fraction.
            straight_through: Whether to use straight-through estimator for gradients.
            x_for_ranking: Optional tensor to use for ranking. If None, x is used.

        Returns:
            Output tensor with BatchTopK applied.
        """
        k_per_token = int(k)
        mask = BatchTopK._compute_mask(x, k_per_token, x_for_ranking)

        if straight_through:
            ctx.save_for_backward(mask)
        else:
            # For non-STE, save original input and mask for more complex gradient (e.g. Gumbel-Softmax style, not implemented here)
            # ctx.save_for_backward(x, mask) # Placeholder for more complex backward
            ctx.save_for_backward(mask)  # For now, non-STE will also behave like STE in backward

        ctx.straight_through = straight_through

        return x * mask.to(x.dtype)

    @staticmethod
    def backward(ctx, *args: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        """Backward pass for BatchTopK.

        Args:
            grad_output: Gradient from the subsequent layer.

        Returns:
            Gradients for input x, k, straight_through, and x_for_ranking.
        """
        # Expecting only one gradient output tensor
        if len(args) != 1:
            raise ValueError(f"BatchTopK backward expected 1 gradient tensor, got {len(args)}")
        grad_output = args[0]

        if ctx.straight_through:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
        else:
            # Placeholder for a potentially different backward pass if not using STE
            # For now, treat non-STE as STE in backward to avoid errors.
            # A more sophisticated backward (e.g., for Gumbel-Softmax) would go here.
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
            # Example of a different backward (not used now):
            # x_original, mask = ctx.saved_tensors
            # grad_input = grad_output * (some_function_of_x_original_and_mask)
            # This is just illustrative.

        return grad_input, None, None, None  # Gradients for k, straight_through, x_for_ranking are None


class TokenTopK(torch.autograd.Function):
    @staticmethod
    def _compute_mask(x: torch.Tensor, k_float: float, x_for_ranking: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper static method to compute the TokenTopK mask."""
        B_tokens, F_total = x.shape

        if F_total == 0:
            return torch.zeros_like(x, dtype=torch.bool)

        k_per_token: int
        if 0 < k_float < 1:
            k_per_token = int(torch.ceil(torch.tensor(k_float * F_total)).item())
        elif k_float >= 1:
            k_per_token = int(k_float)
        else:  # k <= 0
            return torch.zeros_like(x, dtype=torch.bool)

        k_per_token = min(k_per_token, F_total)

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x

        if k_per_token > 0:
            _, topk_indices_per_row = torch.topk(ranking_tensor_to_use, k_per_token, dim=-1, sorted=False)
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(-1, topk_indices_per_row, True)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)

        return mask

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, k: float, straight_through: bool, x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies TokenTopK activation (TopK per token/row).

        Args:
            x: Input tensor of shape [B_tokens, F_total].
            k: The number or fraction of top elements to keep per token.
               If k < 1, it's treated as a fraction of F_total.
               If k >= 1, it's treated as an integer count.
            straight_through: Whether to use straight-through estimator for gradients.
            x_for_ranking: Optional tensor to use for ranking. If None, x is used.

        Returns:
            Output tensor with TokenTopK applied.
        """
        mask = TokenTopK._compute_mask(x, k, x_for_ranking)

        if straight_through:
            ctx.save_for_backward(mask)
        else:
            ctx.save_for_backward(mask)  # For now, non-STE also behaves like STE in backward

        ctx.straight_through = straight_through

        return x * mask.to(x.dtype)

    @staticmethod
    def backward(ctx, *args: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        """Backward pass for TokenTopK (identical to BatchTopK's STE path)."""
        if len(args) != 1:
            raise ValueError(f"TokenTopK backward expected 1 gradient tensor, got {len(args)}")
        grad_output = args[0]

        if ctx.straight_through:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
        else:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)

        return grad_input, None, None, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(input, threshold)
        ctx.bandwidth = bandwidth
        return (input >= threshold).float() * input

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        grad_output = grad_outputs[0]
        input, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        needs_input_grad, needs_threshold_grad, _ = ctx.needs_input_grad

        grad_input = None
        grad_threshold = None

        if needs_input_grad:
            ste_mask = (input >= threshold).type_as(grad_output)
            grad_input = grad_output * ste_mask

        if needs_threshold_grad:
            is_near_threshold = torch.abs(input - threshold) <= (bandwidth / 2.0)
            local_grad_theta = (-input / bandwidth) * is_near_threshold.type_as(input)
            grad_threshold_per_element = grad_output * local_grad_theta

            if grad_threshold_per_element.dim() > threshold.dim():
                dims_to_sum = tuple(range(grad_threshold_per_element.dim() - threshold.dim()))
                grad_threshold = grad_threshold_per_element.sum(dim=dims_to_sum)
                if threshold.shape != torch.Size([]):
                    grad_threshold = grad_threshold.reshape(threshold.shape)
            else:
                grad_threshold = grad_threshold_per_element.sum()
        return grad_input, grad_threshold, None
