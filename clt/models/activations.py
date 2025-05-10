import torch
from typing import Optional


class BatchTopK(torch.autograd.Function):
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
        B = x.size(0)
        # k is now an integer representing desired features *per token*
        k_per_token = int(k)
        if k_per_token <= 0:
            # If k_per_token is 0 or negative, return a zero tensor
            if straight_through:
                ctx.save_for_backward(torch.zeros_like(x, dtype=torch.bool))
            return torch.zeros_like(x)

        # Total features to keep across the batch
        # Clamp to F_total (x.numel()) to avoid errors if k_per_token * B > F_total
        # Also handle the case where x is empty (x.numel() == 0)
        F_total_batch = x.numel()
        if F_total_batch == 0:  # If input is empty, return empty tensor
            if straight_through:
                ctx.save_for_backward(torch.zeros_like(x, dtype=torch.bool))
            return torch.zeros_like(x)

        k_total_batch = min(k_per_token * B, F_total_batch)

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x

        # Flatten input and ranking tensor for global top-k
        x_flat = x.reshape(-1)  # [B*F_total]
        ranking_flat = ranking_tensor_to_use.reshape(-1)

        indices: Optional[torch.Tensor] = None
        if k_total_batch > 0:
            # Global top-k on the flattened ranking tensor
            _, flat_indices = torch.topk(ranking_flat, k_total_batch, sorted=False)
            mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
            mask_flat[flat_indices] = True
            mask = mask_flat.view_as(x)  # Reshape mask to original x shape
        else:  # k_total_batch is 0 (either k_per_token was 0 or input was empty)
            mask = torch.zeros_like(x, dtype=torch.bool)

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
