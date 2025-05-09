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
        if k < 1:
            # k is a fraction, convert to an absolute number of features
            k_abs = max(1, int(round(k * x.size(-1))))
        else:
            k_abs = int(k)

        # Ensure k is not larger than the number of features
        k_abs = min(k_abs, x.size(-1))

        ranking_tensor_to_use = x_for_ranking if x_for_ranking is not None else x

        if k_abs == 0 and x.size(-1) > 0:  # Avoid issues with topk if k_abs is 0 but features exist
            # If k_abs is 0, return a zero tensor of the same shape and type as x
            if straight_through:
                ctx.save_for_backward(torch.zeros_like(x, dtype=torch.bool))
            return torch.zeros_like(x)

        # Handle case where ranking tensor might be empty but k_abs > 0 (e.g. if x_for_ranking was bad)
        if k_abs > 0 and ranking_tensor_to_use.numel() == 0:
            if x.numel() > 0:  # If x has elements, use x for ranking as a fallback
                ranking_tensor_to_use = x
            else:  # Both x and ranking_tensor are empty, return zeros
                if straight_through:
                    ctx.save_for_backward(torch.zeros_like(x, dtype=torch.bool))
                return torch.zeros_like(x)

        indices: Optional[torch.Tensor] = None
        if k_abs > 0:  # Only compute topk if k_abs > 0 and ranking tensor has elements
            if ranking_tensor_to_use.numel() > 0:
                _, indices = torch.topk(ranking_tensor_to_use, k_abs, dim=-1, sorted=False)
            else:  # ranking_tensor is empty, cannot compute topk, treat as k_abs = 0
                k_abs = 0  # effectively
                indices = torch.empty(
                    (ranking_tensor_to_use.size(0), 0), dtype=torch.long, device=ranking_tensor_to_use.device
                )
        elif x.size(-1) == 0:  # k_abs is 0 and features are empty
            indices = torch.empty(
                (ranking_tensor_to_use.size(0), 0), dtype=torch.long, device=ranking_tensor_to_use.device
            )
        # else: k_abs is 0 and x.size(-1) > 0, handled by the first conditional block returning zeros.

        # Create a mask from the indices, based on the shape of x
        mask = torch.zeros_like(x, dtype=torch.bool)
        if indices is not None and indices.numel() > 0 and k_abs > 0:  # only scatter if there are indices and k_abs > 0
            mask.scatter_(-1, indices, True)

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
