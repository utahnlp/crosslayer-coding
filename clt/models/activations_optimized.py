"""Optimized activation functions for better performance."""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OptimizedBatchTopK(torch.autograd.Function):
    """Optimized BatchTopK with fused operations and better memory usage."""
    
    @staticmethod
    def _compute_mask_optimized(
        x: torch.Tensor, 
        k_per_token: int, 
        x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized mask computation with fewer allocations."""
        B = x.size(0)
        if k_per_token <= 0:
            return torch.zeros_like(x, dtype=torch.bool)
        
        # Early exit for full selection
        F_total_batch = x.numel()
        if F_total_batch == 0:
            return torch.zeros_like(x, dtype=torch.bool)
        
        k_total_batch = min(k_per_token * B, F_total_batch)
        
        # Use the ranking tensor if provided, otherwise use x
        ranking_tensor = x_for_ranking if x_for_ranking is not None else x
        
        # Fused reshape and topk - avoid intermediate allocations
        if k_total_batch > 0:
            # Get top-k values and indices in one operation
            _, flat_indices = torch.topk(
                ranking_tensor.view(-1), 
                k_total_batch, 
                sorted=False,
                largest=True
            )
            
            # Create mask directly without intermediate tensor
            mask = torch.zeros(F_total_batch, dtype=torch.bool, device=x.device)
            mask[flat_indices] = True
            return mask.view_as(x)
        else:
            return torch.zeros_like(x, dtype=torch.bool)
    
    @staticmethod 
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, 
        x: torch.Tensor, 
        k: float, 
        straight_through: bool,
        x_for_ranking: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward with mixed precision support."""
        k_per_token = int(k)
        
        # Compute mask in FP32 for accuracy
        with torch.cuda.amp.autocast(enabled=False):
            mask = OptimizedBatchTopK._compute_mask_optimized(
                x.float(), k_per_token, 
                x_for_ranking.float() if x_for_ranking is not None else None
            )
        
        ctx.save_for_backward(mask)
        ctx.straight_through = straight_through
        
        # Apply mask in original dtype
        return x * mask.to(x.dtype)
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        """Optimized backward pass."""
        if ctx.straight_through:
            mask, = ctx.saved_tensors
            # Fused multiplication
            grad_input = grad_output * mask.to(grad_output.dtype)
        else:
            mask, = ctx.saved_tensors
            grad_input = grad_output * mask.to(grad_output.dtype)
        
        return grad_input, None, None, None


def create_optimized_topk_mask_batched(
    concatenated_tensor: torch.Tensor,
    k_values: Dict[int, int],
    layer_sizes: list[tuple[int, int]]
) -> torch.Tensor:
    """Create masks for different layers in parallel when they have different k values."""
    device = concatenated_tensor.device
    dtype = concatenated_tensor.dtype
    batch_size, total_features = concatenated_tensor.shape
    
    # Pre-allocate output mask
    mask = torch.zeros_like(concatenated_tensor, dtype=torch.bool)
    
    # Group layers by k value for batch processing
    k_groups = {}
    for layer_idx, (start_idx, num_features) in enumerate(layer_sizes):
        k_val = k_values.get(layer_idx, 0)
        if k_val not in k_groups:
            k_groups[k_val] = []
        k_groups[k_val].append((layer_idx, start_idx, num_features))
    
    # Process each k-value group
    for k_val, layer_infos in k_groups.items():
        if k_val <= 0:
            continue
            
        # Gather all features for this k value
        indices = []
        for _, start_idx, num_features in layer_infos:
            indices.extend(range(start_idx, start_idx + num_features))
        
        if not indices:
            continue
            
        # Extract relevant features
        group_features = concatenated_tensor[:, indices]
        
        # Compute top-k for this group
        k_total = min(k_val * batch_size, group_features.numel())
        if k_total > 0:
            _, top_indices = torch.topk(
                group_features.view(-1),
                k_total,
                sorted=False
            )
            
            # Convert back to 2D indices
            row_indices = top_indices // len(indices)
            col_indices = top_indices % len(indices)
            
            # Map back to original positions
            for i, (row, col) in enumerate(zip(row_indices, col_indices)):
                original_col = indices[col]
                mask[row, original_col] = True
    
    return mask


# Monkey patch for torch.compile compatibility
def make_compile_compatible():
    """Make activation functions compatible with torch.compile."""
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            # Register custom ops for better compilation
            torch.fx.wrap('OptimizedBatchTopK._compute_mask_optimized')
    except Exception as e:
        logger.debug(f"torch.compile compatibility setup skipped: {e}")
        

# Initialize on module load
make_compile_compatible()