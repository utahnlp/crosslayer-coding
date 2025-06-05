import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup
import math
from typing import Callable, Optional, cast, Tuple

from . import mark_replicated
from clt.parallel import ops as dist_ops


class _ParallelLinear(nn.Module):
    """Base class for parallel linear layers."""

    bias_param: Optional[nn.Parameter]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        process_group: Optional[ProcessGroup],  # Allow None for non-distributed
        partition_dim: int,  # 0 for columns (output features), 1 for rows (input features)
        init_method: Callable = nn.init.xavier_uniform_,
        input_is_parallel: bool = False,
        keep_master_weight: bool = False,  # Not used yet, for future optimizations
        d_model_for_init: Optional[int] = None,  # Add d_model for row parallel init
        num_layers_for_init: Optional[int] = None,  # Add num_layers for row parallel init
        device: Optional[torch.device] = None,  # Add device argument
        dtype: Optional[torch.dtype] = None,  # Add dtype argument
    ):
        super().__init__()
        self.process_group = process_group

        # Handle non-distributed case using new utility functions
        self.world_size = dist_ops.get_world_size(process_group)
        self.rank = dist_ops.get_rank(process_group)
        # If world_size is 1, process_group should effectively be None for logic below
        if self.world_size == 1:
            self.process_group = None

        self.partition_dim = partition_dim
        self.input_is_parallel = input_is_parallel
        self.bias = bias
        self.full_in_features = in_features
        self.full_out_features = out_features
        self.dtype = dtype if dtype is not None else torch.float32

        # Calculate local dimensions with uniform padding for divisibility
        if partition_dim == 0:  # Column Parallelism (Output features sharded)
            # Calculate padded size for uniform distribution
            self.local_out_features = math.ceil(out_features / self.world_size)
            self.local_in_features = in_features
            self.weight = nn.Parameter(
                torch.empty(self.local_out_features, self.local_in_features, device=device, dtype=self.dtype)
            )
            if bias:
                self.bias_param = nn.Parameter(torch.empty(self.local_out_features, device=device, dtype=self.dtype))
            else:
                self.bias_param = None  # Explicitly set to None
        elif partition_dim == 1:  # Row Parallelism (Input features sharded)
            # Calculate padded size for uniform distribution
            self.local_in_features = math.ceil(in_features / self.world_size)
            self.local_out_features = out_features
            self.weight = nn.Parameter(
                torch.empty(self.local_out_features, self.local_in_features, device=device, dtype=self.dtype)
            )
            if bias:
                self.bias_param = nn.Parameter(torch.empty(out_features, device=device, dtype=self.dtype))
                mark_replicated(self.bias_param)
            else:
                self.bias_param = None  # Explicitly set to None
        else:
            raise ValueError("partition_dim must be 0 or 1")

        # Initialize weights (ensure consistency across ranks if needed)
        # Default init methods often depend on full shapes. We might need custom init.
        # Simplified init for CLT (matching original CLT init logic)
        if partition_dim == 0:
            bound = 1.0 / math.sqrt(self.full_out_features)
            nn.init.uniform_(self.weight, -bound, bound)
            if bias and self.bias_param is not None:
                nn.init.zeros_(self.bias_param)
        elif partition_dim == 1:
            if d_model_for_init is None or num_layers_for_init is None:
                raise ValueError("d_model_for_init and num_layers_for_init must be provided for RowParallelLinear init")
            bound = 1.0 / math.sqrt(num_layers_for_init * d_model_for_init)
            nn.init.uniform_(self.weight, -bound, bound)
            if bias and self.bias_param is not None:
                nn.init.zeros_(self.bias_param)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _Gather(torch.autograd.Function):
    """Autograd-aware all-gather + concat.

    During the forward pass each rank contributes its *local* slice and the
    concatenated full tensor is returned to every rank.  In the backward pass
    the incoming gradient is **sliced** so that each rank receives the portion
    corresponding to its original contribution.  This mirrors the behaviour of
    a plain :func:`torch.cat` w.r.t. autograd and enables correct gradient
    propagation through the gather.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, process_group: ProcessGroup, dim: int, full_dim_size: Optional[int]):
        # Use new utility functions
        if not dist_ops.is_dist_initialized_and_available() or dist_ops.get_world_size(process_group) == 1:
            ctx.dim = dim
            ctx.local_dim = input_.size(dim)
            ctx.full_dim_size = full_dim_size or input_.size(dim)
            ctx.process_group = None  # Mark non-distributed case
            return input_

        world_size = dist_ops.get_world_size(process_group)
        rank = dist_ops.get_rank(process_group)

        ctx.dim = dim
        ctx.local_dim = input_.size(dim)
        ctx.full_dim_size = full_dim_size if full_dim_size is not None else ctx.local_dim * world_size
        ctx.process_group = process_group

        # Ensure a contiguous tensor before communication for NCCL efficiency.
        input_contig = input_.contiguous()

        # Create gather tensors with explicit memory format and alignment for CUDA 12.8
        gathered: list[torch.Tensor] = []
        for i in range(world_size):
            if i == rank:
                # Use the original contiguous tensor for this rank
                gathered.append(input_contig)
            else:
                # Create empty tensors with same dtype, device, and memory layout
                empty_tensor = torch.empty(
                    input_contig.shape, 
                    dtype=input_contig.dtype, 
                    device=input_contig.device,
                    memory_format=torch.contiguous_format
                )
                gathered.append(empty_tensor)

        # Ensure CUDA synchronization before collective operation for CUDA 12.8
        if input_contig.device.type == 'cuda':
            torch.cuda.synchronize(device=input_contig.device)
        
        # Perform the collective using new utility function wrapper
        dist_ops.all_gather(gathered, input_contig, group=process_group)

        # Create a safe output tensor manually to avoid CUDA 12.8 concat issues
        device = input_contig.device
        dtype = input_contig.dtype
        
        # Calculate output shape
        output_shape = list(input_contig.shape)
        total_dim_size = sum(t.size(dim) for t in gathered)
        output_shape[dim] = total_dim_size
        
        # Create output tensor with explicit memory format
        output = torch.empty(
            output_shape, 
            dtype=dtype, 
            device=device, 
            memory_format=torch.contiguous_format
        )
        
        # Manually copy data instead of using torch.cat
        current_offset = 0
        for i, tensor in enumerate(gathered):
            # Ensure tensor is properly formatted
            safe_tensor = tensor
            if safe_tensor.device != device:
                safe_tensor = safe_tensor.to(device)
            if safe_tensor.dtype != dtype:
                safe_tensor = safe_tensor.to(dtype)
            if not safe_tensor.is_contiguous():
                safe_tensor = safe_tensor.contiguous()
            
            tensor_size = safe_tensor.size(dim)
            if tensor_size > 0:  # Only copy if tensor has content
                # Create slice indices
                slice_indices = [slice(None)] * output.dim()
                slice_indices[dim] = slice(current_offset, current_offset + tensor_size)
                
                # Copy data
                output[tuple(slice_indices)].copy_(safe_tensor)
                
            current_offset += tensor_size

        # Ensure all copy operations are complete before proceeding
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)

        # If we padded the tensor dimension for divisibility, remove the excess.
        if output.size(dim) > ctx.full_dim_size:
            idx = [slice(None)] * output.dim()
            idx[dim] = slice(0, ctx.full_dim_size)
            output = output[tuple(idx)]

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Expect exactly one gradient tensor from downstream.
        grad_output = grad_outputs[0]

        # Non-distributed: gradient flows straight through.
        # Use new utility functions
        if (
            ctx.process_group is None
            or not dist_ops.is_dist_initialized_and_available()
            or dist_ops.get_world_size(ctx.process_group) == 1
        ):
            return grad_output, None, None, None

        rank = dist_ops.get_rank(ctx.process_group)

        # Compute start/end indices for this rank's slice along the gather dim.
        local_dim_padded = ctx.local_dim  # Already accounts for padding in weight shape.
        start = rank * local_dim_padded
        end = start + ctx.local_dim

        # Extract the gradient slice that corresponds to this rank.
        idx = [slice(None)] * grad_output.dim()
        idx[ctx.dim] = slice(start, end)
        grad_input = grad_output[tuple(idx)].contiguous()

        return grad_input, None, None, None


class _Reduce(torch.autograd.Function):
    """Autograd-aware all-reduce + sum.

    In the forward pass, the input tensor is summed across all ranks in the
    process group, and the result is made available on all ranks.
    In the backward pass, the gradient output (which is identical on all ranks)
    is passed through as the gradient for the input tensor on each rank.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, process_group: Optional[ProcessGroup]):
        # Use new utility functions
        if not dist_ops.is_dist_initialized_and_available() or dist_ops.get_world_size(process_group) == 1:
            ctx.process_group = None  # Mark non-distributed case
            return input_

        ctx.process_group = process_group
        # Create a copy to avoid in-place operations on the input tensor which can cause
        # CUDA 12.8 device-side asserts in autograd graphs
        input_contig = input_.contiguous().clone()

        # Ensure CUDA synchronization before collective operation for CUDA 12.8
        if input_contig.device.type == 'cuda':
            torch.cuda.synchronize(device=input_contig.device)

        # Perform the all-reduce with SUM operation using new utility function wrapper.
        dist_ops.all_reduce(input_contig, op=dist_ops.SUM, group=process_group)
        # The tensor input_contig now holds the sum.
        return input_contig

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Non-distributed: gradient flows straight through.
        # Use new utility functions
        if (
            ctx.process_group is None
            or not dist_ops.is_dist_initialized_and_available()
            or dist_ops.get_world_size(ctx.process_group) == 1
        ):
            # Match the number of forward inputs in return for consistency
            return grad_output.contiguous() if grad_output is not None else None, None

        # The gradient dL/dX_local for each rank's local input X_local is simply dL/dY_sum,
        # where Y_sum is the all-reduced sum, because dY_sum/dX_local = 1.
        # grad_output is dL/dY_sum and is identical on all ranks.
        # Ensure contiguous and handle if grad_output might be None (though unlikely for scalar loss)
        grad_for_input = grad_output.contiguous() if grad_output is not None else None
        return grad_for_input, None  # Grad for input_, None for process_group


def _reduce(input_, process_group):
    """All-reduce the input tensor across the process group (SUM, no additional scaling).
    Now uses the autograd-aware _Reduce.apply for proper gradient tracking.

    For row-parallel layers each rank holds a slice of the input-feature dimension and the
    local matmul produces the *partial* output.  These partial outputs must be **summed**
    across ranks to obtain the final result.  Dividing by `world_size` here incorrectly
    rescales the forward activations (and thus the gradients) leading to loss explosions
    and broken optimisation.  The caller can always divide afterwards if an average is
    truly desired, but for the core TP math we need the raw sum.
    """
    # Use new utility functions
    if not dist_ops.is_dist_initialized_and_available():
        return input_  # No-op if not distributed

    world_size = dist_ops.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Ensure input is contiguous
    input_ = input_.contiguous()

    # Perform the all-reduce with **SUM** operation (the correct aggregation for TP)
    return _Reduce.apply(input_, process_group)


def _split(input_, process_group, dim=-1):
    """Split the tensor along dimension dim and keep the corresponding slice.
    Assumes uniform padding, so each rank gets ceil(full_dim / world_size).
    Handles truncation for ranks that would exceed the original full dimension.
    """
    # Use new utility functions
    if not dist_ops.is_dist_initialized_and_available():
        return input_  # No-op if not distributed

    world_size = dist_ops.get_world_size(process_group)
    if world_size == 1:
        return input_

    rank = dist_ops.get_rank(process_group)
    full_dim_size = input_.size(dim)

    # Calculate the size of each slice (using ceil for uniform distribution)
    local_dim_size_padded = math.ceil(full_dim_size / world_size)

    # Calculate the start index for this rank
    start_index = rank * local_dim_size_padded

    # Calculate the actual size of the slice for this rank, considering the original dimension
    actual_local_dim_size = max(0, min(local_dim_size_padded, full_dim_size - start_index))

    # If the actual size is 0 (rank is beyond the original size), return an empty tensor
    if actual_local_dim_size <= 0:
        new_shape = list(input_.shape)
        new_shape[dim] = 0
        return torch.empty(new_shape, dtype=input_.dtype, device=input_.device)

    # Use torch.narrow to get the slice based on calculated start and actual size
    return input_.narrow(dim, start_index, actual_local_dim_size)


class ColumnParallelLinear(_ParallelLinear):
    """Linear layer with column parallelism.

    Output features are sharded across ranks (uniformly padded).
    Input requires the full tensor.
    Forward pass includes an all-gather operation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,  # Enforce keyword args for options
        bias: bool = True,
        process_group: Optional[ProcessGroup],  # Allow None
        init_method: Callable = nn.init.xavier_uniform_,
        keep_master_weight: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            process_group=process_group,
            partition_dim=0,  # Shard output features (columns)
            init_method=init_method,
            input_is_parallel=False,  # Input is full
            keep_master_weight=keep_master_weight,
            d_model_for_init=None,
            num_layers_for_init=None,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: [..., in_features] (Full tensor)

        # Compute local part of the output: [..., local_out_features] (padded)
        local_output = F.linear(input_, self.weight, self.bias_param if self.bias else None)

        # Gather output across ranks: [..., full_out_features] (truncated)
        # Pass the original full dimension size for potential truncation
        gathered_output = _gather(
            local_output.contiguous(), self.process_group, dim=-1, full_dim_size=self.full_out_features
        )

        return gathered_output


class RowParallelLinear(_ParallelLinear):
    """Linear layer with row parallelism.

    Input features are sharded across ranks (uniformly padded).
    Output is the full tensor (requires all-reduce).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,  # Enforce keyword args for options
        bias: bool = True,
        process_group: Optional[ProcessGroup],  # Allow None
        init_method: Callable = nn.init.xavier_uniform_,
        input_is_parallel: bool = True,  # Expect input to be sharded unless specified
        keep_master_weight: bool = False,
        # Require d_model and num_layers for initialization
        d_model_for_init: int,
        num_layers_for_init: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,  # Bias is applied *after* reduce
            process_group=process_group,
            partition_dim=1,  # Shard input features (rows of weight matrix)
            init_method=init_method,
            input_is_parallel=input_is_parallel,
            keep_master_weight=keep_master_weight,
            d_model_for_init=d_model_for_init,
            num_layers_for_init=num_layers_for_init,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: [..., full_in_features] or [..., local_in_features] (potentially non-padded slice)

        if not self.input_is_parallel:
            # Input is full, split it for this rank (gets potentially non-padded slice)
            local_input = _split(input_, self.process_group, dim=-1)
        else:
            # Input is already the correct local slice (potentially non-padded)
            local_input = input_

        # --- Pad input slice if necessary to match expected local feature dimension ---
        actual_local_features = local_input.size(-1)
        expected_local_features = self.local_in_features  # This is the required padded size for self.weight

        if actual_local_features < expected_local_features:
            # This rank received fewer features than its weight slice expects.
            # This happens on ranks >= (full_in_features % world_size) when full_in_features is not divisible by world_size,
            # or on ranks >= full_in_features when full_in_features < world_size.
            if actual_local_features == 0:
                # Rank received no input features for its slice. Output should be zero before reduction.
                # Determine the batch dimensions from input_
                batch_shape = input_.shape[:-1] if not self.input_is_parallel else local_input.shape[:-1]
                output_shape = batch_shape + (self.local_out_features,)
                # Ensure device and dtype match weight for CUDA 12.8 compatibility
                local_output = torch.zeros(
                    output_shape, 
                    dtype=self.weight.dtype, 
                    device=self.weight.device,
                    memory_format=torch.contiguous_format
                )
            else:
                # Pad the input slice with zeros on the right to match the weight's expected dimension
                pad_size = expected_local_features - actual_local_features
                # Ensure padding preserves device and uses contiguous memory format
                padded_input = F.pad(local_input, (0, pad_size), mode='constant', value=0.0)
                # Ensure input is contiguous before linear operation for CUDA 12.8
                padded_input = padded_input.contiguous()
                local_output = F.linear(padded_input, self.weight)  # Bias is added after reduce
        elif actual_local_features == expected_local_features:
            # Input size matches expected padded size, no padding needed.
            # Ensure input is contiguous before linear operation for CUDA 12.8
            local_input = local_input.contiguous()
            local_output = F.linear(local_input, self.weight)  # Bias is added after reduce
        else:
            # This should not happen if _split and input processing are correct.
            # The actual slice size should never exceed the calculated padded size.
            raise ValueError(
                f"RowParallelLinear (rank {self.rank}): Input slice size ({actual_local_features}) "
                f"unexpectedly greater than expected padded size ({expected_local_features})."
            )
        # --- End Padding ---

        # Reduce outputs across ranks: [..., out_features]
        reduced_output = _reduce(local_output, self.process_group)

        # Add bias *after* reduction
        if self.bias and self.bias_param is not None:
            # The runtime check `self.bias_param is not None` is the primary guard.
            # Casting `self.bias_param` to `torch.Tensor` helps the type checker.
            reduced_output = reduced_output + cast(torch.Tensor, self.bias_param)

        return cast(torch.Tensor, reduced_output)


# --------------------------- Public helper --------------------------- #


def _gather(
    input_: torch.Tensor,
    process_group: Optional[ProcessGroup],
    dim: int = -1,
    full_dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Wrapper around :class:`_Gather` to match original functional interface."""

    return cast(torch.Tensor, _Gather.apply(input_, process_group, dim, full_dim_size))
