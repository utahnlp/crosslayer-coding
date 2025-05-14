import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup
import math
from typing import Callable, Optional, cast, Tuple
import logging

from . import mark_replicated


# ---------------------------------------------------------------------------
# Module-level logger (kept lightweight so it is cheap when DEBUG is disabled)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class _ParallelLinear(nn.Module):
    """Base class for parallel linear layers."""

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
    ):
        super().__init__()
        self.process_group = process_group

        # Handle non-distributed case
        if process_group is None or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
            self.process_group = None  # Ensure it's None if not initialized
        else:
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)

        self.partition_dim = partition_dim
        self.input_is_parallel = input_is_parallel
        self.bias = bias
        self.full_in_features = in_features
        self.full_out_features = out_features

        # Calculate local dimensions with uniform padding for divisibility
        if partition_dim == 0:  # Column Parallelism (Output features sharded)
            # Calculate padded size for uniform distribution
            self.local_out_features = math.ceil(out_features / self.world_size)
            self.local_in_features = in_features
            self.weight = nn.Parameter(torch.empty(self.local_out_features, self.local_in_features, device=device))
            if bias:
                self.bias_param = nn.Parameter(torch.empty(self.local_out_features, device=device))
                # DO NOT mark_replicated here, ColumnParallel bias is sharded
        elif partition_dim == 1:  # Row Parallelism (Input features sharded)
            # Calculate padded size for uniform distribution
            self.local_in_features = math.ceil(in_features / self.world_size)
            self.local_out_features = out_features
            self.weight = nn.Parameter(torch.empty(self.local_out_features, self.local_in_features, device=device))
            if bias:
                # Bias is added *after* the all-reduce, so it's not sharded (globally, but replicated on each rank)
                self.bias_param = nn.Parameter(torch.empty(out_features, device=device))
                mark_replicated(self.bias_param)  # Mark as replicated
        else:
            raise ValueError("partition_dim must be 0 or 1")

        # Initialize weights (ensure consistency across ranks if needed)
        # Default init methods often depend on full shapes. We might need custom init.
        # Simplified init for CLT (matching original CLT init logic)
        if partition_dim == 0:  # Encoder-like
            # Bound depends on *full* output dimension
            bound = 1.0 / math.sqrt(self.full_out_features)  # Use full dim
            nn.init.uniform_(self.weight, -bound, bound)
            if bias:
                nn.init.zeros_(self.bias_param)
        elif partition_dim == 1:  # Decoder-like
            # Use passed d_model and num_layers for initialization bound
            if d_model_for_init is None or num_layers_for_init is None:
                raise ValueError("d_model_for_init and num_layers_for_init must be provided for RowParallelLinear init")
            # Bound depends on full input dimension (num_features) and model config
            bound = 1.0 / math.sqrt(num_layers_for_init * d_model_for_init)
            nn.init.uniform_(self.weight, -bound, bound)
            if bias:
                nn.init.zeros_(self.bias_param)  # Initialize full bias on all ranks, will be reduced

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
        if process_group is None or not dist.is_initialized() or dist.get_world_size(process_group) == 1:
            ctx.dim = dim
            ctx.local_dim = input_.size(dim)
            ctx.full_dim_size = full_dim_size or input_.size(dim)
            ctx.process_group = None  # Mark non-distributed case
            return input_

        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)

        ctx.dim = dim
        ctx.local_dim = input_.size(dim)
        ctx.full_dim_size = full_dim_size if full_dim_size is not None else ctx.local_dim * world_size
        ctx.process_group = process_group

        # Ensure a contiguous tensor before communication for NCCL efficiency.
        input_contig = input_.contiguous()

        print(
            f"[Gather-fwd Rank {rank}] BEFORE all_gather: input_contig shape={input_contig.shape}, dtype={input_contig.dtype}, device={input_contig.device}, is_contiguous={input_contig.is_contiguous()}, data_ptr={input_contig.data_ptr()}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Allocate a single output tensor for all_gather_into_tensor
        # The shape must be (world_size * local_input_shape[dim], other_dims...)
        # if dim is not the first dimension, or (world_size, local_input_shape...)
        # if dim is 0 and we want to gather along a new first dimension.
        # For typical TP gather (dim=-1), it's (..., world_size * local_input_shape[-1])
        # ------------------------------------------------------------------
        output_tensor_shape = list(input_contig.shape)
        # if dim < 0: # Handle negative dim # Removed unused dim_to_gather
        #     dim_to_gather = output_tensor_shape[dim]
        # else:
        #     dim_to_gather = output_tensor_shape[dim]

        # The output tensor for all_gather_into_tensor should be large enough to hold all gathered data.
        # If local_dim is the size of the input tensor's gather dimension for one rank,
        # then the output tensor's gather dimension should be world_size * local_dim.
        output_tensor_shape[dim] = world_size * ctx.local_dim  # ctx.local_dim is input_.size(dim)

        # Ensure the output tensor is created on the correct device with the correct dtype.
        output_tensor = torch.empty(output_tensor_shape, dtype=input_contig.dtype, device=input_contig.device)

        print(
            f"[Gather-fwd Rank {rank}] BEFORE all_gather_into_tensor: output_tensor shape={output_tensor.shape}, input_contig shape={input_contig.shape}",
            flush=True,
        )

        # Perform the all-gather into the single output tensor.
        try:
            print(f"[Gather-fwd Rank {rank}] CALLING dist.all_gather_into_tensor", flush=True)
            dist.all_gather_into_tensor(output_tensor, input_contig, group=process_group)
            print(f"[Gather-fwd Rank {rank}] AFTER dist.all_gather_into_tensor SUCCEEDED", flush=True)
            print(
                f"[Gather-fwd Rank {rank}] AFTER all_gather_into_tensor: output_tensor shape={output_tensor.shape}, dtype={output_tensor.dtype}, device={output_tensor.device}, is_contiguous={output_tensor.is_contiguous()}, data_ptr={output_tensor.data_ptr()}",
                flush=True,
            )
        except Exception as e_ag:
            print(f"[Gather-fwd Rank {rank}] EXCEPTION during dist.all_gather_into_tensor: {e_ag}", flush=True)
            raise

        # Note: torch.cat is no longer needed as all_gather_into_tensor directly produces the concatenated result.
        # The `output_tensor` is now the `output` we need, but it might need truncation if padding was involved.
        output = output_tensor  # Assign output_tensor to output

        # If we padded the tensor dimension for divisibility, remove the excess
        # so that downstream code always sees exactly ``full_dim_size`` cols.
        # ctx.full_dim_size is the target size of the gathered dimension AFTER concatenation.
        if output.size(dim) > ctx.full_dim_size:
            idx = [slice(None)] * output.dim()
            idx[dim] = slice(0, ctx.full_dim_size)
            output = output[tuple(idx)]
            print(
                f"[Gather-fwd Rank {rank}] Truncated output to shape {output.shape} along dim {dim} to match full_dim_size {ctx.full_dim_size}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Save the *actual* (unpadded) number of columns owned by this rank.
        # Rank r owns the slice [start:end) where
        #   start = r * local_dim_padded
        #   end   = min(start + local_dim_padded, full_dim_size)
        # ------------------------------------------------------------------
        local_dim_padded: int = ctx.local_dim
        start_idx = rank * local_dim_padded
        ctx.actual_local_dim = max(0, min(local_dim_padded, ctx.full_dim_size - start_idx))

        # Ensure logger is defined and accessible for the following debug log
        # If logger is module-level, this check might be redundant but safe.
        if "logger" in globals() and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[Gather-fwd] rank={rank} local_dim_padded={local_dim_padded} "
                f"actual_local_dim={ctx.actual_local_dim} full_dim={ctx.full_dim_size}"
            )
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        # Expect exactly one gradient tensor from downstream.
        grad_output = grad_outputs[0]

        # Non-distributed: gradient flows straight through.
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) == 1:
            return grad_output, None, None, None, None

        rank = dist.get_rank(ctx.process_group)

        # Compute start/end indices for this rank's slice along the gather dim.
        local_dim_padded: int = ctx.local_dim
        actual_local_dim: int = ctx.actual_local_dim
        start = rank * local_dim_padded
        end = start + actual_local_dim

        # Extract the gradient slice that corresponds to this rank.
        idx = [slice(None)] * grad_output.dim()
        idx[ctx.dim] = slice(start, end)
        grad_input = grad_output[tuple(idx)].contiguous()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[Gather-bwd] rank={rank} slice=({start}:{end}) "
                f"grad_cols={grad_output.shape[ctx.dim]}  "
                f"local_dim={ctx.local_dim}  actual_local_dim={ctx.actual_local_dim}"
            )

        assert (
            end <= grad_output.shape[ctx.dim]
        ), f"Rank {rank}: gradient slice overruns tensor (end {end} > {grad_output.shape[ctx.dim]})"

        return grad_input, None, None, None, None


class _Reduce(torch.autograd.Function):
    """Autograd-aware all-reduce + sum.

    In the forward pass, the input tensor is summed across all ranks in the
    process group, and the result is made available on all ranks.
    In the backward pass, the gradient output (which is identical on all ranks)
    is passed through as the gradient for the input tensor on each rank.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, process_group: Optional[ProcessGroup]):
        if process_group is None or not dist.is_initialized() or dist.get_world_size(process_group) == 1:
            ctx.process_group = None  # Mark non-distributed case
            return input_

        ctx.process_group = process_group
        input_contig = input_.contiguous()  # Ensure contiguous before collective

        # Perform the all-reduce with SUM operation.
        # The operation is in-place on input_contig if it's the same object for all_reduce's output internally,
        # or if all_reduce returns a new tensor, that's what we return.
        # For clarity, let's assume all_reduce modifies input_contig or we assign its result.
        dist.all_reduce(input_contig, op=dist.ReduceOp.SUM, group=process_group)
        # The tensor input_contig now holds the sum.
        return input_contig

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        # Non-distributed: gradient flows straight through.
        if ctx.process_group is None or not dist.is_initialized() or dist.get_world_size(ctx.process_group) == 1:
            # Match the number of forward inputs in return for consistency
            return grad_outputs[0].contiguous() if grad_outputs[0] is not None else None, None

        # The gradient dL/dX_local for each rank's local input X_local is simply dL/dY_sum,
        # where Y_sum is the all-reduced sum, because dY_sum/dX_local = 1.
        # grad_output is dL/dY_sum and is identical on all ranks.
        # Ensure contiguous and handle if grad_output might be None (though unlikely for scalar loss)
        grad_for_input = grad_outputs[0].contiguous() if grad_outputs[0] is not None else None
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
    if process_group is None or not dist.is_initialized():
        return input_  # No-op if not distributed

    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Ensure input is contiguous
    input_ = input_.contiguous()

    # Perform the all-reduce with **SUM** operation (the correct aggregation for TP)
    # return input_ # Old implementation
    return _Reduce.apply(input_, process_group)  # Use autograd function


def _split(input_, process_group, dim=-1):
    """Split the tensor along dimension dim and keep the corresponding slice.
    Assumes uniform padding, so each rank gets ceil(full_dim / world_size).
    Handles truncation for ranks that would exceed the original full dimension.
    """
    if process_group is None or not dist.is_initialized():
        return input_  # No-op if not distributed

    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    rank = dist.get_rank(process_group)
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
            # Pass None for row parallel init args
            d_model_for_init=None,
            num_layers_for_init=None,
            device=device,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: [..., in_features] (Full tensor)

        # Compute local part of the output: [..., local_out_features] (padded)
        local_output = F.linear(input_, self.weight, self.bias_param if self.bias else None)

        # ↓ insert directly after the F.linear call
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[ColPar fwd rank {self.rank}] local_output shape={local_output.shape} "
            f"weight.shape={self.weight.shape} input_.shape={input_.shape} "
            f"expected local_in={self.local_in_features} local_out={self.local_out_features}",
            flush=True,
        )

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
            # Pass specific init args
            d_model_for_init=d_model_for_init,
            num_layers_for_init=num_layers_for_init,
            device=device,
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
                local_output = torch.zeros(output_shape, dtype=input_.dtype, device=input_.device)
            else:
                # Pad the input slice with zeros on the right to match the weight's expected dimension
                pad_size = expected_local_features - actual_local_features
                padded_input = F.pad(local_input, (0, pad_size))
                local_output = F.linear(padded_input, self.weight)  # Bias is added after reduce
        elif actual_local_features == expected_local_features:
            # Input size matches expected padded size, no padding needed.
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
        if self.bias:
            # Ensure bias_param is correctly shaped [out_features]
            reduced_output = reduced_output + self.bias_param

        return reduced_output


# --------------------------- Public helper --------------------------- #


def _gather(
    input_: torch.Tensor,
    process_group: Optional[ProcessGroup],
    dim: int = -1,
    full_dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Wrapper around :class:`_Gather` to match original functional interface."""

    return cast(torch.Tensor, _Gather.apply(input_, process_group, dim, full_dim_size))
