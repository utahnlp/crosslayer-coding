import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup
import math
from typing import Callable, Optional


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
        elif partition_dim == 1:  # Row Parallelism (Input features sharded)
            # Calculate padded size for uniform distribution
            self.local_in_features = math.ceil(in_features / self.world_size)
            self.local_out_features = out_features
            self.weight = nn.Parameter(torch.empty(self.local_out_features, self.local_in_features, device=device))
            if bias:
                # Bias is added *after* the all-reduce, so it's not sharded
                self.bias_param = nn.Parameter(torch.empty(out_features, device=device))
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


def _gather(input_, process_group, dim=-1, full_dim_size: Optional[int] = None):
    """Gather tensors and concatenate along specified dimension.
    Assumes all ranks have the same local dimension size (padded if needed).
    """
    if process_group is None or not dist.is_initialized():
        return input_  # No-op if not distributed

    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Ensure input is contiguous
    input_ = input_.contiguous()

    # Prepare the output list. Each tensor must have the same shape.
    gathered_list = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(gathered_list, input_, group=process_group)

    # Concatenate along the specified dimension
    output = torch.cat(gathered_list, dim=dim)

    # Truncate the gathered output if the original dimension was not divisible
    if full_dim_size is not None and output.shape[dim] > full_dim_size:
        indices = [slice(None)] * output.dim()
        indices[dim] = slice(0, full_dim_size)
        output = output[tuple(indices)]

    return output


def _reduce(input_, process_group):
    """All-reduce the input tensor across the process group (SUM, no additional scaling).

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
    dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=process_group)

    # No extra scaling — return the summed tensor
    return input_


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

        # Gather output across ranks: [..., full_out_features] (truncated)
        # Pass the original full dimension size for potential truncation
        gathered_output = _gather(local_output, self.process_group, dim=-1, full_dim_size=self.full_out_features)

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
