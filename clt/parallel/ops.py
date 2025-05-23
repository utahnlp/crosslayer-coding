import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp, Work
from typing import Optional, List

# Re-export common ReduceOp for convenience.
# Users can import these from this module (e.g., from clt.parallel.ops import SUM)
SUM = ReduceOp.SUM
AVG = ReduceOp.AVG
PRODUCT = ReduceOp.PRODUCT
MIN = ReduceOp.MIN
MAX = ReduceOp.MAX
BAND = ReduceOp.BAND
BOR = ReduceOp.BOR
BXOR = ReduceOp.BXOR


def is_dist_initialized_and_available() -> bool:
    """Checks if torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Returns the rank of the current process in the group.
    Returns 0 if distributed is not initialized or not available.
    """
    if not is_dist_initialized_and_available():
        return 0
    return dist.get_rank(group=group)


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Returns the world size of the given process group.
    Returns 1 if distributed is not initialized or not available.
    """
    if not is_dist_initialized_and_available():
        return 1
    return dist.get_world_size(group=group)


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Checks if the current process is the main process (rank 0)."""
    return get_rank(group=group) == 0


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = SUM,  # Default to SUM
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    """Reduces the tensor data across all machines.

    Args:
        tensor: Input and output of the collective. The function operates in-place.
        op: The reduction operation (e.g., ReduceOp.SUM, ReduceOp.PRODUCT).
        group: The process group to work on. If None, the default process group will be used.
        async_op: Whether this op should be an async op.

    Returns:
        A Work object if async_op is True, otherwise None.
        Returns None if distributed is not initialized or world_size is 1, as no actual communication occurs.
    """
    if not is_dist_initialized_and_available() or get_world_size(group=group) == 1:
        return None
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    """Broadcasts the tensor to the whole group.

    Args:
        tensor: Data to be sent if src is the rank of current process,
                or tensor to be used to save received data otherwise.
        src: Source rank.
        group: The process group to work on. If None, the default process group will be used.
        async_op: Whether this op should be an async op.

    Returns:
        A Work object if async_op is True, otherwise None.
        Returns None if distributed is not initialized or world_size is 1, as no actual communication occurs.
    """
    if not is_dist_initialized_and_available() or get_world_size(group=group) == 1:
        return None
    return dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def all_gather(
    tensor_list: List[torch.Tensor],
    tensor: torch.Tensor,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    """Gathers tensors from the whole group in a list.

    Args:
        tensor_list: Output list. It should contain correctly-sized tensors to be used for output of the collective.
        tensor: Tensor to be broadcast from current process.
        group: The process group to work on. If None, the default process group will be used.
        async_op: Whether this op should be an async op.

    Returns:
        A Work object if async_op is True, otherwise None.
        If distributed is not initialized, it places the input tensor into tensor_list[0] (assuming single process context).
    """
    if not is_dist_initialized_and_available():
        rank = get_rank(group)
        if rank < len(tensor_list):
            tensor_list[rank] = tensor  # pyright: ignore[reportGeneralTypeIssues]
        return None

    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
