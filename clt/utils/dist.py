import os
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


def init_distributed() -> tuple[int, int]:
    """Initialize the distributed process group and return rank and world size.

    Handles both single-GPU and multi-GPU scenarios using environment variables
    set by torchrun or similar launchers.

    Returns:
        Tuple[int, int]: (rank, world_size)
    """
    if not dist.is_available():
        logger.warning("Distributed training not available.")
        return 0, 1

    world_size_str = os.environ.get("WORLD_SIZE")
    if world_size_str is None or int(world_size_str) <= 1:
        # Single-GPU or non-distributed setup
        logger.info("Running in single-process mode (WORLD_SIZE <= 1 or not set).")
        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        # No process group initialization needed for single process
        return 0, 1

    # Multi-GPU setup
    if not torch.cuda.is_available():
        logger.error("CUDA not available, cannot initialize distributed training.")
        return 0, 1  # Fallback

    try:
        # Initialize process group using environment variables (usually set by torchrun)
        # The default backend is NCCL if CUDA is available.
        dist.init_process_group(backend="nccl", init_method="env://")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank_str = os.environ.get("LOCAL_RANK")

        if local_rank_str is None:
            logger.error("LOCAL_RANK environment variable not set. Cannot set device.")
            # Potentially clean up dist group?
            # dist.destroy_process_group() # Or let it error out later
            return rank, world_size  # Return what we have, but device won't be set

        local_rank = int(local_rank_str)
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Distributed training initialized (backend='nccl'). Rank {rank}/{world_size}, "
            f"Device set to cuda:{local_rank}"
        )
        return rank, world_size

    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}", exc_info=True)
        # Attempt cleanup if initialization failed partially
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0, 1  # Fallback to non-distributed mode
