import torch
import torch.distributed as dist
from typing import TYPE_CHECKING, Dict, Optional, Union
import logging
from dataclasses import dataclass
from torch.distributed import ProcessGroup
import os

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from clt.models.clt import CrossLayerTranscoder  # For type hinting model parameters

    # from clt.models import is_replicated # No longer needed


def average_shared_parameter_grads(model: "CrossLayerTranscoder", world_size: int):
    """Average gradients of parameters that are **replicated** across ranks.

    Tensor-parallel layers shard their weights so those gradients must **not** be
    synchronised.  However parameters that are kept identical on every rank –
    e.g. the JumpReLU `log_threshold` vector (shape `[num_features]`) and any
    unsharded bias vectors – must have their gradients reduced or they will
    diverge between ranks.
    """
    # This function is called when distributed and world_size > 1
    # The original method had a guard: `if not self.distributed or self.world_size == 1: return`
    # That check should be done by the caller now.
    for p in model.parameters():
        if p.grad is None:
            continue

        # Check if parameter is marked as replicated OR if it's a 1D tensor (for backward compatibility)
        # The import for is_replicated will be guarded by TYPE_CHECKING, so use getattr for runtime.
        is_rep = getattr(p, "_is_replicated", False)

        # Only average if explicitly marked as replicated.
        # The p.dim() == 1 heuristic was too broad and could incorrectly average sharded 1D parameters (e.g., encoder biases).
        if is_rep:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size


@dataclass
class DistributedEnvInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    process_group: Optional[ProcessGroup]
    is_distributed: bool  # Reflects if distributed mode is effectively active


def initialize_distributed_env(
    use_distributed_flag: bool, manual_device_override: Optional[Union[str, torch.device]] = None
) -> DistributedEnvInfo:
    """
    Initializes the distributed environment (if requested and available)
    and determines the appropriate device.

    Args:
        use_distributed_flag: Whether distributed training is intended.
        manual_device_override: A specific device requested by the user.
                                In distributed CUDA mode, this is typically ignored
                                in favor of local_rank-based device assignment.
    Returns:
        DistributedEnvInfo containing rank, world_size, local_rank, device,
        process_group, and a flag indicating if distributed mode is active.
    """
    rank = 0
    world_size = 1
    local_rank = 0  # Default for non-distributed or if LOCAL_RANK env var is not set
    process_group = None
    is_effectively_distributed = False  # True only if dist is successfully initialized and used

    if use_distributed_flag:
        if not dist.is_available():
            logger.warning("torch.distributed is not available. Falling back to non-distributed mode.")
            is_effectively_distributed = False
        elif not dist.is_initialized():
            logger.info("torch.distributed is available but not initialized. Attempting to initialize process group...")
            try:
                # NCCL is preferred for NVIDIA GPUs. Gloo for CPU or other GPUs.
                backend_to_use = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(
                    backend=backend_to_use,
                    # init_method='env://' is implicitly used if not specified and
                    # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE are in env.
                )
                is_effectively_distributed = True
                logger.info(
                    f"Successfully initialized process group with backend: {dist.get_backend()}. Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize process group: {e}. Falling back to non-distributed mode.", exc_info=True
                )
                is_effectively_distributed = False
        else:
            # Already initialized
            is_effectively_distributed = True
            logger.info(
                f"Using pre-initialized torch.distributed process group. Backend: {dist.get_backend()}. Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}"
            )

        if is_effectively_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                local_rank = int(local_rank_env)
            else:
                local_rank = rank  # Fallback if LOCAL_RANK is not set
                if world_size > 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    logger.warning(
                        f"LOCAL_RANK environment variable not set in a distributed environment with multiple GPUs. "
                        f"Defaulting local_rank to global rank ({rank}). This may cause issues if multiple "
                        f"processes are on the same node without proper local rank assignment."
                    )

            if torch.cuda.is_available():
                if manual_device_override:
                    logger.warning(
                        f"Manual device override '{manual_device_override}' provided in distributed mode. "
                        f"Device will be set based on local_rank ({local_rank}) for CUDA."
                    )

                if local_rank >= torch.cuda.device_count():
                    logger.error(
                        f"local_rank {local_rank} is out of bounds for available CUDA devices ({torch.cuda.device_count()}). "
                        f"Falling back to CPU. Check distributed launch configuration."
                    )
                    current_device = torch.device("cpu")
                    # If device setup fails critically for distributed CUDA, treat as non-distributed.
                    # This prevents hanging on collective operations with mismatched devices.
                    dist.destroy_process_group()  # Clean up if partially initialized
                    is_effectively_distributed = False
                    rank = 0
                    world_size = 1
                    local_rank = 0  # Reset to non-distributed defaults
                else:
                    current_device = torch.device(f"cuda:{local_rank}")
                    torch.cuda.set_device(current_device)
            else:  # No CUDA available
                current_device = torch.device("cpu")
                logger.warning("Distributed training requested but CUDA not available. Using CPU for all ranks.")

            if is_effectively_distributed:  # Check again in case CUDA setup failed
                process_group = dist.group.WORLD
            else:  # CUDA setup failed, revert to non-distributed device
                current_device = torch.device(manual_device_override or "cpu")  # Fallback to manual or CPU
                if isinstance(current_device, str):
                    current_device = torch.device(current_device)
                process_group = None

        else:  # Fell back from use_distributed_flag=True (e.g. dist.init failed)
            current_device_input = manual_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
            current_device = (
                torch.device(current_device_input) if isinstance(current_device_input, str) else current_device_input
            )
            if current_device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA specified but not available. Falling back to CPU.")
                current_device = torch.device("cpu")
            elif current_device.type == "cuda":  # CUDA is available
                torch.cuda.set_device(current_device)  # Set device if it's a specific CUDA device like cuda:0
            # rank, world_size, local_rank, process_group remain at defaults (0,1,0,None)

    else:  # use_distributed_flag was False from the start
        is_effectively_distributed = False
        current_device_input = manual_device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        current_device = (
            torch.device(current_device_input) if isinstance(current_device_input, str) else current_device_input
        )
        if current_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            current_device = torch.device("cpu")
        elif current_device.type == "cuda":  # CUDA is available
            torch.cuda.set_device(current_device)  # Set device if it's a specific CUDA device like cuda:0
        # rank, world_size, local_rank, process_group remain at defaults

    return DistributedEnvInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=current_device,
        process_group=process_group,
        is_distributed=is_effectively_distributed,
    )
