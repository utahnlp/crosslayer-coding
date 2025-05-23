from typing import TYPE_CHECKING
from clt.parallel import ops as dist_ops

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
            dist_ops.all_reduce(p.grad, op=dist_ops.SUM)
            p.grad /= world_size
