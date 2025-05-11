import torch.distributed as dist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clt.models.clt import CrossLayerTranscoder  # For type hinting model parameters


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
        # Assuming replicated parameters are 1D (like JumpReLU log_threshold or biases)
        # More sophisticated checks might be needed if other replicated parameter structures exist.
        if p.dim() == 1:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size
