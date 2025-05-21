import torch
import torch.nn.functional as F
import logging
from typing import Callable, Dict, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from clt.models.clt import CrossLayerTranscoder  # To avoid circular import

logger = logging.getLogger(__name__)

# Type alias for activation functions used in the registry
# They take the CrossLayerTranscoder instance, pre-activations, and layer_idx
ActivationCallable = Callable[["CrossLayerTranscoder", torch.Tensor, int], torch.Tensor]

ACTIVATION_REGISTRY: Dict[str, ActivationCallable] = {}


def register_activation_fn(name: str) -> Callable[[ActivationCallable], ActivationCallable]:
    """Decorator to register a new activation function."""

    def decorator(fn: ActivationCallable) -> ActivationCallable:
        if name in ACTIVATION_REGISTRY:
            logger.warning(f"Activation function '{name}' is already registered. Overwriting.")
        ACTIVATION_REGISTRY[name] = fn
        return fn

    return decorator


@register_activation_fn("relu")
def relu_activation(model: "CrossLayerTranscoder", preact: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Standard ReLU activation."""
    return F.relu(preact)


@register_activation_fn("jumprelu")
def jumprelu_activation(model: "CrossLayerTranscoder", preact: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """JumpReLU activation."""
    # The model's jumprelu method handles device/dtype and threshold selection
    return model.jumprelu(preact, layer_idx)


@register_activation_fn("batchtopk")
def batchtopk_per_layer_activation(model: "CrossLayerTranscoder", preact: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """BatchTopK activation applied per-layer (not global)."""
    from clt.models.activations import BatchTopK  # Local import to avoid issues if activations.py imports this registry

    logger.warning(
        f"Rank {model.rank}: 'encode' called for BatchTopK on layer {layer_idx}. "
        f"This applies TopK per-layer, not globally. Use 'get_feature_activations' for global BatchTopK."
    )
    k_val_local_int: int
    if model.config.batchtopk_k is not None:
        k_val_local_int = int(model.config.batchtopk_k)
    else:
        # If k is None, default to keeping all features for this layer.
        # This might happen if batchtopk_k is not set in the config,
        # though it typically should be for BatchTopK.
        k_val_local_int = preact.size(1)  # Number of features in this layer's preactivation
        logger.warning(
            f"Rank {model.rank}: batchtopk_k not set in config for per-layer BatchTopK on layer {layer_idx}. "
            f"Defaulting to k={k_val_local_int} (all features for this layer)."
        )

    # BatchTopK.apply takes the original preactivation, k, straight_through, and optional ranking tensor.
    # For per-layer application, we don't have a separate normalized ranking tensor readily available here from encode_all_layers,
    # so we pass preact itself for ranking if x_for_ranking is None.
    # Normalization for ranking, if desired for per-layer, would need to happen here or BatchTopK would need to handle it.
    # The global BatchTopK in _apply_batch_topk_helper does normalization.
    return cast(
        torch.Tensor, BatchTopK.apply(preact, float(k_val_local_int), model.config.batchtopk_straight_through, preact)
    )


@register_activation_fn("topk")
def topk_per_layer_activation(model: "CrossLayerTranscoder", preact: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """TokenTopK activation applied per-layer (not global)."""
    from clt.models.activations import TokenTopK  # Local import

    logger.warning(
        f"Rank {model.rank}: 'encode' called for TopK (TokenTopK) on layer {layer_idx}. "
        f"This applies TopK per-layer, not globally. Use 'get_feature_activations' for global TopK."
    )
    k_val_local_float: float
    if hasattr(model.config, "topk_k") and model.config.topk_k is not None:
        k_val_local_float = float(model.config.topk_k)
    else:
        # Default to keeping all features for this layer if topk_k not set
        k_val_local_float = float(preact.size(1))
        logger.warning(
            f"Rank {model.rank}: topk_k not set in config for per-layer TopK on layer {layer_idx}. "
            f"Defaulting to k={k_val_local_float} (all features for this layer)."
        )

    straight_through_local = getattr(model.config, "topk_straight_through", True)
    # TokenTopK.apply takes preact, k, straight_through, and x_for_ranking.
    # Similar to BatchTopK, for per-layer, we use preact for ranking if x_for_ranking is None.
    return cast(torch.Tensor, TokenTopK.apply(preact, k_val_local_float, straight_through_local, preact))


def get_activation_fn(name: str) -> ActivationCallable:
    """Retrieve an activation function from the registry."""
    fn = ACTIVATION_REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Activation function '{name}' not found in registry. Available: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return fn
