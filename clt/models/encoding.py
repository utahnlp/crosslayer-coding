import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, cast
import logging

from clt.config import CLTConfig
from clt.models.activations import BatchTopK, TokenTopK

# Configure logging (or use existing logger if available)
# It's generally better for the calling module (clt.py) to pass its logger
# or for these functions to use getLogger(__name__) if they are truly standalone.
# For now, let's assume a logger instance is passed or they use their own.
logger = logging.getLogger(__name__)


def get_preactivations(
    x: torch.Tensor,
    layer_idx: int,
    config: CLTConfig,
    encoders: nn.ModuleList,
    model_device: torch.device,
    model_dtype: torch.dtype,
    rank: int = 0,  # Default rank for non-distributed scenarios if logger uses it
) -> torch.Tensor:
    """Get pre-activation values (full tensor) for features at the specified layer."""
    result: Optional[torch.Tensor] = None
    fallback_shape: Optional[Tuple[int, int]] = None

    try:
        # 1. Check input shape and reshape if necessary
        if x.dim() == 2:
            input_for_linear = x
        elif x.dim() == 3:
            batch, seq_len, d_model = x.shape
            if d_model != config.d_model:
                logger.warning(f"Rank {rank}: Input d_model {d_model} != config {config.d_model} layer {layer_idx}")
                fallback_shape = (batch * seq_len, config.num_features)
            else:
                input_for_linear = x.reshape(-1, d_model)
        else:
            logger.warning(f"Rank {rank}: Cannot handle input shape {x.shape} for preactivations layer {layer_idx}")
            fallback_shape = (0, config.num_features)

        # 2. Check d_model match if not already done
        if fallback_shape is None and input_for_linear.shape[1] != config.d_model:
            logger.warning(
                f"Rank {rank}: Input d_model {input_for_linear.shape[1]} != config {config.d_model} layer {layer_idx}"
            )
            fallback_shape = (input_for_linear.shape[0], config.num_features)

        # 3. Proceed if no errors so far
        if fallback_shape is None:
            # Explicitly cast the output of the parallel linear layer
            result = cast(torch.Tensor, encoders[layer_idx](input_for_linear))

    except IndexError:
        logger.error(f"Rank {rank}: Invalid layer index {layer_idx} requested for encoder.")
        fallback_batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
        fallback_shape = (fallback_batch_dim, config.num_features)
    except Exception as e:
        logger.error(f"Rank {rank}: Error during get_preactivations layer {layer_idx}: {e}", exc_info=True)
        fallback_batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
        fallback_shape = (fallback_batch_dim, config.num_features)

    # 4. Return result or fallback tensor
    if result is not None:
        return result
    else:
        if fallback_shape is None:
            logger.error(f"Rank {rank}: Fallback shape not determined for layer {layer_idx}, returning empty tensor.")
            fallback_shape = (0, config.num_features)
        return torch.zeros(fallback_shape, device=model_device, dtype=model_dtype)


def _encode_all_layers(
    inputs: Dict[int, torch.Tensor],
    config: CLTConfig,
    encoders: nn.ModuleList,
    model_device: torch.device,
    model_dtype: torch.dtype,
    rank: int = 0,
) -> Tuple[Dict[int, torch.Tensor], List[Tuple[int, int, int]], torch.device, torch.dtype]:
    """Encodes inputs for all layers and returns pre-activations and original shape info.
    Assumes input tensors in `inputs` are already on model_device and model_dtype.
    """
    preactivations_dict = {}
    original_shapes_info: List[Tuple[int, int, int]] = []

    # Device and dtype are now asserted by the type hints and caller responsibility
    # No inference or .to() calls needed here for the inputs dict items themselves.

    for layer_idx, x in inputs.items():  # x is x_orig, assumed to be on correct device/dtype
        # Optional: Add assertions here if strict checking is desired at this stage
        # assert x.device == model_device, f"Input for layer {layer_idx} not on expected device"
        # assert x.dtype == model_dtype, f"Input for layer {layer_idx} not on expected dtype"

        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            original_shapes_info.append((layer_idx, batch_size, seq_len))
        elif x.dim() == 2:
            batch_size, _ = x.shape
            original_shapes_info.append((layer_idx, batch_size, 1))

        # Pass through model_device and model_dtype as they are confirmed.
        preact = get_preactivations(x, layer_idx, config, encoders, model_device, model_dtype, rank)
        preactivations_dict[layer_idx] = preact

    return preactivations_dict, original_shapes_info, model_device, model_dtype


def _apply_batch_topk(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    processed_device: torch.device,
    processed_dtype: torch.dtype,
    rank: int = 0,
) -> Dict[int, torch.Tensor]:
    """Applies BatchTopK to concatenated pre-activations from all layers."""
    if not preactivations_dict:
        return {}

    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: All preactivations are empty in _apply_batch_topk. Returning empty dict.")
        return {
            layer_idx: torch.zeros((0, config.num_features), device=processed_device, dtype=processed_dtype)
            for layer_idx in preactivations_dict.keys()
        }

    batch_tokens_dim = first_valid_preact.shape[0]
    # total_features_across_layers = config.num_layers * config.num_features # Max possible if all layers have num_features

    # Pre-allocate tensors for original and normalized preactivations
    # The actual number of features might be less if some layers are missing or have 0 features.
    # We will determine the actual total number of features dynamically.

    # Determine layer_feature_sizes and actual total features first
    layer_feature_sizes: List[Tuple[int, int]] = []
    actual_total_concatenated_features = 0
    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            current_num_features = (
                preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features
            )  # Fallback if empty
            # If preact_orig is empty but batch_tokens_dim > 0, it contributes config.num_features of zeros
            if preact_orig.numel() == 0 and batch_tokens_dim > 0:
                current_num_features = config.num_features
            elif preact_orig.numel() == 0 and batch_tokens_dim == 0:  # if batch is also 0, feature count is 0
                current_num_features = 0

            layer_feature_sizes.append((layer_idx, current_num_features))
            actual_total_concatenated_features += current_num_features
        # If layer_idx is not in preactivations_dict, it contributes 0 features to concatenation

    if actual_total_concatenated_features == 0 and batch_tokens_dim > 0:
        # This can happen if all preactivations_dict entries were empty tensors.
        # The first_valid_preact check above handles if preactivations_dict itself is empty or all values are empty.
        # This is a more specific case where preactivations_dict is not empty, but all tensors within are empty for a non-zero batch_tokens_dim.
        # Create zero tensors for all layers that were in preactivations_dict.
        logger.warning(
            f"Rank {rank}: All preactivation tensors are empty for a non-zero batch size in _apply_batch_topk. Returning zero activations."
        )
        activations_dict_zeros: Dict[int, torch.Tensor] = {}
        for layer_idx_key in preactivations_dict.keys():  # Iterate over original keys that existed
            activations_dict_zeros[layer_idx_key] = torch.zeros(
                (batch_tokens_dim, config.num_features), device=processed_device, dtype=processed_dtype
            )
        return activations_dict_zeros
    elif actual_total_concatenated_features == 0 and batch_tokens_dim == 0:
        return {}  # No tokens, no features, return empty dict

    concatenated_preactivations_original = torch.empty(
        (batch_tokens_dim, actual_total_concatenated_features), device=processed_device, dtype=processed_dtype
    )
    concatenated_preactivations_normalized = torch.empty_like(concatenated_preactivations_original)

    current_feature_offset = 0
    processed_layer_feature_sizes: List[Tuple[int, int]] = []  # Store actual sizes for layers that were processed

    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            num_features_this_layer = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            if preact_orig.numel() == 0 and batch_tokens_dim > 0:
                # This layer had an empty tensor, fill with zeros
                num_features_this_layer = config.num_features  # It will occupy this many features (of zeros)
                concatenated_preactivations_original[
                    :, current_feature_offset : current_feature_offset + num_features_this_layer
                ] = 0
                concatenated_preactivations_normalized[
                    :, current_feature_offset : current_feature_offset + num_features_this_layer
                ] = 0
            elif preact_orig.numel() > 0:
                if preact_orig.shape[0] != batch_tokens_dim:
                    logger.warning(
                        f"Rank {rank}: Inconsistent batch_tokens dim for layer {layer_idx}. "
                        f"Expected {batch_tokens_dim}, got {preact_orig.shape[0]}. Filling with zeros."
                    )
                    # Fill this segment with zeros
                    concatenated_preactivations_original[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = 0
                    concatenated_preactivations_normalized[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = 0
                else:
                    concatenated_preactivations_original[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = preact_orig
                    if preact_orig.numel() > 0:  # Avoid division by zero for empty tensors
                        var, mean = torch.var_mean(
                            preact_orig, dim=0, keepdim=True, correction=0
                        )  # Using Bessel's correction N-1 for std
                        std = torch.sqrt(var)
                        concatenated_preactivations_normalized[
                            :, current_feature_offset : current_feature_offset + num_features_this_layer
                        ] = (preact_orig - mean) / (std + 1e-6)
                    else:  # Should ideally not be hit if preact_orig.numel() > 0 is true, but as safety
                        concatenated_preactivations_normalized[
                            :, current_feature_offset : current_feature_offset + num_features_this_layer
                        ] = 0
            else:  # preact_orig.numel() == 0 and batch_tokens_dim == 0
                num_features_this_layer = 0  # No features if no tokens and no elements

            if num_features_this_layer > 0:  # Only add to processed_layer_feature_sizes if it contributes features
                processed_layer_feature_sizes.append((layer_idx, num_features_this_layer))
            current_feature_offset += num_features_this_layer

    if actual_total_concatenated_features == 0:  # Should be caught earlier, but as a safeguard
        logger.warning(
            f"Rank {rank}: No features to process in _apply_batch_topk after attempting to build concatenated tensors."
        )
        return {}

    k_val_int: int
    if config.batchtopk_k is not None:
        k_val_int = int(config.batchtopk_k)
    else:
        logger.error(f"Rank {rank}: BatchTopK k not specified in config. Defaulting to keeping all features.")
        k_val_int = actual_total_concatenated_features

    if k_val_int > actual_total_concatenated_features:
        logger.warning(
            f"Rank {rank}: k_val_int ({k_val_int}) > actual_total_concatenated_features ({actual_total_concatenated_features}). Clamping k."
        )
        k_val_int = actual_total_concatenated_features

    activated_concatenated = BatchTopK.apply(
        concatenated_preactivations_original,
        float(k_val_int),
        config.batchtopk_straight_through,
        concatenated_preactivations_normalized,
    )

    activations_dict: Dict[int, torch.Tensor] = {}
    current_feature_offset_out = 0
    for original_layer_idx, num_features_processed_this_layer in processed_layer_feature_sizes:
        layer_activated_flat = activated_concatenated[
            :, current_feature_offset_out : current_feature_offset_out + num_features_processed_this_layer
        ]
        activations_dict[original_layer_idx] = layer_activated_flat
        current_feature_offset_out += num_features_processed_this_layer

    return activations_dict


def _apply_token_topk(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    processed_device: torch.device,
    processed_dtype: torch.dtype,
    rank: int = 0,
) -> Dict[int, torch.Tensor]:
    """Applies TokenTopK (TopK per token) to concatenated pre-activations from all layers."""
    if not preactivations_dict:
        return {}

    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: All preactivations are empty in _apply_token_topk. Returning empty dict.")
        return {
            layer_idx: torch.zeros((0, config.num_features), device=processed_device, dtype=processed_dtype)
            for layer_idx in preactivations_dict.keys()
        }

    batch_tokens_dim = first_valid_preact.shape[0]

    # Determine layer_feature_sizes and actual total features (similar to _apply_batch_topk)
    layer_feature_sizes: List[Tuple[int, int]] = []
    actual_total_concatenated_features = 0
    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features
            if preact_orig.numel() == 0 and batch_tokens_dim > 0:
                current_num_features = config.num_features
            elif preact_orig.numel() == 0 and batch_tokens_dim == 0:
                current_num_features = 0

            layer_feature_sizes.append((layer_idx, current_num_features))
            actual_total_concatenated_features += current_num_features

    if actual_total_concatenated_features == 0 and batch_tokens_dim > 0:
        logger.warning(
            f"Rank {rank}: All preactivation tensors are empty for a non-zero batch size in _apply_token_topk. Returning zero activations."
        )
        activations_dict_zeros: Dict[int, torch.Tensor] = {}
        for layer_idx_key in preactivations_dict.keys():
            activations_dict_zeros[layer_idx_key] = torch.zeros(
                (batch_tokens_dim, config.num_features), device=processed_device, dtype=processed_dtype
            )
        return activations_dict_zeros
    elif actual_total_concatenated_features == 0 and batch_tokens_dim == 0:
        return {}

    concatenated_preactivations_original = torch.empty(
        (batch_tokens_dim, actual_total_concatenated_features), device=processed_device, dtype=processed_dtype
    )
    concatenated_preactivations_normalized = torch.empty_like(concatenated_preactivations_original)

    current_feature_offset = 0
    processed_layer_feature_sizes: List[Tuple[int, int]] = []

    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            num_features_this_layer = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            if preact_orig.numel() == 0 and batch_tokens_dim > 0:
                num_features_this_layer = config.num_features
                concatenated_preactivations_original[
                    :, current_feature_offset : current_feature_offset + num_features_this_layer
                ] = 0
                concatenated_preactivations_normalized[
                    :, current_feature_offset : current_feature_offset + num_features_this_layer
                ] = 0
            elif preact_orig.numel() > 0:
                if preact_orig.shape[0] != batch_tokens_dim:
                    logger.warning(
                        f"Rank {rank}: Inconsistent batch_tokens dim for layer {layer_idx} in _apply_token_topk. "
                        f"Expected {batch_tokens_dim}, got {preact_orig.shape[0]}. Filling with zeros."
                    )
                    concatenated_preactivations_original[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = 0
                    concatenated_preactivations_normalized[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = 0
                else:
                    concatenated_preactivations_original[
                        :, current_feature_offset : current_feature_offset + num_features_this_layer
                    ] = preact_orig
                    if preact_orig.numel() > 0:
                        var, mean = torch.var_mean(preact_orig, dim=0, keepdim=True, correction=0)
                        std = torch.sqrt(var)
                        concatenated_preactivations_normalized[
                            :, current_feature_offset : current_feature_offset + num_features_this_layer
                        ] = (preact_orig - mean) / (std + 1e-6)
                    else:
                        concatenated_preactivations_normalized[
                            :, current_feature_offset : current_feature_offset + num_features_this_layer
                        ] = 0
            else:  # preact_orig.numel() == 0 and batch_tokens_dim == 0
                num_features_this_layer = 0

            if num_features_this_layer > 0:
                processed_layer_feature_sizes.append((layer_idx, num_features_this_layer))
            current_feature_offset += num_features_this_layer

    if actual_total_concatenated_features == 0:
        logger.warning(
            f"Rank {rank}: No features to process in _apply_token_topk after attempting to build concatenated tensors."
        )
        return {}

    # Use config.topk_k for TokenTopK
    k_val_float: float
    if config.topk_k is not None:  # Ensure topk_k is part of CLTConfig
        k_val_float = float(config.topk_k)
    else:
        logger.error(f"Rank {rank}: TokenTopK k (topk_k) not specified in config. Defaulting to keeping all features.")
        k_val_float = float(actual_total_concatenated_features)  # Keep all if not specified

    # k_val_float can be a fraction (0<k<1) or an integer count (k>=1) for TokenTopK
    # The TokenTopK.apply function handles the interpretation.

    # Ensure topk_straight_through is part of CLTConfig
    straight_through = getattr(config, "topk_straight_through", True)

    # Call the new TokenTopK autograd function
    activated_concatenated = TokenTopK.apply(
        concatenated_preactivations_original,
        k_val_float,
        straight_through,
        concatenated_preactivations_normalized,  # Use normalized for ranking
    )

    activations_dict: Dict[int, torch.Tensor] = {}
    current_feature_offset_out = 0
    for original_layer_idx, num_features_processed_this_layer in processed_layer_feature_sizes:
        layer_activated_flat = activated_concatenated[
            :, current_feature_offset_out : current_feature_offset_out + num_features_processed_this_layer
        ]
        activations_dict[original_layer_idx] = layer_activated_flat
        current_feature_offset_out += num_features_processed_this_layer

    return activations_dict
