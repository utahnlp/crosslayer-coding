import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, cast
import logging
import torch.distributed as dist
from torch.distributed import ProcessGroup

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

    # Iterate in a deterministic layer order so that all TP ranks execute
    # collective operations (all_gather) in the exact same sequence.
    for layer_idx in sorted(inputs.keys()):
        x = inputs[layer_idx]  # x is x_orig, assumed to be on correct device/dtype
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


def _apply_batch_topk_helper(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,  # Add rank
    process_group: Optional[ProcessGroup],  # Add process_group
) -> Dict[int, torch.Tensor]:
    """Helper to apply BatchTopK globally across concatenated layer pre-activations."""

    if not preactivations_dict:
        logger.warning(f"Rank {rank}: _apply_batch_topk_helper received empty preactivations_dict.")
        return {}

    # --- 1. Concatenate Preactivations (Original and Normalized) ---
    # (Existing concatenation logic remains the same)
    # ... (concatenation logic) ...
    ordered_preactivations_original: List[torch.Tensor] = []
    ordered_preactivations_normalized: List[torch.Tensor] = []
    layer_feature_sizes: List[Tuple[int, int]] = []  # Store (original_layer_idx, num_features)

    # Determine batch dimension (number of tokens) from the first valid tensor
    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: No valid preactivations found in dict for BatchTopK. Returning empty dict.")
        # Return structure matching input keys but with empty tensors
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    # Ensure consistent ordering and handle missing layers
    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            # Ensure preact_orig is on the correct device/dtype already
            # preact_orig = preact_orig.to(device=device, dtype=dtype)

            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            # Handle potentially empty tensors or mismatched batch dims gracefully
            if preact_orig.numel() == 0:
                if batch_tokens_dim > 0:  # Only create zeros if batch dim exists
                    # Use zeros if empty but expected
                    zeros_shape = (batch_tokens_dim, current_num_features)
                    ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                    ordered_preactivations_normalized.append(
                        torch.zeros(zeros_shape, device=device, dtype=dtype)
                    )  # Use zeros for norm too
                    logger.debug(f"Rank {rank} Layer {layer_idx}: Using zeros shape {zeros_shape} for empty input.")
                # else: if batch_tokens_dim is 0, we append nothing, loop continues
            elif preact_orig.shape[0] != batch_tokens_dim:
                # This case indicates inconsistency, log warning and use zeros
                logger.warning(
                    f"Rank {rank} Layer {layer_idx}: Mismatched batch dim ({preact_orig.shape[0]} vs {batch_tokens_dim}). Using zeros."
                )
                zeros_shape = (batch_tokens_dim, current_num_features)
                ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            else:
                # Valid tensor
                ordered_preactivations_original.append(preact_orig)
                # Normalize for ranking (handle potential division by zero)
                mean = preact_orig.mean(dim=0, keepdim=True)
                std = preact_orig.std(dim=0, keepdim=True)
                preact_norm = (preact_orig - mean) / (std + 1e-6)  # Add epsilon for stability
                ordered_preactivations_normalized.append(preact_norm)

            layer_feature_sizes.append((layer_idx, current_num_features))  # Track original layer index and its features
        # else: Layer not in dict, skip

    if not ordered_preactivations_original:
        logger.warning(
            f"Rank {rank}: No tensors collected after iterating layers for BatchTopK. Returning empty activations."
        )
        # Return structure matching input keys but with empty tensors
        return {
            layer_idx: torch.empty((batch_tokens_dim, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()  # Use original keys for structure
        }

    concatenated_preactivations_original = torch.cat(ordered_preactivations_original, dim=1)
    concatenated_preactivations_normalized = torch.cat(ordered_preactivations_normalized, dim=1)

    # --- 2. Apply BatchTopK using Normalized values for ranking ---
    k_val: int
    if config.batchtopk_k is not None:
        k_val = int(config.batchtopk_k)
    else:
        # Default to keeping all features if k is not specified
        k_val = concatenated_preactivations_original.size(1)

    # --- MODIFIED SECTION: Mask Computation and Broadcast ---
    mask_shape = concatenated_preactivations_original.shape
    mask: torch.Tensor

    world_size = 1
    if process_group is not None and dist.is_initialized():
        world_size = dist.get_world_size(process_group)

    if world_size > 1:
        # Use uint8 for NCCL-safe broadcast (bool not reliably supported on CUDA 12.x)
        mask_uint8 = torch.empty(mask_shape, dtype=torch.uint8, device=device)

        if rank == 0:
            local_mask_bool = BatchTopK._compute_mask(
                concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
            )
            mask_uint8.copy_(local_mask_bool.to(torch.uint8))

        # Broadcast uint8 tensor from rank 0
        dist.broadcast(mask_uint8, src=0, group=process_group)
        mask = mask_uint8.to(torch.bool)
    else:
        # Single-GPU (or non-distributed) – compute directly
        mask = BatchTopK._compute_mask(
            concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
        )

    # Apply the identical mask on all ranks
    activated_concatenated = concatenated_preactivations_original * mask.to(dtype)
    # --- END MODIFIED SECTION ---

    # --- 3. Split Concatenated Activations back into Dictionary ---
    # (Existing splitting logic remains the same)
    # ... (splitting logic) ...
    activations_dict: Dict[int, torch.Tensor] = {}
    current_total_feature_offset = 0
    # Use layer_feature_sizes to ensure correct splitting based on original layers/features included
    for original_layer_idx, num_features_this_layer in layer_feature_sizes:
        # Extract the segment corresponding to this original layer
        activated_segment = activated_concatenated[
            :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
        ]
        activations_dict[original_layer_idx] = activated_segment
        current_total_feature_offset += num_features_this_layer

    # --- Optional: Theta Estimation Update ---
    # (Update logic remains the same, uses concatenated tensors before splitting)
    # ... (theta update call) ...

    return activations_dict


def _apply_token_topk_helper(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,  # Add rank
    process_group: Optional[ProcessGroup],  # Add process_group
) -> Dict[int, torch.Tensor]:
    """Helper to apply TokenTopK globally across concatenated layer pre-activations."""

    if not preactivations_dict:
        logger.warning(f"Rank {rank}: _apply_token_topk_helper received empty preactivations_dict.")
        return {}

    # --- 1. Concatenate Preactivations (Original and Normalized) ---
    # (Existing concatenation logic, same as BatchTopK helper)
    # ... (concatenation logic) ...
    ordered_preactivations_original: List[torch.Tensor] = []
    ordered_preactivations_normalized: List[torch.Tensor] = []
    layer_feature_sizes: List[Tuple[int, int]] = []  # Store (original_layer_idx, num_features)

    # Determine batch dimension (number of tokens) from the first valid tensor
    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        logger.warning(f"Rank {rank}: No valid preactivations found in dict for TokenTopK. Returning empty dict.")
        # Return structure matching input keys but with empty tensors
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    batch_tokens_dim = first_valid_preact.shape[0]

    # Ensure consistent ordering and handle missing layers
    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            # preact_orig = preact_orig.to(device=device, dtype=dtype) # Assume already correct

            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features

            # Handle potentially empty tensors or mismatched batch dims gracefully
            if preact_orig.numel() == 0:
                if batch_tokens_dim > 0:
                    zeros_shape = (batch_tokens_dim, current_num_features)
                    ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                    ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            elif preact_orig.shape[0] != batch_tokens_dim:
                logger.warning(
                    f"Rank {rank} Layer {layer_idx}: Mismatched batch dim ({preact_orig.shape[0]} vs {batch_tokens_dim}) for TokenTopK. Using zeros."
                )
                zeros_shape = (batch_tokens_dim, current_num_features)
                ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            else:
                # Valid tensor
                ordered_preactivations_original.append(preact_orig)
                # Normalize for ranking
                mean = preact_orig.mean(dim=0, keepdim=True)
                std = preact_orig.std(dim=0, keepdim=True)
                preact_norm = (preact_orig - mean) / (std + 1e-6)
                ordered_preactivations_normalized.append(preact_norm)

            layer_feature_sizes.append((layer_idx, current_num_features))  # Track original layer index and its features
        # else: Layer not in dict, skip

    if not ordered_preactivations_original:
        logger.warning(
            f"Rank {rank}: No tensors collected after iterating layers for TokenTopK. Returning empty activations."
        )
        # Return structure matching input keys but with empty tensors
        return {
            layer_idx: torch.empty((batch_tokens_dim, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()  # Use original keys for structure
        }

    concatenated_preactivations_original = torch.cat(ordered_preactivations_original, dim=1)
    concatenated_preactivations_normalized = torch.cat(ordered_preactivations_normalized, dim=1)

    # --- 2. Apply TokenTopK using Normalized values for ranking ---
    k_val_float: float
    if hasattr(config, "topk_k") and config.topk_k is not None:
        k_val_float = float(config.topk_k)
    else:
        # Default to keeping all features if k is not specified
        k_val_float = float(concatenated_preactivations_original.size(1))

    # --- MODIFIED SECTION: Mask Computation and Broadcast ---
    mask_shape = concatenated_preactivations_original.shape
    if process_group is not None and dist.is_initialized():
        world_size = dist.get_world_size(process_group)

        if world_size > 1:
            mask_uint8 = torch.empty(mask_shape, dtype=torch.uint8, device=device)

            if rank == 0:
                local_mask_bool = TokenTopK._compute_mask(
                    concatenated_preactivations_original,
                    k_val_float,
                    concatenated_preactivations_normalized,
                )
                mask_uint8.copy_(local_mask_bool.to(torch.uint8))

            dist.broadcast(mask_uint8, src=0, group=process_group)
            mask = mask_uint8.to(torch.bool)
        else:
            mask = TokenTopK._compute_mask(
                concatenated_preactivations_original,
                k_val_float,
                concatenated_preactivations_normalized,
            )
    else:
        mask = TokenTopK._compute_mask(
            concatenated_preactivations_original,
            k_val_float,
            concatenated_preactivations_normalized,
        )

    # Apply the identical mask on all ranks
    activated_concatenated = concatenated_preactivations_original * mask.to(dtype)
    # --- END MODIFIED SECTION ---

    # --- 3. Split Concatenated Activations back into Dictionary ---
    # (Existing splitting logic, same as BatchTopK helper)
    # ... (splitting logic) ...
    activations_dict: Dict[int, torch.Tensor] = {}
    current_total_feature_offset = 0
    # Use layer_feature_sizes to ensure correct splitting based on original layers/features included
    for original_layer_idx, num_features_this_layer in layer_feature_sizes:
        # Extract the segment corresponding to this original layer
        activated_segment = activated_concatenated[
            :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
        ]
        activations_dict[original_layer_idx] = activated_segment
        current_total_feature_offset += num_features_this_layer

    return activations_dict
