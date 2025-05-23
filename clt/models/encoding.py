import torch
from typing import Dict, Optional, Tuple, List
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

# The get_preactivations and _encode_all_layers functions previously here
# have been moved to the clt.models.encoder.Encoder class.


def _apply_batch_topk_helper(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,  # Add rank
    process_group: Optional[ProcessGroup],  # Add process_group
) -> Dict[int, torch.Tensor]:
    """Helper to apply BatchTopK globally across concatenated layer pre-activations."""

    world_size = 1
    if process_group is not None and dist.is_initialized():
        world_size = dist.get_world_size(process_group)

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
            preact_orig = preact_orig.to(device=device, dtype=dtype)

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
    # B = concatenated_preactivations_original.shape[0] # Tokens dim
    # F_total_concat = concatenated_preactivations_original.shape[1]
    # k_total_batch = min(k_val * B, concatenated_preactivations_original.numel()) # Clamp k

    # Compute mask on rank 0 and broadcast
    mask_shape = concatenated_preactivations_original.shape
    mask = torch.empty(mask_shape, dtype=torch.bool, device=device)

    if world_size > 1:
        if rank == 0:
            # Rank 0 computes the mask
            local_mask = BatchTopK._compute_mask(
                concatenated_preactivations_original, k_val, concatenated_preactivations_normalized
            )
            mask.copy_(local_mask)  # Copy computed mask to the buffer
            # Broadcast the mask tensor from rank 0
            dist.broadcast(mask, src=0, group=process_group)
        else:
            # Other ranks receive the broadcasted mask
            dist.broadcast(mask, src=0, group=process_group)
    else:
        # Single GPU case: compute mask directly
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

    world_size = 1
    if process_group is not None and dist.is_initialized():
        world_size = dist.get_world_size(process_group)

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
            # Ensure preact_orig is on the correct device/dtype already
            preact_orig = preact_orig.to(device=device, dtype=dtype)

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
    mask = torch.empty(mask_shape, dtype=torch.bool, device=device)

    if world_size > 1:
        if rank == 0:
            # Rank 0 computes the mask
            local_mask = TokenTopK._compute_mask(  # Use TokenTopK's method
                concatenated_preactivations_original,
                k_val_float,  # Pass float k
                concatenated_preactivations_normalized,
            )
            mask.copy_(local_mask)
            # Broadcast the mask tensor from rank 0
            dist.broadcast(mask, src=0, group=process_group)
        else:
            # Other ranks receive the broadcasted mask
            dist.broadcast(mask, src=0, group=process_group)
    else:
        # Single GPU case: compute mask directly
        mask = TokenTopK._compute_mask(  # Use TokenTopK's method
            concatenated_preactivations_original, k_val_float, concatenated_preactivations_normalized  # Pass float k
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
