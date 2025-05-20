import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple, cast, List
import logging  # Import logging
import torch.distributed as dist

from clt.config import CLTConfig
from clt.models.base import BaseTranscoder
from clt.models.parallel import ColumnParallelLinear, RowParallelLinear  # Import parallel layers
from clt.models.activations import BatchTopK, JumpReLU, TokenTopK  # Import BatchTopK, JumpReLU and TokenTopK

# Import the new encoding helper functions
from clt.models.encoding import get_preactivations as _get_preactivations_helper
from clt.models.encoding import _encode_all_layers as _encode_all_layers_helper
from clt.models.encoding import _apply_batch_topk_helper
from clt.models.encoding import _apply_token_topk_helper

from torch.distributed import ProcessGroup

from . import mark_replicated  # Added import

# Configure logging (or use existing logger if available)
logger = logging.getLogger(__name__)


class CrossLayerTranscoder(BaseTranscoder):
    """Implementation of a Cross-Layer Transcoder (CLT) with tensor parallelism."""

    # --- Cache --- #
    _cached_decoder_norms: Optional[torch.Tensor] = None
    _min_selected_preact: Optional[torch.Tensor]

    def __init__(
        self,
        config: CLTConfig,
        process_group: Optional["ProcessGroup"],  # Allow None for non-distributed
        device: Optional[torch.device] = None,
    ):
        """Initialize the Cross-Layer Transcoder.

        Args:
            config: Configuration for the transcoder
            process_group: The process group for tensor parallelism (or None)
            device: Optional device to initialize the model parameters on.
        """
        super().__init__(config)

        self.process_group = process_group
        # Handle non-distributed case for rank/world_size
        if process_group is None or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
            self.process_group = None  # Ensure it's None
        else:
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)

        self.device = device  # Store device if provided
        self.dtype = self._resolve_dtype(config.clt_dtype)

        # Create encoder matrices for each layer using ColumnParallelLinear
        self.encoders = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features=config.d_model,
                    out_features=config.num_features,
                    bias=True,
                    process_group=self.process_group,  # Pass potentially None group
                    device=self.device,  # Pass device
                )
                for _ in range(config.num_layers)
            ]
        )

        # Create decoder matrices for each layer pair using RowParallelLinear
        self.decoders = nn.ModuleDict(
            {
                f"{src_layer}->{tgt_layer}": RowParallelLinear(
                    in_features=config.num_features,  # Full feature dim
                    out_features=config.d_model,
                    bias=True,
                    process_group=self.process_group,  # Pass potentially None group
                    input_is_parallel=False,  # Decoder receives full activation, splits internally
                    # Pass model dims needed for init
                    d_model_for_init=config.d_model,
                    num_layers_for_init=config.num_layers,
                    device=self.device,  # Pass device
                )
                for src_layer in range(config.num_layers)
                for tgt_layer in range(src_layer, config.num_layers)
            }
        )

        # Initialize log_threshold parameter - should be replicated on all ranks
        # Gradients will be implicitly averaged by the autograd engine during backward
        # across data parallel replicas, but for TP, we might need manual handling if issues arise.
        # For now, keep as standard parameter.
        if self.config.activation_fn == "jumprelu":
            # Initialize per-layer thresholds
            initial_threshold_val = torch.ones(
                config.num_layers, config.num_features  # Shape: [num_layers, num_features]
            ) * torch.log(torch.tensor(config.jumprelu_threshold))
            self.log_threshold = nn.Parameter(initial_threshold_val.to(device=self.device, dtype=self.dtype))
            mark_replicated(self.log_threshold)  # Mark as replicated

        self.bandwidth = 1.0  # Bandwidth parameter for straight-through estimator

        # Buffers for on-the-fly theta estimation (BatchTopK → JumpReLU)
        # Create the attributes for type consistency, will be populated by estimate_theta_posthoc if needed
        self.register_buffer("_sum_min_selected_preact", None, persistent=False)
        self.register_buffer("_count_min_selected_preact", None, persistent=False)

        # No need to call _init_parameters separately, it's handled in ParallelLinear init
        # self._init_parameters() # Remove this call

        if self.device:
            logger.info(f"CLT TP model initialized on rank {self.rank} device {self.device} with dtype {self.dtype}")
        else:
            logger.info(
                f"CLT TP model initialized on rank {self.rank} with dtype {self.dtype} " f"(device not specified yet)"
            )

    def _get_current_op_device_dtype(self, x_sample: Optional[torch.Tensor] = None) -> Tuple[torch.device, torch.dtype]:
        """Resolves the device and dtype for an operation.
        Prioitizes self.device and self.dtype if set, otherwise infers from x_sample or defaults.
        """
        current_op_device = self.device
        current_op_dtype = self.dtype  # Should be set by _resolve_dtype in __init__

        if current_op_device is None:
            if x_sample is not None and x_sample.numel() > 0:
                current_op_device = x_sample.device
            else:
                # Fallback if model device is None and no sample tensor provided
                current_op_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Log this fallback if it happens outside of initial self.device setting
                # However, self.device should ideally be set by get_feature_activations if initially None.

        # self.dtype should always be valid after __init__ due to _resolve_dtype.
        # If it were somehow None, a fallback would be needed:
        if current_op_dtype is None:  # Defensive coding, should ideally not be reached.
            if x_sample is not None and x_sample.numel() > 0 and isinstance(x_sample.dtype, torch.dtype):
                current_op_dtype = x_sample.dtype
            else:
                current_op_dtype = torch.float32
            logger.warning(f"Rank {self.rank}: CLT self.dtype was unexpectedly None. Defaulting to {current_op_dtype}.")

        return current_op_device, current_op_dtype

    def _resolve_dtype(self, dtype_input: Optional[Union[str, torch.dtype]]) -> torch.dtype:
        """Converts string dtype names to torch.dtype objects, defaulting to float32."""
        if isinstance(dtype_input, torch.dtype):
            return dtype_input
        if isinstance(dtype_input, str):
            try:
                dtype = getattr(torch, dtype_input)
                if isinstance(dtype, torch.dtype):
                    return dtype
                else:
                    logger.warning(f"Resolved '{dtype_input}' but it is not a torch.dtype. " f"Defaulting to float32.")
                    return torch.float32
            except AttributeError:
                logger.warning(f"Unsupported CLT dtype string: '{dtype_input}'. " f"Defaulting to float32.")
                return torch.float32
        return torch.float32

    def jumprelu(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply JumpReLU activation function for a specific layer."""
        # Select the threshold for the given layer
        if not hasattr(self, "log_threshold") or self.log_threshold is None:
            # This should ideally not happen if config.activation_fn == "jumprelu"
            logger.error(f"Rank {self.rank}: log_threshold not initialized for JumpReLU. Returning input.")
            return x
        if layer_idx >= self.log_threshold.shape[0]:
            logger.error(
                f"Rank {self.rank}: Invalid layer_idx {layer_idx} for log_threshold with shape {self.log_threshold.shape}. Returning input."
            )
            return x

        threshold = torch.exp(self.log_threshold[layer_idx]).to(x.device, x.dtype)
        # Apply JumpReLU - This needs the *full* preactivation dimension
        # Cast output to Tensor to satisfy linter
        return cast(torch.Tensor, JumpReLU.apply(x, threshold, self.bandwidth))

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get pre-activation values (full tensor) for features at the specified layer."""
        # Resolve device and dtype for this operation, similar to encode()
        # current_op_device = self.device
        # current_op_dtype = self.dtype

        # if current_op_device is None:
        #     current_op_device = x.device if x.numel() > 0 else torch.device("cpu")
        # if current_op_dtype is None:
        #     current_op_dtype = x.dtype if x.numel() > 0 and isinstance(x.dtype, torch.dtype) else torch.float32
        current_op_device, current_op_dtype = self._get_current_op_device_dtype(x)

        # Ensure input is on the correct device and dtype before passing to helper
        x_processed = x.to(device=current_op_device, dtype=current_op_dtype)

        return _get_preactivations_helper(
            x_processed,
            layer_idx,
            self.config,
            self.encoders,
            current_op_device,
            current_op_dtype,
            self.rank,
        )

    def _encode_all_layers(
        self, inputs: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[int, int, int]], torch.device, torch.dtype]:
        """Encodes inputs for all layers and returns pre-activations and original shape info."""
        # Determine effective device and dtype for this batch, inferring if necessary
        effective_device = self.device
        effective_dtype = self.dtype

        if effective_device is None or effective_dtype is None:
            first_input_tensor = next((t for t in inputs.values() if t.numel() > 0), None)
            if first_input_tensor is not None:
                if effective_device is None:
                    effective_device = first_input_tensor.device
                if effective_dtype is None:
                    effective_dtype = (
                        first_input_tensor.dtype if isinstance(first_input_tensor.dtype, torch.dtype) else torch.float32
                    )
            else:  # No valid inputs to infer from, and model device/dtype are None
                if effective_device is None:
                    effective_device = torch.device("cpu")
                if effective_dtype is None:
                    effective_dtype = torch.float32
                logger.warning(
                    f"Rank {self.rank}: Could not infer device/dtype from inputs for _encode_all_layers, "
                    f"and model defaults are None. Using {effective_device}/{effective_dtype}."
                )

        # Ensure all input tensors are on the determined effective device and dtype
        processed_inputs: Dict[int, torch.Tensor] = {}
        for layer_idx, x_orig in inputs.items():
            processed_inputs[layer_idx] = x_orig.to(device=effective_device, dtype=effective_dtype)

        # Call the helper function with processed inputs and determined device/dtype
        # The helper returns the device and dtype it operated on, which should match effective_device and effective_dtype
        preactivations_dict, original_shapes_info, returned_device, returned_dtype = _encode_all_layers_helper(
            processed_inputs, self.config, self.encoders, effective_device, effective_dtype, self.rank
        )
        # The returned_device and returned_dtype from the helper reflect what was used.
        return preactivations_dict, original_shapes_info, returned_device, returned_dtype

    @torch.no_grad()
    def _update_min_selected_preactivations(
        self,
        concatenated_preactivations_original: torch.Tensor,
        activated_concatenated: torch.Tensor,
        layer_feature_sizes: List[Tuple[int, int]],
    ):
        """
        Updates the _min_selected_preact buffer with minimum pre-activation values
        for features selected by BatchTopK during the current step.
        This function operates with no_grad.
        """
        if (
            not hasattr(self, "_sum_min_selected_preact")
            or self._sum_min_selected_preact is None
            or self._count_min_selected_preact is None
        ):
            if self.config.activation_fn == "batchtopk":
                logger.warning(f"Rank {self.rank}: running BatchTopK stats buffers not found. Skipping theta update.")
            return

        assert self._sum_min_selected_preact is not None and isinstance(
            self._sum_min_selected_preact, torch.Tensor
        ), f"Rank {self.rank}: _sum_min_selected_preact is not a Tensor or is None."
        assert self._count_min_selected_preact is not None and isinstance(
            self._count_min_selected_preact, torch.Tensor
        ), f"Rank {self.rank}: _count_min_selected_preact is not a Tensor or is None."

        current_total_feature_offset = 0
        for i, (original_layer_idx, num_features_this_layer) in enumerate(layer_feature_sizes):
            if original_layer_idx >= self._sum_min_selected_preact.shape[0]:
                logger.warning(
                    f"Rank {self.rank}: Invalid original_layer_idx {original_layer_idx} for _min_selected_preact update. Skipping layer."
                )
                current_total_feature_offset += num_features_this_layer
                continue

            preact_orig_this_layer = concatenated_preactivations_original[
                :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
            ]
            gated_acts_segment = activated_concatenated[
                :, current_total_feature_offset : current_total_feature_offset + num_features_this_layer
            ]

            if gated_acts_segment.shape == preact_orig_this_layer.shape:
                # Vectorised per-feature min calculation that avoids CPU-only ops like nonzero on MPS.
                mask_active = gated_acts_segment > 0  # Active features after gating

                if mask_active.any():
                    # Replace inactive entries by +inf and take per-feature minimum across tokens
                    masked_preact = torch.where(
                        mask_active,
                        preact_orig_this_layer,
                        torch.full_like(preact_orig_this_layer, float("inf")),
                    )

                    per_feature_min_this_batch = masked_preact.amin(dim=0)

                    if logger.isEnabledFor(logging.DEBUG):
                        # Log characteristics of the minimums being used for theta estimation
                        finite_mins_for_log = per_feature_min_this_batch[torch.isfinite(per_feature_min_this_batch)]
                        if finite_mins_for_log.numel() > 0:
                            logger.debug(
                                f"Rank {self.rank} Layer {original_layer_idx}: per_feature_min_this_batch (finite values for log) "
                                f"min={finite_mins_for_log.min().item():.4f}, "
                                f"max={finite_mins_for_log.max().item():.4f}, "
                                f"mean={finite_mins_for_log.mean().item():.4f}, "
                                f"median={torch.median(finite_mins_for_log).item():.4f}"
                            )
                        else:
                            logger.debug(
                                f"Rank {self.rank} Layer {original_layer_idx}: No finite per_feature_min_this_batch values to log stats for."
                            )

                        # Log how many original pre-activations were negative but still contributed to a positive gated_act
                        original_preacts_leading_to_positive_gated = preact_orig_this_layer[mask_active]
                        if original_preacts_leading_to_positive_gated.numel() > 0:  # Check if tensor is not empty
                            num_negative_contrib = (original_preacts_leading_to_positive_gated < 0).sum().item()
                            if num_negative_contrib > 0:
                                logger.debug(
                                    f"Rank {self.rank} Layer {original_layer_idx}: {num_negative_contrib} negative original pre-activations "
                                    f"(out of {mask_active.sum().item()} active selections) contributed to theta estimation via positive gated_acts_segment."
                                )

                    # Update running sum and count for expected-value calculation
                    valid_mask = torch.isfinite(per_feature_min_this_batch)

                    self._sum_min_selected_preact[original_layer_idx, valid_mask] += per_feature_min_this_batch[
                        valid_mask
                    ]
                    self._count_min_selected_preact[original_layer_idx, valid_mask] += 1
            else:
                logger.warning(
                    f"Rank {self.rank}: Shape mismatch for theta update, layer {original_layer_idx}. "
                    f"Original: {preact_orig_this_layer.shape}, Gated: {gated_acts_segment.shape}"
                )

            current_total_feature_offset += num_features_this_layer

        # Function now purely updates the buffer – it no longer recurses or returns a value.

    def _apply_batch_topk(
        self,
        preactivations_dict: Dict[int, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        """Applies BatchTopK to concatenated pre-activations from all layers by calling the helper."""
        return _apply_batch_topk_helper(preactivations_dict, self.config, device, dtype, self.rank, self.process_group)

    def _apply_token_topk(
        self,
        preactivations_dict: Dict[int, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        """Applies TokenTopK to concatenated pre-activations from all layers by calling the helper."""
        return _apply_token_topk_helper(preactivations_dict, self.config, device, dtype, self.rank, self.process_group)

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.

        Returns the *full* feature activations after nonlinearity.
        This method is used for 'relu' and 'jumprelu' activations.
        For 'batchtopk', use get_feature_activations.

        Args:
            x: Input activations [batch_size, seq_len, d_model] or [batch_tokens, d_model]
            layer_idx: Index of the layer

        Returns:
            Encoded activations after nonlinearity [..., num_features]
        """
        # Resolve device and dtype for this operation
        # current_op_device = self.device
        # current_op_dtype = self.dtype

        # if current_op_device is None:
        #     current_op_device = x.device if x.numel() > 0 else torch.device("cpu")
        # if current_op_dtype is None:
        #     current_op_dtype = x.dtype if x.numel() > 0 and isinstance(x.dtype, torch.dtype) else torch.float32
        current_op_device, current_op_dtype = self._get_current_op_device_dtype(x)

        # Ensure input is on the correct device and dtype
        x = x.to(device=current_op_device, dtype=current_op_dtype)

        fallback_tensor: Optional[torch.Tensor] = None
        activated: Optional[torch.Tensor] = None

        try:
            # Get full preactivations [..., num_features]
            preact = self.get_preactivations(x, layer_idx)

            if preact.numel() == 0:
                # If preactivations failed or returned empty, create fallback based on expected shape
                logger.warning(f"Rank {self.rank}: Received empty preactivations for encode layer {layer_idx}.")
                batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
                fallback_shape = (batch_dim, self.config.num_features)
                fallback_tensor = torch.zeros(fallback_shape, device=current_op_device, dtype=current_op_dtype)
            elif preact.shape[-1] != self.config.num_features:
                logger.warning(
                    f"Rank {self.rank}: Received invalid preactivations shape {preact.shape} for encode layer {layer_idx}."
                )
                fallback_shape = (preact.shape[0], self.config.num_features)  # Try to keep batch dim
                fallback_tensor = torch.zeros(fallback_shape, device=current_op_device, dtype=current_op_dtype)
            else:
                # Apply activation function to the full preactivation tensor
                if self.config.activation_fn == "jumprelu":
                    activated = self.jumprelu(preact, layer_idx)
                elif self.config.activation_fn == "relu":  # "relu"
                    activated = F.relu(preact)
                elif self.config.activation_fn == "batchtopk":
                    # This 'encode' method should ideally not be called directly for batchtopk if
                    # get_feature_activations is used as the main entry point for it.
                    # However, if it is, we apply BatchTopK to this single layer's preactivations.
                    # This might not be the intended global behavior but handles the case.
                    logger.warning(
                        f"Rank {self.rank}: 'encode' called for BatchTopK on layer {layer_idx}. This applies TopK per-layer, not globally. Use 'get_feature_activations' for global BatchTopK."
                    )
                    k_val_local_int: int
                    if self.config.batchtopk_k is not None:
                        k_val_local_int = int(self.config.batchtopk_k)
                    else:
                        k_val_local_int = preact.size(1)

                    activated = BatchTopK.apply(preact, float(k_val_local_int), self.config.batchtopk_straight_through)

                elif self.config.activation_fn == "topk":  # Added handling for 'topk'
                    logger.warning(
                        f"Rank {self.rank}: 'encode' called for TopK on layer {layer_idx}. This applies TopK per-layer, not globally. Use 'get_feature_activations' for global TopK."
                    )
                    k_val_local_float: float
                    if hasattr(self.config, "topk_k") and self.config.topk_k is not None:
                        k_val_local_float = float(self.config.topk_k)
                    else:  # Default to keeping all features for this layer if topk_k not set
                        k_val_local_float = float(preact.size(1))

                    # Get topk_straight_through from config, default to True
                    straight_through_local = getattr(self.config, "topk_straight_through", True)
                    # TokenTopK.apply takes the original preactivation, k, straight_through, and optional ranking tensor
                    # For per-layer application, we don't have a separate normalized ranking tensor readily available here,
                    # so we pass preact itself for ranking. Normalization would ideally happen inside TokenTopK if needed
                    # or be handled if this path were to be seriously supported (it's a fallback/warning path).
                    activated = TokenTopK.apply(preact, k_val_local_float, straight_through_local, preact)

        except Exception as e:
            logger.error(f"Rank {self.rank}: Error during encode layer {layer_idx}: {e}", exc_info=True)
            batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
            fallback_shape = (batch_dim, self.config.num_features)
            fallback_tensor = torch.zeros(fallback_shape, device=current_op_device, dtype=current_op_dtype)

        # Return activated tensor or fallback
        if activated is not None:
            return activated
        elif fallback_tensor is not None:
            return fallback_tensor
        else:
            # Should not happen, but return empty tensor as last resort
            logger.error(f"Rank {self.rank}: Failed to encode or create fallback for layer {layer_idx}.")
            return torch.zeros((0, self.config.num_features), device=current_op_device, dtype=current_op_dtype)

    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Decode the feature activations to reconstruct outputs at the specified layer.

        Input activations `a` are expected to be the *full* tensors.
        The RowParallelLinear decoder splits them internally.

        Args:
            a: Dictionary mapping layer indices to *full* feature activations [..., num_features]
            layer_idx: Index of the layer to reconstruct outputs for

        Returns:
            Reconstructed outputs [..., d_model]
        """
        available_keys = sorted(a.keys())
        if not available_keys:
            logger.warning(f"Rank {self.rank}: No activation keys available in decode method for layer {layer_idx}")
            return torch.zeros((0, self.config.d_model), device=self.device, dtype=self.dtype)

        first_key = available_keys[0]
        example_tensor = a[first_key]
        # Need batch dimension size for reconstruction tensor
        # Handle cases where example_tensor might be empty (though filtered earlier)
        batch_dim_size = example_tensor.shape[0] if example_tensor.numel() > 0 else 0
        # If batch_dim_size is still 0, try finding a non-empty tensor
        if batch_dim_size == 0:
            for key in available_keys:
                if a[key].numel() > 0:
                    batch_dim_size = a[key].shape[0]
                    break

        reconstruction = torch.zeros((batch_dim_size, self.config.d_model), device=self.device, dtype=self.dtype)

        # Sum contributions from features at all contributing layers
        for src_layer in range(layer_idx + 1):
            if src_layer in a:
                # Decoder expects full activation tensor [..., num_features]
                activation_tensor = a[src_layer].to(device=self.device, dtype=self.dtype)

                # Check activation tensor shape
                if activation_tensor.numel() == 0:
                    continue  # Skip empty activations
                if activation_tensor.shape[-1] != self.config.num_features:
                    logger.warning(
                        f"Rank {self.rank}: Activation tensor for layer {src_layer} has incorrect feature dimension {activation_tensor.shape[-1]}, expected {self.config.num_features}. Skipping decode contribution."
                    )
                    continue

                decoder = self.decoders[f"{src_layer}->{layer_idx}"]
                try:
                    # RowParallelLinear takes full input (input_is_parallel=False),
                    # splits it internally, computes local result, and all-reduces.
                    decoded = decoder(activation_tensor)
                    reconstruction += decoded
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Error during decode from src {src_layer} to tgt {layer_idx}: {e}",
                        exc_info=True,
                    )

        return reconstruction

    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Process inputs through the parallel transcoder model.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to reconstructed outputs
        """
        # Get feature activations based on the configured activation function
        activations = self.get_feature_activations(inputs)

        # Decode to reconstruct outputs at each layer
        reconstructions = {}
        for layer_idx in range(self.config.num_layers):
            # Check if any relevant *full* activations exist before decoding
            relevant_activations = {k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0}
            if layer_idx in inputs and relevant_activations:
                try:
                    # Decode takes the dictionary of *full* activations
                    reconstructions[layer_idx] = self.decode(relevant_activations, layer_idx)
                except Exception as e:
                    logger.error(
                        f"Rank {self.rank}: Error during forward pass decode for layer {layer_idx}: {e}",
                        exc_info=True,
                    )
                    # Determine batch size from input if possible
                    batch_size = 0
                    if layer_idx in inputs:
                        input_tensor = inputs[layer_idx]
                        if input_tensor.dim() >= 1:
                            batch_size = input_tensor.shape[0]
                    # Fallback if batch size cannot be determined
                    if batch_size == 0:
                        logger.warning(
                            f"Could not determine batch size for fallback tensor in layer {layer_idx}. Using 0."
                        )

                    reconstructions[layer_idx] = torch.zeros(
                        (batch_size, self.config.d_model),
                        device=self.device,
                        dtype=self.dtype,
                    )

        return reconstructions

    def get_feature_activations(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Get *full* feature activations for all layers.

        Handles different activation functions including global BatchTopK.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to *full* feature activations [..., num_features]
        """
        # Resolve model device and dtype if they haven't been set yet.
        if self.device is None or self.dtype is None:  # self.dtype should be set from _resolve_dtype
            first_input_tensor = next((t for t in inputs.values() if t.numel() > 0), None)
            if self.device is None:  # Infer self.device if not set
                if first_input_tensor is not None:
                    self.device = first_input_tensor.device
                    logger.info(
                        f"Rank {self.rank}: Inferred and set model device: {self.device} from first input tensor."
                    )
                else:
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    logger.warning(
                        f"Rank {self.rank}: Defaulting and setting model device to {self.device} due to empty inputs and no device set."
                    )
            # self.dtype is expected to be set by __init__ via _resolve_dtype.
            # If it were None here, it would indicate an issue with __init__ or subsequent modification.
            if self.dtype is None:  # Defensive, should not happen.
                if first_input_tensor is not None and isinstance(first_input_tensor.dtype, torch.dtype):
                    self.dtype = first_input_tensor.dtype
                else:
                    self.dtype = torch.float32  # Absolute fallback
                logger.error(
                    f"Rank {self.rank}: Inferred and set model dtype: {self.dtype} in get_feature_activations. This is unexpected."
                )

        # Now self.device and self.dtype should be valid.
        # For operations within this function, we can use them directly or use the new helper for local context if needed.
        # current_op_device, current_op_dtype = self._get_current_op_device_dtype(first_input_tensor if 'first_input_tensor' in locals() else None)

        # Ensure all input tensors are on the model's (now resolved) device and dtype
        processed_inputs: Dict[int, torch.Tensor] = {}
        for layer_idx, x_orig in inputs.items():
            processed_inputs[layer_idx] = x_orig.to(device=self.device, dtype=self.dtype)

        if self.config.activation_fn == "batchtopk" or self.config.activation_fn == "topk":
            # _encode_all_layers_helper expects device and dtype to be passed, use self.device/self.dtype
            preactivations_dict, _, processed_device, processed_dtype = _encode_all_layers_helper(
                processed_inputs, self.config, self.encoders, self.device, self.dtype, self.rank
            )

            if not preactivations_dict:
                activations = {}
                dev_fallback = processed_device
                dt_fallback = processed_dtype
                for layer_idx_orig_input in inputs.keys():
                    x_orig_input = inputs[layer_idx_orig_input]
                    batch_dim_fallback = 0
                    if x_orig_input.dim() == 3:  # B, S, D
                        batch_dim_fallback = x_orig_input.shape[0] * x_orig_input.shape[1]
                    elif x_orig_input.dim() == 2:  # B, D or B*S, D
                        batch_dim_fallback = x_orig_input.shape[0]
                    # else: batch_dim_fallback remains 0 if shape is unusual or empty

                    activations[layer_idx_orig_input] = torch.zeros(
                        (batch_dim_fallback, self.config.num_features), device=dev_fallback, dtype=dt_fallback
                    )
                return activations

            if self.config.activation_fn == "batchtopk":
                # Pass rank and process_group to the helper
                activations = _apply_batch_topk_helper(
                    preactivations_dict, self.config, processed_device, processed_dtype, self.rank, self.process_group
                )
            elif self.config.activation_fn == "topk":
                # Pass rank and process_group to the helper
                activations = _apply_token_topk_helper(
                    preactivations_dict, self.config, processed_device, processed_dtype, self.rank, self.process_group
                )
            return activations
        else:  # ReLU or JumpReLU (per-layer activation)
            activations = {}
            # Iterate layers in deterministic ascending order so all ranks
            # invoke the same collective operations in the same sequence.
            for layer_idx in sorted(processed_inputs.keys()):
                x_input = processed_inputs[layer_idx]
                try:
                    # encode() returns the full activation tensor after per-layer ReLU/JumpReLU
                    # encode() itself handles moving x_input to the correct device/dtype internally
                    act = self.encode(x_input, layer_idx)
                    activations[layer_idx] = act
                except Exception as e:
                    # Log the error but continue trying other layers
                    logger.error(
                        f"Rank {self.rank}: Error getting feature activations for layer {layer_idx} (fn: {self.config.activation_fn}): {e}",
                        exc_info=True,
                    )
                    # Fallback: return zero tensor of expected shape for this layer
                    # Determine batch size from input if possible
                    if x_input.dim() >= 1:
                        pass  # This variable is not used for the fallback tensor construction below
                        if x_input.dim() == 3:  # if [B,S,D] -> get B*S from preact if possible, or B here
                            pass  # get_preactivations handles this, encode will use its output shape

                    # Try to infer batch_dim for fallback
                    actual_batch_dim = 0  # Initialize actual_batch_dim
                    if hasattr(x_input, "shape") and len(x_input.shape) > 0:
                        actual_batch_dim = x_input.shape[0]
                        if len(x_input.shape) == 3:  # B, S, D_model
                            actual_batch_dim = x_input.shape[0] * x_input.shape[1]
                    # else: # fallback if x is weird, actual_batch_dim remains 0

                    activations[layer_idx] = torch.zeros(
                        (actual_batch_dim, self.config.num_features), device=self.device, dtype=self.dtype
                    )
            return activations

    def get_decoder_norms(self) -> torch.Tensor:
        """Get L2 norms of all decoder matrices for each feature (gathered across ranks).

        Returns:
            Tensor of shape [num_layers, num_features] containing decoder norms
        """
        # --- Use Cache --- #
        if self._cached_decoder_norms is not None:
            return self._cached_decoder_norms

        full_decoder_norms = torch.zeros(
            self.config.num_layers, self.config.num_features, device=self.device, dtype=self.dtype  # Match model dtype
        )

        # Use self.world_size which is correctly set for non-distributed case
        # rank = self.rank # Removed unused variable

        for src_layer in range(self.config.num_layers):
            # Accumulate squared norms locally first, then reduce
            # Need full size for indexing, but will only fill the local slice for this rank
            local_norms_sq_accum = torch.zeros(
                self.config.num_features, device=self.device, dtype=torch.float32  # Use float32 for accumulation
            )

            for tgt_layer in range(src_layer, self.config.num_layers):
                decoder_key = f"{src_layer}->{tgt_layer}"
                decoder = self.decoders[decoder_key]
                assert isinstance(decoder, RowParallelLinear), f"Decoder {decoder_key} is not RowParallelLinear"

                # decoder.weight shape: [d_model, local_num_features (padded)]
                # Calculate norms on local weight shard
                current_norms_sq = torch.norm(decoder.weight, dim=0).pow(2).to(torch.float32)
                # current_norms_sq shape: [local_num_features (padded)]

                # Determine the slice of the *full* feature dimension this rank owns
                full_dim = decoder.full_in_features  # Original number of features
                local_dim_padded = decoder.local_in_features  # Padded local size

                # Calculate start and end indices in the *full* dimension
                # Correct calculation using integer division based on full dimension
                features_per_rank = (full_dim + self.world_size - 1) // self.world_size
                start_idx = self.rank * features_per_rank
                end_idx = min(start_idx + features_per_rank, full_dim)
                actual_local_dim = max(0, end_idx - start_idx)

                # Check if local padded size matches expected local dimension
                # This is a sanity check for RowParallelLinear's partitioning logic
                if local_dim_padded != features_per_rank and self.rank == self.world_size - 1:
                    # The last rank might have fewer features if full_dim is not divisible by world_size
                    # RowParallelLinear pads its weight, so local_dim_padded might be larger than actual_local_dim
                    pass  # Padding is expected here
                elif local_dim_padded != actual_local_dim and local_dim_padded != features_per_rank:
                    logger.warning(
                        f"Rank {self.rank}: Padded local dim ({local_dim_padded}) doesn't match calculated actual local dim ({actual_local_dim}) or features_per_rank ({features_per_rank}) for {decoder_key}. This might indicate an issue with RowParallelLinear partitioning."
                    )
                    # Proceed cautiously, but log the potential discrepancy

                # If this rank has valid features for this layer (based on correct calculation)
                if actual_local_dim > 0:
                    # The norms correspond to the first `actual_local_dim` columns of the weight
                    # We slice the norms up to the *actual* number of features this rank owns, ignoring padding
                    valid_norms_sq = current_norms_sq[:actual_local_dim]

                    # Ensure shapes match before adding
                    if valid_norms_sq.shape[0] == actual_local_dim:
                        # Accumulate into the correct global slice determined by start_idx and end_idx
                        global_slice = slice(start_idx, end_idx)
                        local_norms_sq_accum[global_slice] += valid_norms_sq
                    else:
                        # This should not happen with the slicing logic above
                        logger.warning(
                            f"Rank {self.rank}: Shape mismatch in decoder norm calculation for {decoder_key}. "
                            f"Valid norms shape {valid_norms_sq.shape}, expected size {actual_local_dim}."
                        )

            # Reduce the accumulated squared norms across all ranks
            # Each feature's decoder weight vector lives entirely on a single rank
            # (row-parallel sharding over the feature dimension).  To reconstruct the
            # correct global ‖w‖₂ we must therefore **sum** the per-rank contributions,
            # not average them – averaging would shrink every norm by `world_size` and
            # drastically weaken the sparsity penalty.
            if self.process_group is not None and dist.is_initialized():
                dist.all_reduce(local_norms_sq_accum, op=dist.ReduceOp.SUM, group=self.process_group)

            # Now take the square root and store in the final tensor (cast back to model dtype)
            full_decoder_norms[src_layer] = torch.sqrt(local_norms_sq_accum).to(self.dtype)

        # --- Populate Cache --- #
        self._cached_decoder_norms = full_decoder_norms

        return full_decoder_norms

    @torch.no_grad()
    def estimate_theta_posthoc(
        self,
        data_iter: torch.utils.data.IterableDataset,  # More generic iterable
        num_batches: Optional[int] = None,
        default_theta_value: float = 1e6,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Estimate theta post-hoc using a specified number of batches.

        Args:
            data_iter: An iterable yielding (inputs, targets) batches.
            num_batches: Number of batches to process for estimation. If None, iterates through all.
            default_theta_value: Value for features never activated. (Note: currently not directly used in final theta calculation in convert_to_jumprelu_inplace)
            device: Device to run estimation on.

        Returns:
            The estimated theta tensor.
        """
        original_device = next(self.parameters()).device
        target_device = device if device is not None else self.device
        if target_device is None:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Rank {self.rank}: Starting post-hoc theta estimation on device {target_device}.")

        self.eval()  # Set model to evaluation mode
        if target_device != original_device:
            self.to(target_device)

        # Initialize buffers for summing normalized preactivations and counts
        if not hasattr(self, "_sum_min_selected_preact") or self._sum_min_selected_preact is None:
            self.register_buffer(
                "_sum_min_selected_preact",
                torch.zeros(
                    (self.config.num_layers, self.config.num_features),
                    dtype=self.dtype,
                    device=target_device,
                ),
                persistent=False,
            )
        else:
            # Ensure it's on the correct device and zeroed out
            self._sum_min_selected_preact = self._sum_min_selected_preact.to(target_device)
            self._sum_min_selected_preact.data.zero_()

        if not hasattr(self, "_count_min_selected_preact") or self._count_min_selected_preact is None:
            self.register_buffer(
                "_count_min_selected_preact",
                torch.zeros(
                    (self.config.num_layers, self.config.num_features),
                    dtype=self.dtype,
                    device=target_device,
                ),
                persistent=False,
            )
        else:
            self._count_min_selected_preact = self._count_min_selected_preact.to(target_device)
            self._count_min_selected_preact.data.zero_()

        # Initialize buffers for averaging layer-wise normalization statistics (mu and sigma)
        buffer_shape = (self.config.num_layers, self.config.num_features)
        if not hasattr(self, "_avg_layer_means") or self._avg_layer_means is None:
            self.register_buffer(
                "_avg_layer_means", torch.zeros(buffer_shape, dtype=self.dtype, device=target_device), persistent=False
            )
        else:
            self._avg_layer_means = self._avg_layer_means.to(target_device)
            self._avg_layer_means.data.zero_()

        if not hasattr(self, "_avg_layer_stds") or self._avg_layer_stds is None:
            self.register_buffer(
                "_avg_layer_stds", torch.zeros(buffer_shape, dtype=self.dtype, device=target_device), persistent=False
            )
        else:
            self._avg_layer_stds = self._avg_layer_stds.to(target_device)
            self._avg_layer_stds.data.zero_()

        if not hasattr(self, "_processed_batches_for_stats") or self._processed_batches_for_stats is None:
            self.register_buffer(
                "_processed_batches_for_stats",
                torch.zeros(self.config.num_layers, dtype=torch.long, device=target_device),
                persistent=False,
            )
        else:
            self._processed_batches_for_stats = self._processed_batches_for_stats.to(target_device)
            self._processed_batches_for_stats.data.zero_()

        processed_batches_total = 0

        try:
            from tqdm.auto import tqdm

            iterable_data_iter = (
                tqdm(data_iter, total=num_batches, desc=f"Estimating Theta & Stats (Rank {self.rank})")
                if num_batches
                else tqdm(data_iter, desc=f"Estimating Theta & Stats (Rank {self.rank})")
            )
        except ImportError:
            logger.info("tqdm not found, proceeding without progress bar for theta estimation.")
            iterable_data_iter = data_iter

        for inputs, _ in iterable_data_iter:
            if num_batches is not None and processed_batches_total >= num_batches:
                break

            inputs_on_device = {k: v.to(target_device) for k, v in inputs.items()}
            preactivations_dict, _, _, _ = self._encode_all_layers(inputs_on_device)

            if not preactivations_dict:
                logger.warning(f"Rank {self.rank}: No preactivations. Skipping batch {processed_batches_total + 1}.")
                processed_batches_total += 1
                continue

            first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
            if first_valid_preact is None:
                logger.warning(
                    f"Rank {self.rank}: All preactivations empty. Skipping batch {processed_batches_total + 1}."
                )
                processed_batches_total += 1
                continue

            ordered_preactivations_original_posthoc: List[torch.Tensor] = []
            ordered_preactivations_normalized_posthoc: List[torch.Tensor] = []
            layer_feature_sizes_posthoc: List[Tuple[int, int]] = []
            batch_tokens_dim_posthoc = first_valid_preact.shape[0]

            for layer_idx_loop in range(self.config.num_layers):
                num_feat_for_layer: int
                mean_loop: Optional[torch.Tensor] = None
                std_loop: Optional[torch.Tensor] = None
                preact_norm_loop: Optional[torch.Tensor] = None

                if layer_idx_loop in preactivations_dict:
                    preact_orig_loop = preactivations_dict[layer_idx_loop]
                    num_feat_for_layer = (
                        preact_orig_loop.shape[1] if preact_orig_loop.numel() > 0 else self.config.num_features
                    )

                    if preact_orig_loop.shape[0] != batch_tokens_dim_posthoc and preact_orig_loop.numel() > 0:
                        logger.warning(
                            f"Rank {self.rank} Layer {layer_idx_loop}: Mismatched token dim (expected {batch_tokens_dim_posthoc}, got {preact_orig_loop.shape[0]}). Using zeros."
                        )
                        mean_loop = torch.zeros((1, num_feat_for_layer), device=target_device, dtype=self.dtype)
                        std_loop = torch.ones((1, num_feat_for_layer), device=target_device, dtype=self.dtype)
                        preact_norm_loop = torch.zeros(
                            (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                        )
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)
                    elif preact_orig_loop.numel() == 0 and batch_tokens_dim_posthoc > 0:
                        mean_loop = torch.zeros((1, num_feat_for_layer), device=target_device, dtype=self.dtype)
                        std_loop = torch.ones((1, num_feat_for_layer), device=target_device, dtype=self.dtype)
                        preact_norm_loop = torch.zeros(
                            (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                        )
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)
                    elif preact_orig_loop.numel() > 0:
                        mean_loop = preact_orig_loop.mean(dim=0, keepdim=True)
                        std_loop = preact_orig_loop.std(dim=0, keepdim=True)
                        preact_norm_loop = (preact_orig_loop - mean_loop) / (std_loop + 1e-6)  # Add epsilon to std_loop
                        ordered_preactivations_original_posthoc.append(preact_orig_loop)
                        ordered_preactivations_normalized_posthoc.append(preact_norm_loop)

                        # Accumulate means and stds for this layer
                        # These buffers were already ensured to be on target_device
                        self._avg_layer_means.data[layer_idx_loop] += mean_loop.squeeze().clone()
                        self._avg_layer_stds.data[layer_idx_loop] += std_loop.squeeze().clone()
                        self._processed_batches_for_stats.data[layer_idx_loop] += 1
                    else:  # Layer in dict, but preact_orig_loop is empty and batch_tokens_dim_posthoc is 0 - num_feat_for_layer is from config
                        num_feat_for_layer = self.config.num_features  # Fallback
                        # No data to append or normalize, but need to track for layer_feature_sizes_posthoc
                else:  # layer_idx_loop not in preactivations_dict
                    num_feat_for_layer = self.config.num_features  # Fallback
                    if batch_tokens_dim_posthoc > 0:
                        ordered_preactivations_original_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                            )
                        )
                        ordered_preactivations_normalized_posthoc.append(
                            torch.zeros(
                                (batch_tokens_dim_posthoc, num_feat_for_layer), device=target_device, dtype=self.dtype
                            )
                        )

                layer_feature_sizes_posthoc.append((layer_idx_loop, num_feat_for_layer))

            if not ordered_preactivations_normalized_posthoc or not any(
                t.numel() > 0 for t in ordered_preactivations_normalized_posthoc
            ):
                logger.warning(
                    f"Rank {self.rank}: No normalized preactivations. Skipping batch {processed_batches_total + 1}."
                )
                processed_batches_total += 1
                continue

            # Use normalized preactivations for ranking, but original for BatchTopK/TokenTopK values if available
            # If original list is empty/all-empty, use normalized for values too (as a fallback)
            if not ordered_preactivations_original_posthoc or not any(
                t.numel() > 0 for t in ordered_preactivations_original_posthoc
            ):
                concatenated_preactivations_for_gating = torch.cat(ordered_preactivations_normalized_posthoc, dim=1)
                logger.debug(
                    f"Rank {self.rank} Batch {processed_batches_total + 1}: Using normalized preactivations for gating due to empty/all-empty original list."
                )
            else:
                concatenated_preactivations_for_gating = torch.cat(ordered_preactivations_original_posthoc, dim=1)

            concatenated_preactivations_for_ranking = torch.cat(ordered_preactivations_normalized_posthoc, dim=1)

            activated_concatenated_posthoc: Optional[torch.Tensor] = None
            if self.config.activation_fn == "batchtopk":
                k_val_int = (
                    int(self.config.batchtopk_k)
                    if self.config.batchtopk_k is not None
                    else concatenated_preactivations_for_gating.size(1)
                )
                # batchtopk_straight_through is expected to be in config (defaults to True in CLTConfig)
                straight_through_btk = self.config.batchtopk_straight_through
                activated_concatenated_posthoc = BatchTopK.apply(
                    concatenated_preactivations_for_gating,
                    float(k_val_int),
                    straight_through_btk,
                    concatenated_preactivations_for_ranking,
                )
            elif self.config.activation_fn == "topk":
                if not hasattr(self.config, "topk_k") or self.config.topk_k is None:
                    logger.error(
                        f"Rank {self.rank}: 'topk_k' not found in config for 'topk' activation during theta estimation. Defaulting to all features for this batch."
                    )
                    k_val_float = float(concatenated_preactivations_for_gating.size(1))  # Keep all
                else:
                    k_val_float = float(self.config.topk_k)

                straight_through_tk = getattr(self.config, "topk_straight_through", True)
                activated_concatenated_posthoc = TokenTopK.apply(
                    concatenated_preactivations_for_gating,
                    k_val_float,
                    straight_through_tk,
                    concatenated_preactivations_for_ranking,
                )
            else:
                logger.error(
                    f"Rank {self.rank}: Unsupported activation_fn '{self.config.activation_fn}' for theta estimation. Cannot determine gating mechanism. Using zeros for activated_concatenated_posthoc."
                )
                activated_concatenated_posthoc = torch.zeros_like(concatenated_preactivations_for_gating)

            # Update sum/count stats using NORMALIZED preactivations for selected features
            if activated_concatenated_posthoc is not None:  # Ensure it was set
                self._update_min_selected_preactivations(
                    concatenated_preactivations_for_ranking,  # Sum of *normalized* values
                    activated_concatenated_posthoc,
                    layer_feature_sizes_posthoc,
                )
            processed_batches_total += 1

        logger.info(
            f"Rank {self.rank}: Processed {processed_batches_total} batches for theta estimation and stats accumulation."
        )

        # Finalize average mu and sigma if the buffers exist (they should if estimation ran)
        if (
            hasattr(self, "_processed_batches_for_stats")
            and self._processed_batches_for_stats is not None
            and hasattr(self, "_avg_layer_means")
            and self._avg_layer_means is not None
            and hasattr(self, "_avg_layer_stds")
            and self._avg_layer_stds is not None
        ):

            active_stat_batches = self._processed_batches_for_stats.data.unsqueeze(-1).clamp_min(
                1.0
            )  # ensure broadcasting
            self._avg_layer_means.data /= active_stat_batches
            self._avg_layer_stds.data /= active_stat_batches
            logger.info(f"Rank {self.rank}: Averaged layer-wise normalization stats computed.")
        else:
            logger.warning(f"Rank {self.rank}: Could not finalize normalization stats, buffers missing.")

        self.convert_to_jumprelu_inplace(default_theta_value=default_theta_value)

        # Clean up non-persistent buffers
        for buf_name in [
            "_sum_min_selected_preact",
            "_count_min_selected_preact",
            "_avg_layer_means",
            "_avg_layer_stds",
            "_processed_batches_for_stats",
        ]:
            if hasattr(self, buf_name):
                delattr(self, buf_name)

        if target_device != original_device:
            self.to(original_device)

        logger.info(f"Rank {self.rank}: Post-hoc theta estimation and conversion to JumpReLU complete.")
        return torch.exp(self.log_threshold.data)

    @torch.no_grad()
    def convert_to_jumprelu_inplace(self, default_theta_value: float = 1e6) -> None:
        """
        Converts the model to use JumpReLU activation based on learned BatchTopK thresholds.
        This method should be called after training with BatchTopK.
        It finalizes the _min_selected_preact buffer (expected to contain sums of *normalized* preacts)
        and uses _avg_layer_means, _avg_layer_stds for unnormalization.
        Updates the model config and sets the log_threshold parameter.

        Args:
            default_theta_value: Value for features never activated (used in initial per-feature calculation in normalized space).
                                 (Note: This parameter is not directly used in the current implementation's final theta calculation,
                                  as behavior for non-activating features is handled by fallback_norm_theta_value and clamping.)
        """
        if self.config.activation_fn not in ["batchtopk", "topk"]:
            logger.warning(
                f"Rank {self.rank}: Model original activation_fn was {self.config.activation_fn}, not batchtopk or topk. "
                "Skipping conversion to JumpReLU based on learned thetas."
            )
            if self.config.activation_fn == "relu":  # Keep this specific error for ReLU
                logger.error(f"Rank {self.rank}: Model is ReLU, cannot convert to JumpReLU via learned thetas.")
            return

        required_buffers = [
            "_sum_min_selected_preact",
            "_count_min_selected_preact",
            "_avg_layer_means",
            "_avg_layer_stds",
        ]
        for buf_name in required_buffers:
            if not hasattr(self, buf_name) or getattr(self, buf_name) is None:
                raise RuntimeError(
                    f"Rank {self.rank}: Required buffer {buf_name} for JumpReLU conversion not found or not populated. "
                    "Run estimate_theta_posthoc() with appropriate settings before converting."
                )

        assert isinstance(self._sum_min_selected_preact, torch.Tensor)
        assert isinstance(self._count_min_selected_preact, torch.Tensor)
        assert isinstance(self._avg_layer_means, torch.Tensor)
        assert isinstance(self._avg_layer_stds, torch.Tensor)

        logger.info(
            f"Rank {self.rank}: Starting conversion of BatchTopK model to JumpReLU (per-layer avg norm. theta, then unnormalize)."
        )

        # These sums/counts are of NORMALIZED preactivation values
        theta_sum_norm = self._sum_min_selected_preact.clone()
        theta_cnt_norm = self._count_min_selected_preact.clone()

        avg_mus = self._avg_layer_means.clone()
        avg_sigmas = self._avg_layer_stds.clone()

        if self.process_group is not None and dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(theta_sum_norm, op=dist.ReduceOp.SUM, group=self.process_group)
            dist.all_reduce(theta_cnt_norm, op=dist.ReduceOp.SUM, group=self.process_group)
            # Mus and Sigmas should have been averaged per rank over their batches,
            # then all-reduced if they were supposed to be global pre-defined stats.
            # For now, assuming estimate_theta_posthoc gives each rank the same avg_mus and avg_sigmas
            # (e.g. from rank 0, or each rank computes them identically on its data shard then averages).
            # If they were calculated independently per rank on sharded data without final sync,
            # they might differ. The current setup in estimate_theta_posthoc has each rank calculate its own.
            # For consistent unnormalization, all ranks should use the same mu/sigma for a given layer.
            # Let's assume for now estimate_theta_posthoc has made them consistent or this is handled by estimate_theta_posthoc
            # For a truly robust solution, mus and sigmas would also need an all_reduce sum and divide by world_size * num_batches_per_rank.
            # The current `active_stat_batches` division in estimate_theta_posthoc is per-rank, then averaged here.
            # Let's assume the per-rank averaged mus/sigmas are what we want to use for unnormalizing that rank's part.
            # But the final log_threshold must be identical. So the unnormalization must use globally agreed mu/sigma.
            # Simplest: AllReduce sum for avg_mus * counts and avg_sigmas * counts, and sum counts, then divide.
            # OR, more simply, after local averaging in estimate_theta_posthoc, all_reduce sum them and divide by world_size.
            dist.all_reduce(avg_mus, op=dist.ReduceOp.SUM, group=self.process_group)
            avg_mus /= self.world_size
            dist.all_reduce(avg_sigmas, op=dist.ReduceOp.SUM, group=self.process_group)
            avg_sigmas /= self.world_size
            logger.info(f"Rank {self.rank}: AllReduced and averaged mu/sigma for unnormalization across ranks.")

        # Initialize the final RAW theta tensor (will store per-feature raw thresholds)
        theta_raw = torch.zeros_like(theta_sum_norm)
        fallback_norm_theta_value = 1e-5  # Fallback for a layer's normalized theta

        for l_idx in range(self.config.num_layers):
            layer_theta_sum_norm = theta_sum_norm[l_idx]  # Sums of min selected *normalized* preacts
            layer_theta_cnt_norm = theta_cnt_norm[l_idx]  # Counts for these

            active_mask_layer = layer_theta_cnt_norm > 0
            # Per-feature expected values in NORMALIZED space
            per_feature_thetas_norm_layer = torch.full_like(layer_theta_sum_norm, float("inf"))

            if active_mask_layer.any():
                per_feature_thetas_norm_layer[active_mask_layer] = layer_theta_sum_norm[
                    active_mask_layer
                ] / layer_theta_cnt_norm[active_mask_layer].clamp_min(1.0)

            finite_positive_thetas_norm_layer = per_feature_thetas_norm_layer[
                torch.isfinite(per_feature_thetas_norm_layer) & (per_feature_thetas_norm_layer > 0)
            ]

            # SCALAR threshold in NORMALIZED space for this layer
            theta_norm_scalar_for_this_layer: float
            if finite_positive_thetas_norm_layer.numel() > 0:
                theta_norm_scalar_for_this_layer = finite_positive_thetas_norm_layer.mean().item()
                logger.info(
                    f"Rank {self.rank} Layer {l_idx}: Derived normalized theta (scalar, mean of positive active features) = {theta_norm_scalar_for_this_layer:.4e}"
                )
            else:
                theta_norm_scalar_for_this_layer = fallback_norm_theta_value
                logger.warning(
                    f"Rank {self.rank} Layer {l_idx}: No positive, finite per-feature normalized thetas. Using fallback normalized theta = {theta_norm_scalar_for_this_layer:.4e}"
                )

            # Un-normalize to get RAW thresholds PER FEATURE for this layer
            mu_vec_layer = avg_mus[l_idx]  # Shape: [num_features]
            sigma_vec_layer = avg_sigmas[l_idx].clamp_min(1e-6)  # Shape: [num_features], clamp std

            # theta_norm_scalar_for_this_layer will be broadcast
            theta_raw_vec_for_layer = theta_norm_scalar_for_this_layer * sigma_vec_layer + mu_vec_layer
            theta_raw[l_idx] = theta_raw_vec_for_layer

            if self.rank == 0 and l_idx < 5:  # Log first few layers for detail
                logger.info(
                    f"Rank 0 Layer {l_idx}: Normalized Theta_scalar={theta_norm_scalar_for_this_layer:.3e}. Mu (sample): {mu_vec_layer[:3].tolist()}. Sigma (sample): {sigma_vec_layer[:3].tolist()}. Raw Theta (sample): {theta_raw_vec_for_layer[:3].tolist()}"
                )

        logger.info(f"Rank {self.rank}: Per-feature raw thresholds computed via unnormalization.")

        # This count is based on NORMALIZED stats. It tells how many features never had normalized stats.
        num_norm_feat_no_stats = (theta_cnt_norm == 0).sum().item()
        logger.info(
            f"Rank {self.rank}: Number of features that had no BatchTopK stats (norm counts==0) across all layers: {num_norm_feat_no_stats}"
        )

        if self.rank == 0:
            logger.info(f"Rank {self.rank}: Final RAW Theta stats (per-feature, shape {theta_raw.shape}):")
            for l_idx in range(self.config.num_layers):
                layer_raw_thetas = theta_raw[l_idx]
                logger.info(
                    f"  Layer {l_idx}: min={layer_raw_thetas.min().item():.4e}, mean={layer_raw_thetas.mean().item():.4e}, max={layer_raw_thetas.max().item():.4e}"
                )
            try:
                import wandb

                if wandb.run:
                    for l_idx in range(self.config.num_layers):
                        layer_raw_thetas_for_hist = theta_raw[l_idx].cpu().float()
                        finite_layer_raw_thetas = layer_raw_thetas_for_hist[
                            torch.isfinite(layer_raw_thetas_for_hist) & (layer_raw_thetas_for_hist > 0)
                        ]  # Ensure positive for log10
                        if finite_layer_raw_thetas.numel() > 0:
                            wandb.log(
                                {
                                    f"debug/theta_layer_{l_idx}_raw_dist_log10": wandb.Histogram(
                                        torch.log10(finite_layer_raw_thetas).tolist()
                                    )
                                },
                                commit=False,
                            )
                        else:
                            logger.debug(
                                f"Rank {self.rank}: Layer {l_idx} had no finite positive raw thetas for histogram."
                            )

                    # Log overall min/max/mean of all raw thetas
                    all_raw_thetas_flat = theta_raw.flatten().cpu().float()
                    finite_all_raw_thetas = all_raw_thetas_flat[
                        torch.isfinite(all_raw_thetas_flat) & (all_raw_thetas_flat > 0)
                    ]
                    if finite_all_raw_thetas.numel() > 0:
                        wandb.log(
                            {
                                "debug/theta_raw_overall_min_log10": torch.log10(finite_all_raw_thetas.min()).item(),
                                "debug/theta_raw_overall_max_log10": torch.log10(finite_all_raw_thetas.max()).item(),
                                "debug/theta_raw_overall_mean_log10": torch.log10(finite_all_raw_thetas.mean()).item(),
                            },
                            commit=False,
                        )

            except ImportError:
                logger.info("WandB not installed, skipping raw theta distribution logging.")
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error logging raw theta distributions to WandB: {e}")

        # Clamp final raw thetas before log to ensure they are positive for torch.log
        # This primarily handles cases where mu was very negative and sigma very small, pulling a small positive norm_theta negative.
        min_final_raw_theta = 1e-7  # Very small positive value
        num_clamped_final = (theta_raw < min_final_raw_theta).sum().item()
        if num_clamped_final > 0:
            logger.warning(
                f"Rank {self.rank}: Clamping {num_clamped_final} final raw theta values below {min_final_raw_theta} to {min_final_raw_theta} before taking log."
            )
            theta_raw.clamp_min_(min_final_raw_theta)

        log_theta = torch.log(theta_raw)

        # Update config
        original_activation_fn = self.config.activation_fn  # Store before changing
        self.config.activation_fn = "jumprelu"
        # The original jumprelu_threshold in config is a scalar, now we have per-feature, per-layer.
        # The JumpReLU function itself uses self.log_threshold if available.
        # We mark the original config field to signify it's superseded.
        self.config.jumprelu_threshold = 0.0  # Mark as effectively superseded

        if original_activation_fn == "batchtopk":
            self.config.batchtopk_k = None
            # batchtopk_straight_through is bool. Set to False as it's no longer actively used.
            self.config.batchtopk_straight_through = False
        elif original_activation_fn == "topk":
            if hasattr(self.config, "topk_k"):  # Not in CLTConfig, so check hasattr
                del self.config.topk_k  # Dynamically added, so can be deleted
            if hasattr(self.config, "topk_straight_through"):  # Not in CLTConfig
                del self.config.topk_straight_through  # Dynamically added

        # Create or update self.log_threshold as an nn.Parameter
        if not hasattr(self, "log_threshold") or self.log_threshold is None:
            self.log_threshold = nn.Parameter(log_theta.to(device=self.device, dtype=self.dtype))
        else:
            if not isinstance(self.log_threshold, nn.Parameter):
                # If it exists but is not a Parameter, re-assign it as one
                self.log_threshold = nn.Parameter(
                    log_theta.to(device=self.log_threshold.device, dtype=self.log_threshold.dtype)
                )
            else:
                # Update data in-place, ensuring it's on the correct device and dtype
                self.log_threshold.data = log_theta.to(device=self.log_threshold.device, dtype=self.log_threshold.dtype)

        mark_replicated(self.log_threshold)  # Mark as replicated after creation or update

        logger.info(f"Rank {self.rank}: Model converted to JumpReLU. activation_fn='{self.config.activation_fn}'.")
        if self.rank == 0:
            min_log_thresh = (
                self.log_threshold.data.min().item() if self.log_threshold.data.numel() > 0 else float("nan")
            )
            max_log_thresh = (
                self.log_threshold.data.max().item() if self.log_threshold.data.numel() > 0 else float("nan")
            )
            mean_log_thresh = (
                self.log_threshold.data.mean().item() if self.log_threshold.data.numel() > 0 else float("nan")
            )
            logger.info(
                f"Rank {self.rank}: Final log_threshold stats: min={min_log_thresh:.4f}, max={max_log_thresh:.4f}, mean={mean_log_thresh:.4f}"
            )
