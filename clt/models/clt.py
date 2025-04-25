import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple, TYPE_CHECKING, cast
import logging  # Import logging
import torch.distributed as dist

from clt.config import CLTConfig
from clt.models.base import BaseTranscoder
from clt.models.parallel import ColumnParallelLinear, RowParallelLinear  # Import parallel layers

# Type hint for ProcessGroup only during static analysis
if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# Configure logging (or use existing logger if available)
logger = logging.getLogger(__name__)


class JumpReLU(torch.autograd.Function):
    """Custom JumpReLU activation function with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """Forward pass of JumpReLU activation.

        Args:
            input: Input tensor
            threshold: Activation threshold tensor (can be scalar or per-feature)
            bandwidth: Bandwidth parameter for straight-through estimator

        Returns:
            Output tensor with JumpReLU applied
        """
        ctx.save_for_backward(input, threshold)
        ctx.bandwidth = bandwidth

        # JumpReLU: 0 if x < threshold, x if x >= threshold
        return (input >= threshold).float() * input

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Backward pass with straight-through gradient estimator.

        Performs calculations in the original tensor dtypes.

        Args:
            grad_outputs: Gradient(s) from subsequent layer(s)

        Returns:
            Gradients for input, threshold, and None for bandwidth.
        """
        grad_output = grad_outputs[0]  # Unpack the primary gradient
        input, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        # Check which inputs need gradients
        needs_input_grad, needs_threshold_grad, _ = ctx.needs_input_grad

        grad_input = None
        grad_threshold = None

        # 1. Gradient for input (Straight-through estimator)
        if needs_input_grad:
            # Calculate grad_input in original dtype
            # Mask for inputs within the STE window around the threshold
            is_near_threshold = torch.abs(input - threshold) <= (bandwidth / 2.0)
            grad_input = grad_output * is_near_threshold.type_as(grad_output)

        # 2. Gradient for threshold
        if needs_threshold_grad:
            # Calculate grad_threshold in original dtype
            is_near_threshold = torch.abs(input - threshold) <= (bandwidth / 2.0)
            local_grad_theta = (-input / bandwidth) * is_near_threshold.type_as(input)
            grad_threshold_per_element = grad_output * local_grad_theta

            # Sum gradients across non-feature dimensions if threshold is per-feature
            # This assumes threshold might be broadcasted to match input shape
            if grad_threshold_per_element.dim() > threshold.dim():
                dims_to_sum = tuple(range(grad_threshold_per_element.dim() - threshold.dim()))
                grad_threshold = grad_threshold_per_element.sum(dim=dims_to_sum)
                # Ensure shape matches original threshold shape if it wasn't originally scalar
                if threshold.shape != torch.Size([]):
                    grad_threshold = grad_threshold.reshape(threshold.shape)
            else:
                # If threshold was scalar, sum everything
                grad_threshold = grad_threshold_per_element.sum()

        # Return gradients corresponding to the inputs of forward
        # (input, threshold, bandwidth)
        return grad_input, grad_threshold, None


class CrossLayerTranscoder(BaseTranscoder):
    """Implementation of a Cross-Layer Transcoder (CLT) with tensor parallelism."""

    def __init__(
        self,
        config: CLTConfig,
        process_group: Optional[ProcessGroup],  # Allow None for non-distributed
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
                    bias=False,
                    process_group=self.process_group,  # Pass potentially None group
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
                    bias=False,
                    process_group=self.process_group,  # Pass potentially None group
                    input_is_parallel=False,  # Decoder receives full activation, splits internally
                    # Pass model dims needed for init
                    d_model_for_init=config.d_model,
                    num_layers_for_init=config.num_layers,
                )
                for src_layer in range(config.num_layers)
                for tgt_layer in range(src_layer, config.num_layers)
            }
        )

        # Initialize log_threshold parameter - should be replicated on all ranks
        # Gradients will be implicitly averaged by the autograd engine during backward
        # across data parallel replicas, but for TP, we might need manual handling if issues arise.
        # For now, keep as standard parameter.
        initial_threshold_val = torch.ones(config.num_features) * torch.log(torch.tensor(config.jumprelu_threshold))
        self.log_threshold = nn.Parameter(initial_threshold_val.to(device=self.device, dtype=self.dtype))

        self.bandwidth = 1.0  # Bandwidth parameter for straight-through estimator

        # No need to call _init_parameters separately, it's handled in ParallelLinear init
        # self._init_parameters() # Remove this call

        if self.device:
            logger.info(f"CLT TP model initialized on rank {self.rank} device {self.device} with dtype {self.dtype}")
        else:
            logger.info(
                f"CLT TP model initialized on rank {self.rank} with dtype {self.dtype} " f"(device not specified yet)"
            )

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

    def jumprelu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation function."""
        # Threshold is replicated, use it directly
        threshold = torch.exp(self.log_threshold).to(x.device, x.dtype)
        # Apply JumpReLU - This needs the *full* preactivation dimension
        # Cast output to Tensor to satisfy linter
        return cast(torch.Tensor, JumpReLU.apply(x, threshold, self.bandwidth))

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get pre-activation values (full tensor) for features at the specified layer."""
        result: Optional[torch.Tensor] = None
        fallback_shape: Optional[Tuple[int, int]] = None

        try:
            # 1. Check input shape and reshape if necessary
            if x.dim() == 2:
                input_for_linear = x
            elif x.dim() == 3:
                batch, seq_len, d_model = x.shape
                if d_model != self.config.d_model:
                    # Handle d_model mismatch early
                    logger.warning(
                        f"Rank {self.rank}: Input d_model {d_model} != config {self.config.d_model} layer {layer_idx}"
                    )
                    fallback_shape = (batch * seq_len, self.config.num_features)
                else:
                    input_for_linear = x.reshape(-1, d_model)
            else:
                logger.warning(
                    f"Rank {self.rank}: Cannot handle input shape {x.shape} for preactivations layer {layer_idx}"
                )
                fallback_shape = (0, self.config.num_features)

            # 2. Check d_model match if not already done
            if fallback_shape is None and input_for_linear.shape[1] != self.config.d_model:
                logger.warning(
                    f"Rank {self.rank}: Input d_model {input_for_linear.shape[1]} != config {self.config.d_model} layer {layer_idx}"
                )
                fallback_shape = (input_for_linear.shape[0], self.config.num_features)

            # 3. Proceed if no errors so far
            if fallback_shape is None:
                # Explicitly cast the output of the parallel linear layer
                result = cast(torch.Tensor, self.encoders[layer_idx](input_for_linear))

        except IndexError:
            logger.error(f"Rank {self.rank}: Invalid layer index {layer_idx} requested for encoder.")
            fallback_batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
            fallback_shape = (fallback_batch_dim, self.config.num_features)
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error during get_preactivations layer {layer_idx}: {e}", exc_info=True)
            fallback_batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
            fallback_shape = (fallback_batch_dim, self.config.num_features)

        # 4. Return result or fallback tensor
        if result is not None:
            return result
        else:
            # Ensure fallback_shape is defined if result is None
            if fallback_shape is None:
                # This case should ideally not happen if logic above is correct
                logger.error(
                    f"Rank {self.rank}: Fallback shape not determined for layer {layer_idx}, returning empty tensor."
                )
                fallback_shape = (0, self.config.num_features)
            return torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.

        Returns the *full* feature activations after nonlinearity.

        Args:
            x: Input activations [batch_size, seq_len, d_model] or [batch_tokens, d_model]
            layer_idx: Index of the layer

        Returns:
            Encoded activations after nonlinearity [..., num_features]
        """
        # Ensure input is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

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
                fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
            elif preact.shape[-1] != self.config.num_features:
                logger.warning(
                    f"Rank {self.rank}: Received invalid preactivations shape {preact.shape} for encode layer {layer_idx}."
                )
                fallback_shape = (preact.shape[0], self.config.num_features)  # Try to keep batch dim
                fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
            else:
                # Apply activation function to the full preactivation tensor
                if self.config.activation_fn == "jumprelu":
                    activated = self.jumprelu(preact)
                else:  # "relu"
                    activated = F.relu(preact)

        except Exception as e:
            logger.error(f"Rank {self.rank}: Error during encode layer {layer_idx}: {e}", exc_info=True)
            batch_dim = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1] if x.dim() == 3 else 0
            fallback_shape = (batch_dim, self.config.num_features)
            fallback_tensor = torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

        # Return activated tensor or fallback
        if activated is not None:
            return activated
        elif fallback_tensor is not None:
            return fallback_tensor
        else:
            # Should not happen, but return empty tensor as last resort
            logger.error(f"Rank {self.rank}: Failed to encode or create fallback for layer {layer_idx}.")
            return torch.zeros((0, self.config.num_features), device=self.device, dtype=self.dtype)

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
        # Encode inputs at each layer -> returns *full* activations
        activations = {}
        for layer_idx, x in inputs.items():
            activations[layer_idx] = self.encode(x, layer_idx)

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

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to *full* feature activations [..., num_features]
        """
        activations = {}
        for layer_idx, x in inputs.items():
            try:
                # encode() returns the full activation tensor
                act = self.encode(x, layer_idx)
                activations[layer_idx] = act
            except Exception as e:
                # Log the error but continue trying other layers
                logger.error(
                    f"Rank {self.rank}: Error getting feature activations for layer {layer_idx}: {e}", exc_info=True
                )
                # Optionally, return a fallback or skip this layer
                # For now, just log and continue
                pass
        return activations

    def get_decoder_norms(self) -> torch.Tensor:
        """Get L2 norms of all decoder matrices for each feature (gathered across ranks).

        Returns:
            Tensor of shape [num_layers, num_features] containing decoder norms
        """
        full_decoder_norms = torch.zeros(
            self.config.num_layers, self.config.num_features, device=self.device, dtype=self.dtype  # Match model dtype
        )

        # Use self.world_size which is correctly set for non-distributed case
        rank = self.rank

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
                start_idx = rank * local_dim_padded
                end_idx_padded = start_idx + local_dim_padded

                # The actual end index in the original dimension might be smaller
                actual_end_idx = min(end_idx_padded, full_dim)
                actual_local_dim = max(0, actual_end_idx - start_idx)

                # If this rank has valid features for this layer
                if actual_local_dim > 0:
                    # The norms correspond to the first `actual_local_dim` columns of the weight
                    valid_norms_sq = current_norms_sq[:actual_local_dim]

                    # Ensure shapes match before adding
                    if valid_norms_sq.shape[0] == actual_local_dim:
                        global_slice = slice(start_idx, actual_end_idx)
                        local_norms_sq_accum[global_slice] += valid_norms_sq
                    else:
                        # This should not happen with the slicing logic above
                        logger.warning(
                            f"Rank {rank}: Shape mismatch in decoder norm calculation for {decoder_key}. "
                            f"Valid norms shape {valid_norms_sq.shape}, expected size {actual_local_dim}."
                        )

            # Reduce the accumulated squared norms across all ranks
            if self.process_group is not None and dist.is_initialized():
                dist.all_reduce(local_norms_sq_accum, op=dist.ReduceOp.SUM, group=self.process_group)

            # Now take the square root and store in the final tensor (cast back to model dtype)
            full_decoder_norms[src_layer] = torch.sqrt(local_norms_sq_accum).to(self.dtype)

        return full_decoder_norms

    # Add save/load methods that handle sharded parameters
    # For now, rely on Trainer using FSDP-style full state dict save/load logic
    # or implement manual gathering/scattering.
    # def save(self, path: str):
    #     # Gather parameters on rank 0 and save
    #     pass
    #
    # @classmethod
    # def load(cls, path: str, process_group: ProcessGroup, device: torch.device):
    #     # Load on rank 0, broadcast/scatter to other ranks
    #     pass
