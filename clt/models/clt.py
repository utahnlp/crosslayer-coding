import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
import logging  # Import logging
import math  # Import math for sqrt

from clt.config import CLTConfig
from clt.models.base import BaseTranscoder

# Configure logging (or use existing logger if available)
logger = logging.getLogger(__name__)


class JumpReLU(torch.autograd.Function):
    """Custom JumpReLU activation function with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, input, threshold=0.03, bandwidth=1.0):
        """Forward pass of JumpReLU activation.

        Args:
            input: Input tensor
            threshold: Activation threshold
            bandwidth: Bandwidth parameter for straight-through estimator

        Returns:
            Output tensor with JumpReLU applied
        """
        # Ensure threshold is on the same device and has a suitable dtype for comparison
        # Use the input tensor's properties as reference
        threshold_tensor = torch.tensor(threshold, device=input.device)
        ctx.save_for_backward(input, threshold_tensor)
        ctx.bandwidth = bandwidth

        # JumpReLU: 0 if x < threshold, x if x >= threshold
        # Ensure comparison happens with compatible types
        return (input >= threshold_tensor.to(input.dtype)).float() * input

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through gradient estimator.

        Performs calculations in float32 for stability and casts results back.

        Args:
            grad_output: Gradient from subsequent layer

        Returns:
            Gradients for input and log_threshold.
        """
        input_original_dtype, threshold_original_dtype = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Store original dtypes to cast back later
        original_input_dtype = input_original_dtype.dtype
        original_threshold_dtype = threshold_original_dtype.dtype

        # --- Perform calculations in float32 for stability --- #
        input_fp32 = input_original_dtype.float()
        threshold_fp32 = threshold_original_dtype.float()
        grad_output_fp32 = grad_output.float()
        bandwidth_fp32 = float(bandwidth)

        # Mask for inputs within the STE window around the threshold
        is_near_threshold = torch.abs(input_fp32 - threshold_fp32) <= (
            bandwidth_fp32 / 2.0
        )
        is_near_threshold_float = is_near_threshold.float()

        # 1. Gradient for input (Straight-through estimator)
        grad_input_fp32 = grad_output_fp32 * is_near_threshold_float

        # 2. Gradient for threshold
        local_grad_theta_fp32 = (-input_fp32 / bandwidth_fp32) * is_near_threshold_float
        grad_threshold_per_element_fp32 = grad_output_fp32 * local_grad_theta_fp32

        # Sum gradients across batch/sequence dimensions
        if grad_threshold_per_element_fp32.dim() > 1:
            dims_to_sum = tuple(range(grad_threshold_per_element_fp32.dim() - 1))
            grad_threshold_fp32 = grad_threshold_per_element_fp32.sum(dim=dims_to_sum)
        else:
            grad_threshold_fp32 = grad_threshold_per_element_fp32

        # --- Cast gradients back to original dtypes --- #
        grad_input = grad_input_fp32.to(original_input_dtype)
        grad_threshold = grad_threshold_fp32.to(original_threshold_dtype)

        # Return gradients for input, threshold, and None for bandwidth
        # Note: We return grad for threshold, not log_threshold directly here.
        # The autograd engine handles the chain rule back to log_threshold parameter.
        return grad_input, grad_threshold, None


class CrossLayerTranscoder(BaseTranscoder):
    """Implementation of a Cross-Layer Transcoder (CLT)."""

    def __init__(self, config: CLTConfig, device: Optional[torch.device] = None):
        """Initialize the Cross-Layer Transcoder.

        Args:
            config: Configuration for the transcoder
            device: Optional device to initialize the model parameters on.
        """
        super().__init__(config)

        self.device = device  # Store device if provided
        self.dtype = self._resolve_dtype(config.clt_dtype)

        # Determine device and dtype for layer creation
        creation_kwargs = {"device": self.device, "dtype": self.dtype}

        # Create encoder matrices for each layer
        self.encoders = nn.ModuleList(
            [
                nn.Linear(
                    config.d_model, config.num_features, bias=False, **creation_kwargs
                )
                for _ in range(config.num_layers)
            ]
        )

        # Create decoder matrices for each layer pair
        self.decoders = nn.ModuleDict(
            {
                f"{src_layer}->{tgt_layer}": nn.Linear(
                    config.num_features, config.d_model, bias=False, **creation_kwargs
                )
                for src_layer in range(config.num_layers)
                for tgt_layer in range(src_layer, config.num_layers)
            }
        )

        # Initialize log_threshold parameter with correct dtype and device
        initial_threshold_val = torch.ones(config.num_features) * torch.log(
            torch.tensor(config.jumprelu_threshold)
        )
        self.log_threshold = nn.Parameter(
            initial_threshold_val.to(device=self.device, dtype=self.dtype)
        )

        self.bandwidth = 1.0  # Bandwidth parameter for straight-through estimator

        self._init_parameters()  # Call initialization method

        if self.device:
            logger.info(
                f"CLT model initialized on {self.device} with dtype {self.dtype}"
            )
        else:
            logger.info(
                f"CLT model initialized with dtype {self.dtype} (device not specified yet)"
            )

    def _resolve_dtype(
        self, dtype_input: Optional[Union[str, torch.dtype]]
    ) -> torch.dtype:
        """Converts string dtype names to torch.dtype objects, defaulting to float32."""
        if isinstance(dtype_input, torch.dtype):
            return dtype_input
        if isinstance(dtype_input, str):
            try:
                dtype = getattr(torch, dtype_input)
                if isinstance(dtype, torch.dtype):
                    return dtype
                else:
                    logger.warning(
                        f"Resolved '{dtype_input}' but it is not a torch.dtype. Defaulting to float32."
                    )
                    return torch.float32
            except AttributeError:
                logger.warning(
                    f"Unsupported CLT dtype string: '{dtype_input}'. Defaulting to float32."
                )
                return torch.float32
        return torch.float32

    def _init_parameters(self):
        """Initialize encoder and decoder parameters according to spec."""
        # Initialize encoders
        encoder_bound = 1.0 / math.sqrt(self.config.num_features)
        for encoder in self.encoders:
            nn.init.uniform_(encoder.weight, -encoder_bound, encoder_bound)

        # Initialize decoders
        decoder_bound = 1.0 / math.sqrt(self.config.num_layers * self.config.d_model)
        for decoder in self.decoders.values():
            nn.init.uniform_(decoder.weight, -decoder_bound, decoder_bound)

        logger.info(
            f"Initialized encoder weights U(-{encoder_bound:.4f}, {encoder_bound:.4f})"
        )
        logger.info(
            f"Initialized decoder weights U(-{decoder_bound:.4f}, {decoder_bound:.4f})"
        )

    def jumprelu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation function."""
        threshold = torch.exp(self.log_threshold)
        return JumpReLU.apply(x, threshold, self.bandwidth)

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get pre-activation values for features at the specified layer."""
        try:
            # Determine expected output shape before potential reshape
            if x.dim() == 2:
                expected_out_shape = (x.shape[0], self.config.num_features)
                input_for_linear = x
            elif x.dim() == 3:
                batch, seq_len, d_model = x.shape
                expected_out_shape = (batch * seq_len, self.config.num_features)
                input_for_linear = x.reshape(-1, d_model)
            else:
                logger.warning(
                    f"Cannot handle input shape {x.shape} for preactivations layer {layer_idx}"
                )
                return torch.zeros(
                    (0, self.config.num_features), device=self.device, dtype=self.dtype
                )

            if input_for_linear.shape[1] != self.config.d_model:
                logger.warning(
                    f"Input d_model {input_for_linear.shape[1]} != config {self.config.d_model} layer {layer_idx}"
                )
                return torch.zeros(
                    expected_out_shape, device=self.device, dtype=self.dtype
                )

            result = self.encoders[layer_idx](input_for_linear)
            return result

        except IndexError:
            logger.error(f"Invalid layer index {layer_idx} requested for encoder.")
            # Calculate expected shape based on input dimension before error
            fallback_shape = (
                (x.shape[0], self.config.num_features)
                if x.dim() == 2
                else (
                    x.reshape(-1, self.config.d_model).shape[0],
                    self.config.num_features,
                )
            )
            return torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)
        except Exception as e:
            logger.error(
                f"Error during get_preactivations layer {layer_idx}: {e}", exc_info=True
            )
            # Calculate expected shape based on input dimension before error
            fallback_shape = (
                (x.shape[0], self.config.num_features)
                if x.dim() == 2
                else (
                    x.reshape(-1, self.config.d_model).shape[0],
                    self.config.num_features,
                )
            )
            return torch.zeros(fallback_shape, device=self.device, dtype=self.dtype)

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.

        Ensures input tensor `x` is cast to the model's internal dtype.

        Args:
            x: Input activations [batch_size, seq_len, d_model] or [batch_tokens, d_model]
            layer_idx: Index of the layer

        Returns:
            Encoded activations after nonlinearity
        """
        # Ensure input is on the correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Determine expected fallback shape based on input dimensions
        if x.dim() == 2:
            expected_feature_shape = (x.shape[0], self.config.num_features)
        elif x.dim() == 3:
            expected_feature_shape = (
                x.reshape(-1, self.config.d_model).shape[0],
                self.config.num_features,
            )
        else:
            # If input shape is already bad, return empty tensor early
            logger.warning(
                f"Cannot handle input shape {x.shape} for encode layer {layer_idx}"
            )
            return torch.zeros(
                (0, self.config.num_features), device=self.device, dtype=self.dtype
            )

        fallback_tensor = torch.zeros(
            expected_feature_shape, device=self.device, dtype=self.dtype
        )

        try:
            preact = self.get_preactivations(x, layer_idx)

            # Check if preact is empty or has mismatched shape (indicating upstream error)
            # No need to check dtype here as input `x` was already cast
            if preact.numel() == 0 or preact.shape[-1] != self.config.num_features:
                logger.warning(
                    f"Received invalid preactivations for encode layer {layer_idx}. Returning fallback."
                )
                return fallback_tensor

            # Apply activation function
            if self.config.activation_fn == "jumprelu":
                return self.jumprelu(preact)
            else:  # "relu"
                return F.relu(preact)

        except Exception as e:
            logger.error(f"Error during encode layer {layer_idx}: {e}", exc_info=True)
            return fallback_tensor

    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Decode the feature activations to reconstruct outputs at the specified layer.

        Ensures feature activation tensors `a` are cast to the model's internal dtype.

        Args:
            a: Dictionary mapping layer indices to feature activations
            layer_idx: Index of the layer to reconstruct outputs for

        Returns:
            Reconstructed outputs
        """
        # Check keys available and get example tensor for shape/device info
        available_keys = sorted(a.keys())
        if not available_keys:
            logger.warning("No activation keys available in decode method")
            # Need a way to determine expected output shape if 'a' is empty
            # This is tricky. Let's assume we cannot proceed if 'a' is empty.
            # Returning an empty tensor, but this might cause downstream issues.
            return torch.zeros(
                (0, self.config.d_model), device=self.device, dtype=self.dtype
            )

        # Use first available tensor to determine batch dimension and device
        first_key = available_keys[0]
        example_tensor = a[first_key]
        batch_dim_size = example_tensor.shape[0]

        # Create reconstruction tensor with appropriate dimensions, device and dtype
        reconstruction = torch.zeros(
            (batch_dim_size, self.config.d_model), device=self.device, dtype=self.dtype
        )

        # Sum contributions from features at all contributing layers
        for src_layer in range(layer_idx + 1):
            if src_layer in a:
                # Ensure activation tensor is on the correct device and dtype
                activation_tensor = a[src_layer].to(
                    device=self.device, dtype=self.dtype
                )

                decoder = self.decoders[f"{src_layer}->{layer_idx}"]
                try:
                    decoded = decoder(activation_tensor)
                    reconstruction += decoded
                except Exception as e:
                    logger.error(
                        f"Error during decode from src {src_layer} to tgt {layer_idx}: {e}",
                        exc_info=True,
                    )
                    # Optionally continue to next layer or return partial/zero reconstruction

        return reconstruction

    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Process inputs through the transcoder model.

        Casting of inputs/activations to model dtype happens within encode/decode.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to reconstructed outputs
        """
        # Encode inputs at each layer (encode handles casting)
        activations = {}
        for layer_idx, x in inputs.items():
            activations[layer_idx] = self.encode(x, layer_idx)

        # Decode to reconstruct outputs at each layer (decode handles casting)
        reconstructions = {}
        for layer_idx in range(self.config.num_layers):
            # Only decode if we had inputs for this layer (or earlier layers)
            # Check if any relevant activations exist before decoding
            relevant_activations = {
                k: v for k, v in activations.items() if k <= layer_idx and v.numel() > 0
            }
            if layer_idx in inputs and relevant_activations:
                try:
                    reconstructions[layer_idx] = self.decode(
                        relevant_activations, layer_idx
                    )
                except Exception as e:
                    logger.error(
                        f"Error during forward pass decode for layer {layer_idx}: {e}",
                        exc_info=True,
                    )
                    # Store an empty tensor as placeholder if decode fails
                    # Get expected batch size from input if possible
                    batch_size = (
                        inputs[layer_idx].shape[0] if inputs[layer_idx].dim() > 1 else 0
                    )
                    reconstructions[layer_idx] = torch.zeros(
                        (batch_size, self.config.d_model),
                        device=self.device,
                        dtype=self.dtype,
                    )

        return reconstructions

    def get_feature_activations(
        self, inputs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Get feature activations for all layers.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to feature activations
        """
        activations = {}
        for layer_idx, x in inputs.items():
            try:
                act = self.encode(x, layer_idx)
                activations[layer_idx] = act
            except Exception:
                pass
        return activations

    def get_decoder_norms(self) -> torch.Tensor:
        """Get L2 norms of all decoder matrices for each feature.

        Returns:
            Tensor of shape [num_layers, num_features] containing decoder norms
        """
        decoder_norms = torch.zeros(
            self.config.num_layers,
            self.config.num_features,
            device=next(self.parameters()).device,
        )

        for src_layer in range(self.config.num_layers):
            # Compute norm of concatenated decoder matrices for each feature
            feature_norms = torch.zeros(
                self.config.num_features, device=decoder_norms.device
            )

            for tgt_layer in range(src_layer, self.config.num_layers):
                decoder_key = f"{src_layer}->{tgt_layer}"
                decoder = self.decoders[decoder_key]
                # Compute contribution of each feature (columns are features)
                norms = torch.norm(decoder.weight, dim=0) ** 2
                feature_norms += norms

            decoder_norms[src_layer] = torch.sqrt(feature_norms)

        return decoder_norms
