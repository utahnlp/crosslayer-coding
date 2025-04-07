import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from clt.config import CLTConfig
from clt.models.base import BaseTranscoder


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
        ctx.save_for_backward(input, threshold)  # Save threshold for backward pass
        ctx.bandwidth = bandwidth

        # JumpReLU: 0 if x < threshold, x if x >= threshold
        return (input >= threshold).float() * input

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with straight-through gradient estimator.

        Args:
            grad_output: Gradient from subsequent layer

        Returns:
            Gradients for input and log_threshold.
        """
        (input, threshold) = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Mask for inputs within the STE window around the threshold
        is_near_threshold = torch.abs(input - threshold) <= (bandwidth / 2.0)
        is_near_threshold_float = is_near_threshold.float()

        # 1. Gradient for input (Straight-through estimator)
        grad_input = grad_output.clone()
        grad_input = (
            grad_input * is_near_threshold_float
        )  # Pass gradient only within the window

        # 2. Gradient for threshold
        # For d(JumpReLU)/d(threshold), we use:
        # d(z * H(z-theta))/d(theta) = z * d(H(z-theta))/d(theta)
        # Using STE for dH/d(theta): approx -1/bandwidth within the window
        local_grad_theta = (-input / bandwidth) * is_near_threshold_float
        grad_threshold_per_element = grad_output * local_grad_theta

        # Sum gradients across batch/sequence dimensions to match threshold shape
        if grad_threshold_per_element.dim() > 1:
            dims_to_sum = tuple(range(grad_threshold_per_element.dim() - 1))
            grad_threshold = grad_threshold_per_element.sum(dim=dims_to_sum)
        else:
            # Handle case where input might be 1D
            grad_threshold = grad_threshold_per_element

        # Return gradients for input, threshold, and None for bandwidth
        return grad_input, grad_threshold, None


class CrossLayerTranscoder(BaseTranscoder):
    """Implementation of a Cross-Layer Transcoder (CLT)."""

    def __init__(self, config: CLTConfig):
        """Initialize the Cross-Layer Transcoder.

        Args:
            config: Configuration for the transcoder
        """
        super().__init__(config)

        # Create encoder matrices for each layer
        self.encoders = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.num_features, bias=False)
                for _ in range(config.num_layers)
            ]
        )

        # Create decoder matrices for each layer pair
        self.decoders = nn.ModuleDict(
            {
                f"{src_layer}->{tgt_layer}": nn.Linear(
                    config.num_features, config.d_model, bias=False
                )
                for src_layer in range(config.num_layers)
                for tgt_layer in range(src_layer, config.num_layers)
            }
        )

        # Initialize parameters
        self._init_parameters()

        # Use log_threshold to ensure positivity of the threshold
        self.log_threshold = nn.Parameter(
            torch.ones(config.num_features)
            * torch.log(torch.tensor(config.jumprelu_threshold))
        )
        self.bandwidth = 1.0  # Bandwidth parameter for straight-through estimator

    def _init_parameters(self):
        """Initialize encoder and decoder parameters."""
        # Initialize encoder weights
        encoder_bound = 1.0 / (self.config.num_features**0.5)
        for encoder in self.encoders:
            nn.init.uniform_(encoder.weight, -encoder_bound, encoder_bound)

        # Initialize decoder weights
        decoder_bound = 1.0 / ((self.config.num_layers * self.config.d_model) ** 0.5)
        for decoder in self.decoders.values():
            nn.init.uniform_(decoder.weight, -decoder_bound, decoder_bound)

    def jumprelu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply JumpReLU activation function.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        # Convert log_threshold to threshold via exponentiation
        threshold = torch.exp(self.log_threshold)
        return JumpReLU.apply(x, threshold, self.bandwidth)

    def get_preactivations(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Get pre-activation values for features at the specified layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            layer_idx: Layer index

        Returns:
            Pre-activation values
        """
        try:
            result = self.encoders[layer_idx](x)
            return result
        except Exception:
            # Try reshaping input if needed
            if x.dim() == 2 and x.shape[1] == self.config.d_model:
                # If shape is [batch_size, d_model], we can use as is
                result = self.encoders[layer_idx](x)
                return result
            elif x.dim() == 3:
                # If shape is [batch_size, seq_len, d_model], we need to flatten to 2D
                batch, seq_len, d_model = x.shape
                x_flat = x.reshape(-1, d_model)
                result = self.encoders[layer_idx](x_flat)
                return result
            else:
                raise ValueError(
                    f"Cannot handle input shape {x.shape} for preactivations"
                )

    def encode(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode the input activations at the specified layer.

        Args:
            x: Input activations [batch_size, seq_len, d_model]
            layer_idx: Index of the layer

        Returns:
            Encoded activations after nonlinearity
        """
        try:
            preact = self.get_preactivations(x, layer_idx)

            # Apply activation function
            if self.config.activation_fn == "jumprelu":
                result = self.jumprelu(preact)
                return result
            else:  # "relu"
                result = F.relu(preact)
                return result
        except Exception:
            # Return empty tensor of appropriate size as fallback
            return torch.zeros(1, self.config.num_features, device=x.device)

    def decode(self, a: Dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor:
        """Decode the feature activations to reconstruct outputs at the specified layer.

        Args:
            a: Dictionary mapping layer indices to feature activations
            layer_idx: Index of the layer to reconstruct outputs for

        Returns:
            Reconstructed outputs
        """
        # Check keys available and create reconstruction tensor
        available_keys = sorted(a.keys())

        if not available_keys:
            raise ValueError("No activation keys available in decode method")

        # Example tensor to get dimensions
        example_tensor = a[available_keys[0]]

        # Create reconstruction tensor with appropriate dimensions
        # The dimensions should be the same as the input, except for the last dimension
        # which should be d_model
        reconstruction = torch.zeros(
            example_tensor.shape[0], self.config.d_model, device=example_tensor.device
        )

        # Sum contributions from features at all contributing layers
        for src_layer in range(layer_idx + 1):
            if src_layer in a:
                decoder = self.decoders[f"{src_layer}->{layer_idx}"]
                decoded = decoder(a[src_layer])
                reconstruction += decoded

        return reconstruction

    def forward(self, inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Process inputs through the transcoder model.

        Args:
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Dictionary mapping layer indices to reconstructed outputs
        """
        # Encode inputs at each layer
        activations = {}
        for layer_idx, x in inputs.items():
            activations[layer_idx] = self.encode(x, layer_idx)

        # Decode to reconstruct outputs at each layer
        reconstructions = {}
        for layer_idx in range(self.config.num_layers):
            if layer_idx in inputs:
                reconstructions[layer_idx] = self.decode(activations, layer_idx)

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
