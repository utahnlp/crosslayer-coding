import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from clt.config.clt_config import TrainingConfig
from clt.models.clt import CrossLayerTranscoder


class LossManager:
    """Manages the computation of loss functions for CLT training."""

    def __init__(self, config: TrainingConfig):
        """Initialize the loss manager.

        Args:
            config: Training configuration
        """
        self.config = config
        self.mse_loss = nn.MSELoss()

    def compute_reconstruction_loss(
        self, predicted: Dict[int, torch.Tensor], target: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reconstruction loss (MSE) between predicted and target outputs.

        Args:
            predicted: Dictionary mapping layer indices to predicted outputs
            target: Dictionary mapping layer indices to target outputs

        Returns:
            MSE loss
        """
        total_loss = torch.tensor(
            0.0,
            device=(
                next(iter(predicted.values())).device
                if predicted
                else torch.device("cpu")
            ),
        )
        num_layers = 0
        for layer_idx in predicted:
            if layer_idx in target:
                layer_loss = self.mse_loss(predicted[layer_idx], target[layer_idx])
                total_loss += layer_loss
                num_layers += 1

        # Average over layers
        return total_loss / num_layers if num_layers > 0 else total_loss

    def compute_sparsity_penalty(
        self,
        model: CrossLayerTranscoder,
        activations: Dict[int, torch.Tensor],
        current_step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Compute the sparsity penalty for the feature activations.

        Args:
            model: CLT model
            activations: Dictionary mapping layer indices to feature activations
            current_step: Current training step
            total_steps: Total number of training steps

        Returns:
            Sparsity penalty loss
        """
        if not activations:
            return torch.tensor(0.0)

        # Create a linear scaling for lambda from 0 to config value
        # This allows the model to learn useful features before enforcing sparsity
        progress = min(1.0, current_step / (total_steps))
        lambda_factor = self.config.sparsity_lambda * progress

        # Get decoder norms for feature weighting
        decoder_norms = model.get_decoder_norms()

        device = next(iter(activations.values())).device
        total_penalty = torch.tensor(0.0, device=device)
        total_elements = 0

        for layer_idx, layer_activations in activations.items():
            try:
                # Get effective weight for each feature based on decoder norms
                feature_weights = decoder_norms[layer_idx].to(device)

                # Ensure layer_activations is the correct shape before proceeding
                if layer_activations.numel() == 0:
                    continue  # Skip empty activations

                # Handle different shapes of activations (2D or 3D)
                if len(layer_activations.shape) == 3:
                    # Shape is [batch_size, seq_len, num_features]
                    batch_size, seq_len, num_features = layer_activations.shape
                    activations_flat = layer_activations.reshape(-1, num_features)
                elif len(layer_activations.shape) == 2:
                    # Shape is already [batch_size * seq_len, num_features] or [batch_size, num_features]
                    activations_flat = layer_activations
                else:
                    # Handle 1D tensor if needed
                    if layer_activations.dim() == 1:
                        # Handle 1D tensor by reshaping it
                        activations_flat = layer_activations.unsqueeze(0)
                    else:
                        continue  # Skip this activation

                # Ensure the shapes are compatible for multiplication
                if activations_flat.shape[1] != feature_weights.shape[0]:
                    continue  # Skip this activation

                # Apply feature weights to activations (broadcasting)
                weighted_acts = activations_flat * feature_weights.unsqueeze(0)

                # Compute tanh penalty
                penalty = torch.tanh(self.config.sparsity_c * weighted_acts)
                total_penalty += penalty.sum()
                total_elements += layer_activations.numel()
            except Exception:
                # Skip this layer if there are issues
                continue

        # Normalize by number of elements and apply lambda
        if total_elements > 0:
            return lambda_factor * total_penalty / total_elements
        else:
            return torch.tensor(0.0, device=device)

    def compute_preactivation_loss(
        self, model: CrossLayerTranscoder, inputs: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Compute pre-activation loss to prevent dead features.

        Args:
            model: CLT model
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Pre-activation loss
        """
        if not inputs:
            return torch.tensor(0.0)

        device = next(iter(inputs.values())).device
        total_penalty = torch.tensor(0.0, device=device)
        num_elements = 0

        for layer_idx, x in inputs.items():
            # Ensure proper shape for encoder input
            if x.numel() == 0:
                continue  # Skip empty tensors

            # Check if x needs reshaping for encoder
            if len(x.shape) == 1:
                # Reshape 1D tensor to 2D
                x = x.unsqueeze(0)

            try:
                # Get pre-activations
                preacts = model.get_preactivations(x, layer_idx).to(device)

                # Apply ReLU to negative pre-activations
                penalty = F.relu(-preacts)
                total_penalty += penalty.sum()
                num_elements += preacts.numel()
            except Exception:
                continue

        # Apply coefficient and normalize
        if num_elements > 0:
            return self.config.preactivation_coef * total_penalty / num_elements
        else:
            return torch.tensor(0.0, device=device)

    def compute_total_loss(
        self,
        model: CrossLayerTranscoder,
        inputs: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        current_step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for CLT training.

        Args:
            model: CLT model
            inputs: Dictionary mapping layer indices to input activations
            targets: Dictionary mapping layer indices to target outputs
            current_step: Current training step
            total_steps: Total number of training steps

        Returns:
            Tuple of (total loss, dictionary of individual loss components)
        """
        # Get predictions
        predictions = model(inputs)

        # Get feature activations
        activations = model.get_feature_activations(inputs)

        # Compute loss components
        reconstruction_loss = self.compute_reconstruction_loss(predictions, targets)
        sparsity_loss = self.compute_sparsity_penalty(
            model, activations, current_step, total_steps
        )
        preactivation_loss = self.compute_preactivation_loss(model, inputs)

        # Compute total loss
        total_loss = reconstruction_loss + sparsity_loss + preactivation_loss

        # Return loss components
        return total_loss, {
            "total": total_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_loss.item(),
            "preactivation": preactivation_loss.item(),
        }
