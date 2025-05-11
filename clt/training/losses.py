import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from clt.config import TrainingConfig
from clt.models.clt import CrossLayerTranscoder


class LossManager:
    """Manages the computation of different loss components for the CLT."""

    def __init__(
        self,
        config: TrainingConfig,
        mean_tg: Optional[Dict[int, torch.Tensor]] = None,
        std_tg: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Initialize the loss manager.

        Args:
            config: Training configuration
            mean_tg: Optional dictionary of per-layer target means for de-normalising outputs
            std_tg: Optional dictionary of per-layer target stds for de-normalising outputs
        """
        self.config = config
        self.reconstruction_loss_fn = nn.MSELoss()
        self.current_sparsity_lambda = 0.0  # Initialize lambda
        # Store normalisation stats if provided
        self.mean_tg = mean_tg or {}
        self.std_tg = std_tg or {}
        self.aux_loss_factor = config.aux_loss_factor  # New: coefficient for auxiliary loss
        self.apply_sparsity_penalty_to_batchtopk = config.apply_sparsity_penalty_to_batchtopk

        # Validate sparsity schedule params
        assert self.config.sparsity_lambda_schedule in ["linear", "delayed_linear"], "Invalid sparsity_lambda_schedule"
        if self.config.sparsity_lambda_schedule == "delayed_linear":
            assert (
                0.0 <= self.config.sparsity_lambda_delay_frac < 1.0
            ), "sparsity_lambda_delay_frac must be between 0.0 (inclusive) and 1.0 (exclusive)"

    def compute_reconstruction_loss(
        self, predicted: Dict[int, torch.Tensor], target: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reconstruction loss (MSE) between predicted and target outputs.

        If normalisation statistics were provided, the method first *de-normalises* both
        the predictions and the targets using the stored mean/std tensors so that the
        loss is measured in the *original* activation space.  This avoids the loss
        scale changing depending on whether the inputs were normalised by the
        ActivationStore.

        Args:
            predicted: Dictionary mapping layer indices to predicted outputs
            target: Dictionary mapping layer indices to target outputs

        Returns:
            MSE loss (summed over layers)
        """
        total_loss = torch.tensor(
            0.0,
            device=(next(iter(predicted.values())).device if predicted else torch.device("cpu")),
        )

        for layer_idx in predicted:
            if layer_idx not in target:
                continue

            pred_layer = predicted[layer_idx]
            tgt_layer = target[layer_idx]

            # De-normalise if stats available for this layer
            if layer_idx in self.mean_tg and layer_idx in self.std_tg:
                mean = self.mean_tg[layer_idx].to(pred_layer.device, pred_layer.dtype)
                std = self.std_tg[layer_idx].to(pred_layer.device, pred_layer.dtype)
                # mean/std were stored with an added batch dim – ensure broadcast shape
                pred_layer = pred_layer * std + mean
                tgt_layer = tgt_layer * std + mean

            layer_loss = self.reconstruction_loss_fn(pred_layer, tgt_layer)
            total_loss += layer_loss

        return total_loss

    def compute_sparsity_penalty(
        self,
        model: CrossLayerTranscoder,
        activations: Dict[int, torch.Tensor],
        current_step: int,
        total_steps: int,
    ) -> Tuple[torch.Tensor, float]:
        """Compute the sparsity penalty for the feature activations.

        Args:
            model: CLT model
            activations: Dictionary mapping layer indices to feature activations
            current_step: Current training step
            total_steps: Total number of training steps

        Returns:
            Tuple of (sparsity penalty loss, current lambda)
        """
        # --- Sparsity penalty calculation restored --- #

        if not activations:
            return torch.tensor(0.0, device=next(iter(model.parameters())).device), 0.0

        # --- Calculate current lambda based on schedule --- #
        target_lambda = self.config.sparsity_lambda
        delay_frac = self.config.sparsity_lambda_delay_frac
        schedule = self.config.sparsity_lambda_schedule

        progress = min(1.0, current_step / total_steps)
        lambda_factor = 0.0

        if schedule == "linear":
            lambda_factor = target_lambda * progress
        elif schedule == "delayed_linear":
            delay_start_step = delay_frac * total_steps
            if current_step >= delay_start_step:
                # Avoid division by zero if delay_frac is very close to 1
                if total_steps > delay_start_step:
                    delayed_progress = (current_step - delay_start_step) / (total_steps - delay_start_step)
                    lambda_factor = target_lambda * min(1.0, delayed_progress)
                else:
                    lambda_factor = target_lambda  # Start at max lambda immediately if delay covers whole training
            else:
                lambda_factor = 0.0  # Still in delay phase

        # If lambda factor is effectively zero, no penalty
        if lambda_factor < 1e-9:  # Use a small tolerance
            return torch.tensor(0.0, device=next(iter(model.parameters())).device), 0.0

        # Get decoder norms for feature weighting
        decoder_norms = model.get_decoder_norms()

        device = next(iter(activations.values())).device
        total_penalty = torch.tensor(0.0, device=device)

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

                # Summed over features as in the paper, then added to the layer total so far
                total_penalty += penalty.sum()
            except Exception:
                # raise an error
                raise ValueError(
                    f"Error computing sparsity penalty for layer {layer_idx}. "
                    f"Activations shape: {layer_activations.shape}. "
                    f"Feature weights shape: {feature_weights.shape}."
                )

        # Main sparsity penalty (tanh on activations)
        main_penalty = lambda_factor * total_penalty

        # Skip sparsity penalty if using BatchTopK and config says so
        if model.config.activation_fn == "batchtopk" and not self.apply_sparsity_penalty_to_batchtopk:
            return torch.tensor(0.0, device=next(iter(model.parameters())).device), 0.0

        return main_penalty, lambda_factor

    def compute_preactivation_loss(self, model: CrossLayerTranscoder, inputs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute pre-activation loss to prevent dead features.

        Args:
            model: CLT model
            inputs: Dictionary mapping layer indices to input activations

        Returns:
            Pre-activation loss
        """
        # Early exit if coefficient is None or effectively zero
        if self.config.preactivation_coef is None or abs(self.config.preactivation_coef) < 1e-9:
            # Determine device for the zero tensor
            # If inputs is not empty, use the device of the first tensor in inputs
            # Otherwise, attempt to get device from model or default to CPU
            if inputs:
                device = next(iter(inputs.values())).device
            else:
                # Fallback: try model's device if model is accessible here, or default to CPU
                # This part might need adjustment if model isn't directly accessible
                # or if a more robust way to get a default device is needed.
                # For now, assuming a CPU fallback is acceptable if inputs is empty.
                device = torch.device("cpu")
                # A more robust way if model was passed: device = next(model.parameters()).device
            return torch.tensor(0.0, device=device)

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

    def compute_auxiliary_loss(
        self,
        model: CrossLayerTranscoder,
        inputs: Dict[int, torch.Tensor],
        residuals: Dict[int, torch.Tensor],
        dead_mask: Optional[torch.Tensor] = None,
        k_aux_default: int = 512,
    ) -> torch.Tensor:
        """Compute the auxiliary reconstruction loss for BatchTopK dead features.

        This follows the AuxK strategy from the TopK/BatchTopK papers: use the top-k dead
        latents to reconstruct the residual error.
        """
        if self.aux_loss_factor is None or self.aux_loss_factor == 0:
            return torch.tensor(0.0, device=next(iter(model.parameters())).device)
        if dead_mask is None:
            return torch.tensor(0.0, device=next(iter(model.parameters())).device)

        # Prepare hidden pre-activations (encoder linear output before nonlinearity)
        hidden_pre: Dict[int, torch.Tensor] = {}
        for layer_idx, x in inputs.items():
            try:
                pre = model.get_preactivations(x, layer_idx)
                hidden_pre[layer_idx] = pre  # shape [batch_tokens, num_features]
            except Exception:
                continue

        aux_loss_total = torch.tensor(0.0, device=next(iter(model.parameters())).device)

        for layer_idx, pre in hidden_pre.items():
            if pre.numel() == 0:
                continue
            if layer_idx >= dead_mask.shape[0]:
                continue  # Safety
            dead_layer_mask = dead_mask[layer_idx]  # [num_features] bool
            num_dead = int(dead_layer_mask.sum().item())
            if num_dead == 0:
                continue
            # Exclude living latents by setting them to -inf so they won't be selected in topk
            pre_dead = pre.clone()
            live_mask = ~dead_layer_mask
            pre_dead[:, live_mask] = -float("inf")

            k_aux = min(k_aux_default, num_dead)
            k_aux = max(1, k_aux)
            # topk over feature dim
            topk_vals, topk_idx = pre_dead.topk(k_aux, dim=-1)
            aux_acts = torch.zeros_like(pre)
            aux_acts.scatter_(-1, topk_idx, topk_vals)

            # Reconstruct residual using only these aux acts
            aux_input_dict = {layer_idx: aux_acts}
            try:
                recon_aux = model.decode(aux_input_dict, layer_idx)
            except Exception:
                continue

            if layer_idx not in residuals:
                continue
            residual_layer = residuals[layer_idx].to(recon_aux.device, recon_aux.dtype)
            aux_loss_layer = F.mse_loss(recon_aux, residual_layer)
            scale = min(num_dead / k_aux, 1.0)
            aux_loss_total += scale * aux_loss_layer

        # Scale by coefficient alpha
        aux_loss_total = self.aux_loss_factor * aux_loss_total
        return aux_loss_total

    def compute_total_loss(
        self,
        model: CrossLayerTranscoder,
        inputs: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        current_step: int,
        total_steps: int,
        precomputed_activations: Optional[Dict[int, torch.Tensor]] = None,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss for CLT training.

        Args:
            model: CLT model
            inputs: Dictionary mapping layer indices to input activations
            targets: Dictionary mapping layer indices to target outputs
            current_step: Current training step
            total_steps: Total number of training steps
            precomputed_activations: Optional dictionary of feature activations that have already been
                computed outside this function. Supplying this avoids redundant encoder forward passes.
            dead_neuron_mask: Optional tensor indicating dead neurons

        Returns:
            Tuple of (total loss, dictionary of individual loss components)
        """
        # Get predictions
        predictions = model(inputs)

        # Get feature activations
        if precomputed_activations is None:
            activations = model.get_feature_activations(inputs)
        else:
            activations = precomputed_activations

        # Compute loss components
        reconstruction_loss = self.compute_reconstruction_loss(predictions, targets)
        sparsity_penalty, current_lambda = self.compute_sparsity_penalty(model, activations, current_step, total_steps)
        self.current_sparsity_lambda = current_lambda  # Store the lambda
        preactivation_loss = self.compute_preactivation_loss(model, inputs)

        # Compute residuals for auxiliary loss if needed
        residuals = {}
        for layer_idx in predictions:
            if layer_idx in targets:
                residuals[layer_idx] = targets[layer_idx] - predictions[layer_idx]

        # Compute auxiliary loss (only if configured and using BatchTopK)
        aux_loss = torch.tensor(0.0, device=reconstruction_loss.device)
        if model.config.activation_fn == "batchtopk" or model.config.activation_fn == "topk":
            aux_loss = self.compute_auxiliary_loss(model, inputs, residuals, dead_neuron_mask)

        # Compute total loss
        total_loss = reconstruction_loss + sparsity_penalty + preactivation_loss + aux_loss

        # Return loss components
        return total_loss, {
            "total": total_loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_penalty.item(),
            "preactivation": preactivation_loss.item(),
            "auxiliary": aux_loss.item(),
        }

    def get_current_sparsity_lambda(self) -> float:
        """Returns the most recently calculated sparsity lambda."""
        return self.current_sparsity_lambda
