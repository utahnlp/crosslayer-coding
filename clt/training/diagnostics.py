import torch
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from clt.models.clt import CrossLayerTranscoder
    from clt.config import TrainingConfig


@torch.no_grad()
def compute_sparsity_diagnostics(
    model: "CrossLayerTranscoder",
    training_config: "TrainingConfig",
    # inputs: Dict[int, torch.Tensor], # Inputs are implicitly used by model.get_decoder_norms if not cached
    feature_activations: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """Computes detailed sparsity diagnostics (z-scores, tanh saturation, etc.).

    Args:
        model: The CrossLayerTranscoder model instance.
        training_config: The training configuration for parameters like sparsity_c.
        feature_activations: Dictionary of feature activations (pre-computed).

    Returns:
        Dictionary containing sparsity diagnostic metrics.
    """
    diag_metrics: Dict[str, Any] = {}
    layerwise_z_median: Dict[str, float] = {}
    layerwise_z_p90: Dict[str, float] = {}
    layerwise_mean_tanh: Dict[str, float] = {}
    layerwise_sat_frac: Dict[str, float] = {}
    layerwise_mean_abs_act: Dict[str, float] = {}
    layerwise_mean_dec_norm: Dict[str, float] = {}

    all_layer_medians = []
    all_layer_p90s = []
    all_layer_mean_tanhs = []
    all_layer_sat_fracs = []
    all_layer_abs_act = []
    all_layer_dec_norm = []

    sparsity_c = training_config.sparsity_c

    # Norms should be cached from the loss calculation earlier in the step,
    # or recomputed if necessary by get_decoder_norms()
    diag_dec_norms = model.get_decoder_norms()  # [L, F]

    for l_idx, layer_acts in feature_activations.items():
        if layer_acts.numel() == 0:
            layer_key = f"layer_{l_idx}"
            layerwise_z_median[layer_key] = float("nan")
            layerwise_z_p90[layer_key] = float("nan")
            layerwise_mean_tanh[layer_key] = float("nan")
            layerwise_sat_frac[layer_key] = float("nan")
            # Initialize other layerwise metrics as well for consistency if layer is skipped
            layerwise_mean_abs_act[layer_key] = float("nan")
            layerwise_mean_dec_norm[layer_key] = float("nan")
            continue

        # Ensure norms and activations are compatible and on the same device
        norms_l = diag_dec_norms[l_idx].to(layer_acts.device, layer_acts.dtype).unsqueeze(0)  # [1, F]
        layer_acts = layer_acts.to(norms_l.device, norms_l.dtype)

        z = sparsity_c * norms_l * layer_acts  # [tokens, F]
        on_mask = layer_acts > 1e-6  # Use a small threshold > 0
        z_on = z[on_mask]

        if z_on.numel() > 0:
            med = torch.median(z_on).item()
            p90 = torch.quantile(z_on.float(), 0.9).item()  # Ensure float for quantile
            tanh_z_on = torch.tanh(z_on)
            mean_tanh = tanh_z_on.mean().item()
            sat_frac = (tanh_z_on.abs() > 0.99).float().mean().item()  # Use abs for saturation
        else:
            med, p90, mean_tanh, sat_frac = float("nan"), float("nan"), float("nan"), float("nan")

        layer_key = f"layer_{l_idx}"
        layerwise_z_median[layer_key] = med
        layerwise_z_p90[layer_key] = p90
        layerwise_mean_tanh[layer_key] = mean_tanh
        layerwise_sat_frac[layer_key] = sat_frac

        mean_abs_act_val = layer_acts.abs().mean().item() if layer_acts.numel() > 0 else float("nan")
        # Ensure l_idx is valid for diag_dec_norms before accessing
        mean_dec_norm_val = diag_dec_norms[l_idx].mean().item() if l_idx < diag_dec_norms.shape[0] else float("nan")

        layerwise_mean_abs_act[layer_key] = mean_abs_act_val  # Use layer_key for consistency
        layerwise_mean_dec_norm[layer_key] = mean_dec_norm_val  # Use layer_key for consistency

        if not torch.isnan(torch.tensor(mean_abs_act_val)):
            all_layer_abs_act.append(mean_abs_act_val)
        if not torch.isnan(torch.tensor(mean_dec_norm_val)):
            all_layer_dec_norm.append(mean_dec_norm_val)

        if not torch.isnan(torch.tensor(med)):
            all_layer_medians.append(med)
        if not torch.isnan(torch.tensor(p90)):
            all_layer_p90s.append(p90)
        if not torch.isnan(torch.tensor(mean_tanh)):
            all_layer_mean_tanhs.append(mean_tanh)
        if not torch.isnan(torch.tensor(sat_frac)):
            all_layer_sat_fracs.append(sat_frac)

    agg_z_median = torch.tensor(all_layer_medians).mean().item() if all_layer_medians else float("nan")
    agg_z_p90 = torch.tensor(all_layer_p90s).mean().item() if all_layer_p90s else float("nan")
    agg_mean_tanh = torch.tensor(all_layer_mean_tanhs).mean().item() if all_layer_mean_tanhs else float("nan")
    agg_sat_frac = torch.tensor(all_layer_sat_fracs).mean().item() if all_layer_sat_fracs else float("nan")
    agg_mean_abs_act = torch.tensor(all_layer_abs_act).mean().item() if all_layer_abs_act else float("nan")
    agg_mean_dec_norm = torch.tensor(all_layer_dec_norm).mean().item() if all_layer_dec_norm else float("nan")

    diag_metrics["layerwise/sparsity_z_median"] = layerwise_z_median
    diag_metrics["layerwise/sparsity_z_p90"] = layerwise_z_p90
    diag_metrics["layerwise/sparsity_mean_tanh"] = layerwise_mean_tanh
    diag_metrics["layerwise/sparsity_sat_frac"] = layerwise_sat_frac
    diag_metrics["layerwise/mean_abs_activation"] = layerwise_mean_abs_act
    diag_metrics["layerwise/mean_decoder_norm"] = layerwise_mean_dec_norm
    diag_metrics["sparsity/z_median_agg"] = agg_z_median
    diag_metrics["sparsity/z_p90_agg"] = agg_z_p90
    diag_metrics["sparsity/mean_tanh_agg"] = agg_mean_tanh
    diag_metrics["sparsity/sat_frac_agg"] = agg_sat_frac
    diag_metrics["sparsity/mean_abs_activation_agg"] = agg_mean_abs_act
    diag_metrics["sparsity/mean_decoder_norm_agg"] = agg_mean_dec_norm

    # No need to explicitly delete local tensors like z, z_on etc. Python GC handles it.
    return diag_metrics
