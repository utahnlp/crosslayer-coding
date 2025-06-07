#!/usr/bin/env python3
"""
Test if rescaling the model outputs fixes the evaluation metrics.
"""

import torch
import sys
import json
from pathlib import Path
import logging
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.evaluator import CLTEvaluator
from safetensors.torch import load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_optimal_scale(targets: torch.Tensor, reconstructions: torch.Tensor) -> float:
    """Compute the optimal scale factor to minimize MSE."""
    # Optimal scale is: sum(target * reconstruction) / sum(reconstruction^2)
    num = (targets * reconstructions).sum()
    denom = (reconstructions * reconstructions).sum()
    return (num / denom).item() if denom > 0 else 1.0


def main():
    checkpoint_path = "clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors"
    config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"
    device = torch.device("cuda:0")
    
    logger.info("=== TESTING RESCALING FIX ===")
    
    # Load model
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    state_dict = {k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) 
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get test data
    activation_store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=1024,
        device=device,
        dtype="float16",
        rank=0,
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=True,
    )
    
    # Get normalization stats for proper evaluation
    mean_tg = {}
    std_tg = {}
    if hasattr(activation_store, 'mean_tg') and activation_store.mean_tg:
        for layer_idx, mean_tensor in activation_store.mean_tg.items():
            mean_tg[layer_idx] = mean_tensor.to(device)
            std_tg[layer_idx] = activation_store.std_tg[layer_idx].to(device)
    
    # Initialize evaluator with normalization stats
    evaluator = CLTEvaluator(
        model=model,
        device=device,
        mean_tg=mean_tg,
        std_tg=std_tg,
    )
    
    # Test on multiple batches
    num_batches = 5
    all_scales = []
    
    logger.info("\nTesting on multiple batches...")
    
    for batch_idx in range(num_batches):
        inputs, targets = next(activation_store)
        
        with torch.no_grad():
            # Get original metrics
            metrics_original = evaluator.compute_metrics(inputs, targets)
            nmse_original = metrics_original.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
            ev_original = metrics_original.get("reconstruction/explained_variance", 0.0)
            
            # Get reconstructions
            inputs_f32 = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
            reconstructions = model(inputs_f32)
            
            # Compute optimal scale for each layer
            layer_scales = {}
            for layer_idx in reconstructions.keys():
                if layer_idx in targets:
                    target = targets[layer_idx].to(dtype=torch.float32)
                    recon = reconstructions[layer_idx]
                    scale = compute_optimal_scale(target, recon)
                    layer_scales[layer_idx] = scale
            
            # Average scale across layers
            avg_scale = np.mean(list(layer_scales.values()))
            all_scales.append(avg_scale)
            
            # Apply scale and recompute metrics
            scaled_reconstructions = {k: v * avg_scale for k, v in reconstructions.items()}
            
            # Manually compute metrics with scaled reconstructions
            total_mse = 0
            total_var = 0
            total_ev = 0
            num_layers = 0
            
            for layer_idx in targets.keys():
                if layer_idx in scaled_reconstructions:
                    target = targets[layer_idx].to(dtype=torch.float32)
                    recon = scaled_reconstructions[layer_idx]
                    
                    # Denormalize if we have stats
                    if layer_idx in mean_tg and layer_idx in std_tg:
                        mean = mean_tg[layer_idx]
                        std = std_tg[layer_idx]
                        target_denorm = target * std + mean
                        recon_denorm = recon * std + mean
                    else:
                        target_denorm = target
                        recon_denorm = recon
                    
                    mse = torch.nn.functional.mse_loss(recon_denorm, target_denorm).item()
                    var = target_denorm.var().item()
                    
                    if var > 1e-9:
                        nmse = mse / var
                        ev = 1 - ((target_denorm - recon_denorm).var() / var).item()
                    else:
                        nmse = 0.0
                        ev = 1.0
                    
                    total_mse += nmse
                    total_ev += ev
                    num_layers += 1
            
            nmse_scaled = total_mse / num_layers if num_layers > 0 else float("nan")
            ev_scaled = total_ev / num_layers if num_layers > 0 else 0.0
            
            logger.info(f"\nBatch {batch_idx}:")
            logger.info(f"  Original: NMSE={nmse_original:.4f}, EV={ev_original:.4f}")
            logger.info(f"  Scale factor: {avg_scale:.4f}")
            logger.info(f"  Scaled: NMSE={nmse_scaled:.4f}, EV={ev_scaled:.4f}")
            logger.info(f"  Layer scales: {[f'{k}:{v:.3f}' for k, v in sorted(layer_scales.items())[:3]]}")
    
    # Summary
    overall_scale = np.mean(all_scales)
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Average scale factor needed: {overall_scale:.4f}")
    logger.info(f"Scale factor std: {np.std(all_scales):.4f}")
    
    if 0.7 < overall_scale < 0.9:
        logger.info("\nThe model outputs are systematically too large by ~{:.1f}%".format((1/overall_scale - 1) * 100))
        logger.info("This suggests a scale mismatch during training, possibly due to:")
        logger.info("  1. The auxiliary loss (aux_loss_factor=0.03125)")
        logger.info("  2. Numerical precision issues with fp16 training")
        logger.info("  3. Normalization/denormalization mismatch")
    
    # Test if we can fix the model by scaling decoder weights
    logger.info(f"\n=== TESTING DECODER WEIGHT SCALING ===")
    logger.info(f"Scaling all decoder weights by {overall_scale:.4f}...")
    
    # Scale decoder weights
    for name, param in model.named_parameters():
        if "decoder" in name and "weight" in name:
            param.data *= overall_scale
    
    # Re-evaluate
    logger.info("\nRe-evaluating with scaled decoder weights...")
    metrics_fixed = evaluator.compute_metrics(inputs, targets)
    nmse_fixed = metrics_fixed.get("reconstruction/normalized_mean_reconstruction_error", float("nan"))
    ev_fixed = metrics_fixed.get("reconstruction/explained_variance", 0.0)
    
    logger.info(f"After decoder scaling: NMSE={nmse_fixed:.4f}, EV={ev_fixed:.4f}")


if __name__ == "__main__":
    main()