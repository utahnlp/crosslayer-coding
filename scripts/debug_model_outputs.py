#!/usr/bin/env python3
"""
Debug script to compare model outputs and understand why reconstruction is so poor.
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
from safetensors.torch import load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_model_behavior():
    """Analyze what the model is actually doing."""
    
    checkpoint_path = "clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors"
    config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"
    device = torch.device("cuda:0")
    
    logger.info("=== ANALYZING MODEL BEHAVIOR ===")
    
    # Load config and model
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    
    # Check some weight statistics before loading
    logger.info("\n1. Checking loaded checkpoint weights:")
    for key in list(state_dict.keys())[:5]:
        tensor = state_dict[key]
        logger.info(f"  {key}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
    
    # Load weights
    state_dict = {k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) 
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Check loaded model weights
    logger.info("\n2. Checking model weights after loading:")
    for name, param in list(model.named_parameters())[:5]:
        if param is not None:
            logger.info(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
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
    
    inputs, targets = next(activation_store)
    
    # Run model
    logger.info("\n3. Running model forward pass:")
    with torch.no_grad():
        # Convert to float32 for model
        inputs_f32 = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
        targets_f32 = {k: v.to(dtype=torch.float32) for k, v in targets.items()}
        
        # Get reconstructions
        reconstructions = model(inputs_f32)
        
        # Get feature activations
        feature_acts = model.get_feature_activations(inputs_f32)
    
    # Analyze layer 0 in detail
    layer_idx = 0
    logger.info(f"\n4. Detailed analysis of layer {layer_idx}:")
    
    inp = inputs_f32[layer_idx]
    tgt = targets_f32[layer_idx]
    recon = reconstructions[layer_idx]
    feat = feature_acts[layer_idx]
    
    logger.info(f"  Input: shape={inp.shape}, mean={inp.mean():.4f}, std={inp.std():.4f}")
    logger.info(f"  Target: shape={tgt.shape}, mean={tgt.mean():.4f}, std={tgt.std():.4f}")
    logger.info(f"  Features: shape={feat.shape}, nonzero={feat.nonzero().shape[0]}, mean_nonzero={feat[feat!=0].mean() if feat.any() else 0:.4f}")
    logger.info(f"  Reconstruction: shape={recon.shape}, mean={recon.mean():.4f}, std={recon.std():.4f}")
    
    # Check reconstruction error
    mse = torch.nn.functional.mse_loss(recon, tgt).item()
    logger.info(f"  MSE: {mse:.6f}")
    
    # Check correlation
    tgt_flat = tgt.flatten()
    recon_flat = recon.flatten()
    if len(tgt_flat) > 1:
        correlation = np.corrcoef(tgt_flat.cpu().numpy(), recon_flat.cpu().numpy())[0, 1]
        logger.info(f"  Correlation: {correlation:.4f}")
    
    # Check if decoder is producing reasonable outputs
    logger.info("\n5. Checking decoder behavior:")
    
    # Get decoder for layer 0->0
    decoder = model.decoder_module.decoders[f"{layer_idx}->{layer_idx}"]
    decoder_weight = decoder.weight
    logger.info(f"  Decoder {layer_idx}->{layer_idx} weight: shape={decoder_weight.shape}, "
                f"mean={decoder_weight.mean():.6f}, std={decoder_weight.std():.6f}")
    
    # Manually compute reconstruction for a few features
    active_indices = feat[0].nonzero().squeeze()
    if len(active_indices) > 0:
        logger.info(f"  First token has {len(active_indices)} active features")
        if len(active_indices) <= 10:
            logger.info(f"  Active feature indices: {active_indices.tolist()}")
        
        # Manual reconstruction
        manual_recon = torch.zeros_like(tgt[0])
        for idx in active_indices[:10]:  # Just check first 10
            feature_value = feat[0, idx].item()
            decoder_column = decoder_weight[:, idx]
            contribution = feature_value * decoder_column
            manual_recon += contribution
            if idx < 3:  # Log first 3
                logger.info(f"    Feature {idx}: value={feature_value:.4f}, "
                           f"decoder_norm={decoder_column.norm():.4f}, "
                           f"contribution_norm={contribution.norm():.4f}")
    
    # Check if the issue is with the scale
    logger.info("\n6. Checking scale mismatch:")
    logger.info(f"  Target L2 norm: {tgt.norm():.4f}")
    logger.info(f"  Reconstruction L2 norm: {recon.norm():.4f}")
    logger.info(f"  Ratio: {(recon.norm() / tgt.norm()):.4f}")
    
    # Check explained variance manually
    target_var = tgt.var()
    error_var = (tgt - recon).var()
    ev = 1 - (error_var / target_var) if target_var > 0 else 0
    logger.info(f"  Manual EV calculation: {ev:.4f}")
    
    # Check if features are too sparse
    logger.info("\n7. Sparsity analysis:")
    for layer_idx in range(min(3, len(feature_acts))):
        feat = feature_acts[layer_idx]
        active_per_token = (feat != 0).sum(dim=1).float()
        logger.info(f"  Layer {layer_idx}: mean active={active_per_token.mean():.1f}, "
                   f"min={active_per_token.min():.0f}, max={active_per_token.max():.0f}")


def main():
    analyze_model_behavior()


if __name__ == "__main__":
    main()