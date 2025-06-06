#!/usr/bin/env python3
"""
Debug script to understand why evaluation metrics are terrible.
Focus on normalization handling during evaluation.
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.evaluator import CLTEvaluator
from clt.training.data.local_activation_store import LocalActivationStore
from safetensors.torch import load_file as load_safetensors_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> Optional[CrossLayerTranscoder]:
    """Load a CLT model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Determine paths
    if checkpoint_path.suffix == ".safetensors":
        model_path = checkpoint_path
        config_path = checkpoint_path.parent / "cfg.json"
    else:
        model_path = checkpoint_path / "model.safetensors"
        config_path = checkpoint_path / "cfg.json"
    
    if not model_path.exists() or not config_path.exists():
        logger.error(f"Model or config not found")
        return None
    
    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    # Create model
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    
    # Load state dict
    state_dict = load_safetensors_file(str(model_path), device="cpu")
    state_dict = {k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) 
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    return model


def debug_normalization(
    model: CrossLayerTranscoder,
    activation_path: str,
    batch_size: int,
    device: torch.device,
) -> None:
    """Debug normalization issues in evaluation."""
    
    logger.info("=== DEBUGGING NORMALIZATION ===")
    
    # 1. Create activation store
    logger.info("\n1. Creating activation store...")
    activation_store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=batch_size,
        device=device,
        dtype="float16",
        rank=0,
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=True,
    )
    
    # 2. Check what normalization stats the store loaded
    logger.info("\n2. Checking activation store normalization:")
    logger.info(f"   Apply normalization: {activation_store.apply_normalization}")
    logger.info(f"   Has mean_in: {hasattr(activation_store, 'mean_in') and bool(activation_store.mean_in)}")
    logger.info(f"   Has std_in: {hasattr(activation_store, 'std_in') and bool(activation_store.std_in)}")
    logger.info(f"   Has mean_tg: {hasattr(activation_store, 'mean_tg') and bool(activation_store.mean_tg)}")
    logger.info(f"   Has std_tg: {hasattr(activation_store, 'std_tg') and bool(activation_store.std_tg)}")
    
    # 3. Get a batch and check its statistics
    logger.info("\n3. Getting a batch to check statistics...")
    inputs, targets = next(activation_store)
    
    logger.info("   Input statistics (after activation store processing):")
    for layer_idx, inp in inputs.items():
        logger.info(f"     Layer {layer_idx}: mean={inp.mean().item():.4f}, std={inp.std().item():.4f}, "
                   f"shape={inp.shape}")
    
    logger.info("   Target statistics (after activation store processing):")
    for layer_idx, tgt in targets.items():
        logger.info(f"     Layer {layer_idx}: mean={tgt.mean().item():.4f}, std={tgt.std().item():.4f}, "
                   f"shape={tgt.shape}")
    
    # 4. Run model forward pass
    logger.info("\n4. Running model forward pass...")
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            reconstructions = model(inputs)
    
    logger.info("   Reconstruction statistics:")
    for layer_idx, recon in reconstructions.items():
        logger.info(f"     Layer {layer_idx}: mean={recon.mean().item():.4f}, std={recon.std().item():.4f}")
    
    # 5. Create evaluator WITHOUT normalization stats
    logger.info("\n5. Testing evaluation WITHOUT normalization stats...")
    evaluator_no_norm = CLTEvaluator(model=model, device=device)
    
    with torch.no_grad():
        metrics_no_norm = evaluator_no_norm.compute_metrics(inputs, targets)
    
    logger.info(f"   NMSE (no norm): {metrics_no_norm.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}")
    logger.info(f"   EV (no norm): {metrics_no_norm.get('reconstruction/explained_variance', -1):.4f}")
    
    # 6. Create evaluator WITH normalization stats from activation store
    logger.info("\n6. Testing evaluation WITH normalization stats...")
    
    # Extract normalization stats from activation store
    mean_tg = {}
    std_tg = {}
    
    if hasattr(activation_store, 'mean_tg') and activation_store.mean_tg:
        for layer_idx, mean_tensor in activation_store.mean_tg.items():
            mean_tg[layer_idx] = mean_tensor.to(device)
            logger.info(f"   Found mean_tg for layer {layer_idx}: shape={mean_tensor.shape}")
    
    if hasattr(activation_store, 'std_tg') and activation_store.std_tg:
        for layer_idx, std_tensor in activation_store.std_tg.items():
            std_tg[layer_idx] = std_tensor.to(device)
            logger.info(f"   Found std_tg for layer {layer_idx}: shape={std_tensor.shape}")
    
    evaluator_with_norm = CLTEvaluator(
        model=model,
        device=device,
        mean_tg=mean_tg,
        std_tg=std_tg,
    )
    
    with torch.no_grad():
        metrics_with_norm = evaluator_with_norm.compute_metrics(inputs, targets)
    
    logger.info(f"   NMSE (with norm): {metrics_with_norm.get('reconstruction/normalized_mean_reconstruction_error', -1):.4f}")
    logger.info(f"   EV (with norm): {metrics_with_norm.get('reconstruction/explained_variance', -1):.4f}")
    
    # 7. Manually compute metrics to verify
    logger.info("\n7. Manual metric computation for verification...")
    
    # Pick first layer for detailed analysis
    layer_idx = 0
    target = targets[layer_idx]
    recon = reconstructions[layer_idx]
    
    # Without denormalization
    mse_normalized = torch.nn.functional.mse_loss(recon, target).item()
    var_target_normalized = target.var().item()
    nmse_normalized = mse_normalized / var_target_normalized if var_target_normalized > 0 else float('inf')
    
    logger.info(f"   Layer {layer_idx} (normalized space):")
    logger.info(f"     MSE: {mse_normalized:.6f}")
    logger.info(f"     Target variance: {var_target_normalized:.6f}")
    logger.info(f"     NMSE: {nmse_normalized:.6f}")
    
    # With denormalization (if stats available)
    if layer_idx in mean_tg and layer_idx in std_tg:
        mean = mean_tg[layer_idx]
        std = std_tg[layer_idx]
        
        target_denorm = target * std + mean
        recon_denorm = recon * std + mean
        
        mse_denorm = torch.nn.functional.mse_loss(recon_denorm, target_denorm).item()
        var_target_denorm = target_denorm.var().item()
        nmse_denorm = mse_denorm / var_target_denorm if var_target_denorm > 0 else float('inf')
        
        logger.info(f"   Layer {layer_idx} (denormalized space):")
        logger.info(f"     MSE: {mse_denorm:.6f}")
        logger.info(f"     Target variance: {var_target_denorm:.6f}")
        logger.info(f"     NMSE: {nmse_denorm:.6f}")
        logger.info(f"     Target denorm stats: mean={target_denorm.mean().item():.4f}, std={target_denorm.std().item():.4f}")
        logger.info(f"     Recon denorm stats: mean={recon_denorm.mean().item():.4f}, std={recon_denorm.std().item():.4f}")
    
    # 8. Check if the model is actually doing anything useful
    logger.info("\n8. Checking model behavior:")
    
    # Check sparsity
    feature_acts = model.get_feature_activations(inputs)
    for layer_idx, acts in feature_acts.items():
        sparsity = (acts == 0).float().mean().item()
        logger.info(f"   Layer {layer_idx} sparsity: {sparsity:.4f}")
        if layer_idx == 0:  # Detailed check for first layer
            num_active = (acts != 0).sum(dim=-1).float().mean().item()
            logger.info(f"   Layer {layer_idx} avg active features: {num_active:.1f}")
    
    logger.info("\n=== END DEBUGGING ===")


def main():
    parser = argparse.ArgumentParser(description="Debug evaluation normalization issues")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--activation-path", type=str, required=True, help="Path to activation dataset")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    if model is None:
        logger.error("Failed to load model")
        return 1
    
    logger.info(f"Model loaded: {model.config.num_features} features, {model.config.num_layers} layers")
    
    # Run debugging
    debug_normalization(model, args.activation_path, args.batch_size, device)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())