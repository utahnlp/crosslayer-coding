#!/usr/bin/env python3
"""
Debug script to trace the exact shapes and values in BatchTopK computation.
"""

import torch
import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.local_activation_store import LocalActivationStore
from safetensors.torch import load_file as load_safetensors_file
from clt.models.activations import BatchTopK

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def trace_batch_topk_computation():
    """Trace through the exact BatchTopK computation to find the bug."""
    
    # Create a simple test case
    logger.info("=== TESTING BATCHTOPK DIRECTLY ===")
    
    # Test 1: Simple case - 4 tokens, 10 features, k=2 per token
    batch_size = 4
    num_features = 10
    k_per_token = 2
    
    x = torch.randn(batch_size, num_features)
    logger.info(f"\nTest 1: Simple case")
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  k_per_token: {k_per_token}")
    logger.info(f"  Expected active: {k_per_token * batch_size}")
    
    mask = BatchTopK._compute_mask(x, k_per_token)
    actual_active = mask.sum().item()
    logger.info(f"  Actual active: {actual_active}")
    logger.info(f"  Active per token: {mask.sum(dim=1).tolist()}")
    
    # Test 2: Larger case matching the model
    batch_size = 1024
    num_features = 393216  # 12 layers * 32768 features
    k_per_token = 200
    
    x = torch.randn(batch_size, num_features)
    logger.info(f"\nTest 2: Model-like case")
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  k_per_token: {k_per_token}")
    logger.info(f"  Expected active: {k_per_token * batch_size}")
    
    mask = BatchTopK._compute_mask(x, k_per_token)
    actual_active = mask.sum().item()
    logger.info(f"  Actual active: {actual_active}")
    logger.info(f"  Active per token (first 10): {mask.sum(dim=1)[:10].tolist()}")
    logger.info(f"  Active per token (mean): {mask.sum(dim=1).float().mean().item()}")
    
    # Test 3: Check if there's an issue with how k is passed
    logger.info(f"\nTest 3: Testing different k values")
    for test_k in [1, 10, 100, 200, 1000]:
        mask = BatchTopK._compute_mask(x, test_k)
        actual_active = mask.sum().item()
        avg_per_token = mask.sum(dim=1).float().mean().item()
        logger.info(f"  k={test_k}: total active={actual_active}, avg per token={avg_per_token}")


def trace_model_computation():
    """Trace through actual model computation."""
    
    checkpoint_path = "clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors"
    config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"
    device = torch.device("cuda:0")
    
    logger.info("\n=== TRACING MODEL COMPUTATION ===")
    
    # Load config and model
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CLTConfig(**config_dict)
    
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    state_dict = {k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) 
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
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
    
    inputs, _ = next(activation_store)
    inputs_f32 = {k: v.to(dtype=torch.float32) for k, v in inputs.items()}
    
    # Manually trace through _apply_batch_topk_helper
    logger.info("\nManually tracing _apply_batch_topk_helper...")
    
    # Get preactivations
    preactivations_dict = {}
    with torch.no_grad():
        for layer_idx, layer_input in inputs_f32.items():
            encoder = model.encoder_module.encoders[layer_idx]
            preact = encoder(layer_input)
            preactivations_dict[layer_idx] = preact
            logger.info(f"  Layer {layer_idx} preact shape: {preact.shape}")
    
    # Concatenate (matching _apply_batch_topk_helper logic)
    ordered_preactivations = []
    for layer_idx in range(model.config.num_layers):
        if layer_idx in preactivations_dict:
            ordered_preactivations.append(preactivations_dict[layer_idx])
    
    concatenated = torch.cat(ordered_preactivations, dim=1)
    logger.info(f"\n  Concatenated shape: {concatenated.shape}")
    logger.info(f"  Config batchtopk_k: {config.batchtopk_k}")
    
    # Apply BatchTopK
    from clt.models.activations import _apply_batch_topk_helper
    
    # Monkey-patch to add logging
    original_compute_mask = BatchTopK._compute_mask
    
    def logged_compute_mask(x, k_per_token, x_for_ranking=None):
        logger.info(f"\n  BatchTopK._compute_mask called with:")
        logger.info(f"    x.shape: {x.shape}")
        logger.info(f"    k_per_token: {k_per_token}")
        logger.info(f"    B (batch size from x): {x.size(0)}")
        logger.info(f"    k_total_batch will be: min({k_per_token} * {x.size(0)}, {x.numel()}) = {min(k_per_token * x.size(0), x.numel())}")
        result = original_compute_mask(x, k_per_token, x_for_ranking)
        logger.info(f"    Result mask sum: {result.sum().item()}")
        return result
    
    BatchTopK._compute_mask = logged_compute_mask
    
    try:
        activations = _apply_batch_topk_helper(
            preactivations_dict, config, device, torch.float32, 0, None
        )
        
        logger.info("\n  Activation results:")
        for layer_idx, acts in activations.items():
            active_count = (acts != 0).sum().item()
            avg_per_token = (acts != 0).sum(dim=1).float().mean().item()
            logger.info(f"    Layer {layer_idx}: total active={active_count}, avg per token={avg_per_token}")
            
    finally:
        # Restore original
        BatchTopK._compute_mask = original_compute_mask


def main():
    trace_batch_topk_computation()
    trace_model_computation()


if __name__ == "__main__":
    main()