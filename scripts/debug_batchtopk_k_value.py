#!/usr/bin/env python3
"""
Debug script to investigate why BatchTopK is only activating ~8 features instead of 200.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Hardcoded paths for quick testing
    checkpoint_path = "clt_training_logs/gpt2_batchtopk/full_model_90000.safetensors"
    config_path = "clt_training_logs/gpt2_batchtopk/cfg.json"
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"
    device = torch.device("cuda:0")
    
    logger.info("=== DEBUGGING BATCHTOPK K VALUE ===")
    
    # 1. Load config and check BatchTopK settings
    logger.info("\n1. Loading config...")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    logger.info(f"   Config activation_fn: {config_dict.get('activation_fn')}")
    logger.info(f"   Config batchtopk_k: {config_dict.get('batchtopk_k')}")
    logger.info(f"   Config num_features: {config_dict.get('num_features')}")
    
    config = CLTConfig(**config_dict)
    
    # 2. Create model and check its configuration
    logger.info("\n2. Creating model...")
    model = CrossLayerTranscoder(config, device=device, process_group=None)
    
    logger.info(f"   Model config.activation_fn: {model.config.activation_fn}")
    logger.info(f"   Model config.batchtopk_k: {model.config.batchtopk_k}")
    
    # 3. Load checkpoint
    logger.info("\n3. Loading checkpoint...")
    state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    state_dict = {k: v.to(device=device, dtype=model.encoder_module.encoders[0].weight.dtype) 
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # 4. Get a batch of data
    logger.info("\n4. Getting test batch...")
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
    
    # 5. Manually trace through the encoder to see what's happening
    logger.info("\n5. Tracing through encoder...")
    
    # Get preactivations from one layer
    layer_idx = 0
    layer_input = inputs[layer_idx]
    encoder = model.encoder_module.encoders[layer_idx]
    
    # Compute preactivations
    with torch.no_grad():
        preact = encoder(layer_input)
    
    logger.info(f"   Layer {layer_idx} preactivation shape: {preact.shape}")
    logger.info(f"   Layer {layer_idx} preactivation stats: mean={preact.mean():.4f}, std={preact.std():.4f}")
    
    # 6. Test BatchTopK directly
    logger.info("\n6. Testing BatchTopK activation directly...")
    
    # Import the activation function
    from clt.models.activations import BatchTopK
    
    # Test with different k values
    for test_k in [8, 50, 200, 1000]:
        mask = BatchTopK._compute_mask(preact, k_per_token=test_k)
        num_active = mask.sum().item()
        avg_per_token = mask.float().sum(dim=-1).mean().item()
        logger.info(f"   k={test_k}: total active={num_active}, avg per token={avg_per_token:.1f}")
    
    # 7. Run full forward pass and check activations
    logger.info("\n7. Running full model forward pass...")
    model.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # Get feature activations
            feature_acts = model.get_feature_activations(inputs)
            
            # Check how the model computes activations
            logger.info("   Checking model's actual k value during forward pass...")
            
            # The key is to understand what k value is being used
            # Let's check the activation function being called
            if hasattr(model, '_apply_activation'):
                logger.info(f"   Model has _apply_activation method")
            
            # Check activations per layer
            for layer_idx, acts in feature_acts.items():
                num_active = (acts != 0).sum(dim=-1).float().mean().item()
                total_active = (acts != 0).sum().item()
                logger.info(f"   Layer {layer_idx}: avg active per token={num_active:.1f}, "
                           f"total active={total_active}")
    
    # 8. Check if there's a discrepancy in how activations are computed
    logger.info("\n8. Checking encoder module activation logic...")
    
    # Look at how the encoder module applies activations
    if hasattr(model.encoder_module, 'activation_fn'):
        logger.info(f"   Encoder module activation_fn: {model.encoder_module.activation_fn}")
    
    # Try to trace the actual computation
    logger.info("\n9. Detailed trace of activation computation...")
    
    # Get all preactivations
    preactivations = {}
    with torch.no_grad():
        for layer_idx, layer_input in inputs.items():
            encoder = model.encoder_module.encoders[layer_idx]
            preact = encoder(layer_input)
            preactivations[layer_idx] = preact
    
    # Check what _apply_activation does
    if model.config.activation_fn == "batchtopk":
        # The model should concatenate all preactivations and apply BatchTopK globally
        logger.info("   Model uses BatchTopK - should apply globally across all layers")
        
        # Manually compute what should happen
        all_preacts = []
        for i in range(model.config.num_layers):
            if i in preactivations:
                all_preacts.append(preactivations[i])
        
        if all_preacts:
            concat_preacts = torch.cat(all_preacts, dim=1)
            logger.info(f"   Concatenated preactivations shape: {concat_preacts.shape}")
            logger.info(f"   Expected k value: {model.config.batchtopk_k}")
            logger.info(f"   Expected total active: {model.config.batchtopk_k * concat_preacts.shape[0]}")
            
            # Test what mask would be computed
            test_mask = BatchTopK._compute_mask(concat_preacts, k_per_token=model.config.batchtopk_k)
            actual_active = test_mask.sum().item()
            logger.info(f"   Actual active with k={model.config.batchtopk_k}: {actual_active}")
            
    logger.info("\n=== END DEBUGGING ===")


if __name__ == "__main__":
    main()