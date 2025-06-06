#!/usr/bin/env python3
"""
Verify that BatchTopK state (theta values) is being saved and loaded correctly.
This focuses specifically on the BatchTopK activation function state.
"""

import torch
import os
import sys
import json
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clt.config import CLTConfig  # noqa: E402
from clt.models.clt import CrossLayerTranscoder  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402


def create_batchtopk_model(device: torch.device) -> CrossLayerTranscoder:
    """Create a simple BatchTopK model for testing."""
    config = CLTConfig(
        num_features=1024,
        num_layers=4,
        d_model=256,
        activation_fn="batchtopk",
        batchtopk_k=50,
        batchtopk_straight_through=True,
    )
    return CrossLayerTranscoder(config, process_group=None, device=device)


def test_batchtopk_save_load(device: torch.device):
    """Test saving and loading a BatchTopK model."""

    print("\n=== Testing BatchTopK Save/Load ===")

    # Create model
    print("1. Creating BatchTopK model...")
    model1 = create_batchtopk_model(device)

    # Check initial state
    print("\n2. Initial model state:")
    if hasattr(model1, "theta_manager") and model1.theta_manager is not None:
        if hasattr(model1.theta_manager, "log_threshold") and model1.theta_manager.log_threshold is not None:
            log_theta1 = model1.theta_manager.log_threshold
            print(f"   - Has log_threshold: shape={log_theta1.shape}")
            print(f"   - log_threshold dtype: {log_theta1.dtype}")
            print(f"   - log_threshold device: {log_theta1.device}")
            print(f"   - log_threshold mean: {log_theta1.mean().item():.6f}")
            print(f"   - log_threshold std: {log_theta1.std().item():.6f}")
            print(f"   - theta (exp) mean: {log_theta1.exp().mean().item():.6f}")

            # Modify theta values to make them distinguishable
            with torch.no_grad():
                model1.theta_manager.log_threshold.data = torch.randn_like(log_theta1) * 0.5 + 1.0
            print(f"\n   - Modified log_threshold mean: {model1.theta_manager.log_threshold.mean().item():.6f}")
        else:
            print("   ERROR: Model does not have log_threshold!")
            return
    else:
        print("   ERROR: Model does not have theta_manager!")
        return

    # Save model
    print("\n3. Saving model state...")
    state_dict1 = model1.state_dict()
    print(f"   - State dict keys: {list(state_dict1.keys())}")

    # Check if log_threshold is in state dict
    theta_key = None
    for key in state_dict1.keys():
        if "log_threshold" in key:
            theta_key = key
            print(f"   - Found theta key: {key}")
            print(f"   - Theta tensor shape in state dict: {state_dict1[key].shape}")
            print(f"   - Theta tensor mean in state dict: {state_dict1[key].mean().item():.6f}")
            break

    if theta_key is None:
        print("   WARNING: log_threshold not found in state dict!")

    # Save to file
    save_path = "test_batchtopk_model.safetensors"
    save_file(state_dict1, save_path)
    print(f"   - Saved to {save_path}")

    # Create new model and load
    print("\n4. Creating new model and loading state...")
    model2 = create_batchtopk_model(device)

    # Check theta values before loading
    if hasattr(model2, "theta_manager") and hasattr(model2.theta_manager, "log_threshold"):
        log_threshold = model2.theta_manager.log_threshold
        if log_threshold is not None:
            print(f"   - New model log_threshold mean (before load): {log_threshold.mean().item():.6f}")

    # Load state dict
    state_dict2 = load_file(save_path, device=str(device))
    model2.load_state_dict(state_dict2)
    print("   - State loaded successfully")

    # Check theta values after loading
    print("\n5. Comparing theta values...")
    if hasattr(model2, "theta_manager") and hasattr(model2.theta_manager, "log_threshold"):
        log_theta2 = model2.theta_manager.log_threshold
        if log_theta2 is not None:
            print(f"   - Loaded log_threshold mean: {log_theta2.mean().item():.6f}")
            print(f"   - Loaded log_threshold std: {log_theta2.std().item():.6f}")

            # Compare with original
            log_theta1_after = model1.theta_manager.log_threshold
            if log_theta1_after is not None:
                diff = (log_theta1_after - log_theta2).abs().max().item()
                print(f"   - Max absolute difference: {diff:.2e}")
                print(f"   - Values match: {diff < 1e-6}")
            else:
                print("   ERROR: Original model lost theta values!")
        else:
            print("   ERROR: Loaded model does not have theta values!")
    else:
        print("   ERROR: Loaded model does not have theta_manager!")

    # Test forward pass
    print("\n6. Testing forward pass...")
    test_input = torch.randn(10, 256, device=device)
    test_inputs = {0: test_input}

    with torch.no_grad():
        acts1 = model1.get_feature_activations(test_inputs)
        acts2 = model2.get_feature_activations(test_inputs)

        if 0 in acts1 and 0 in acts2:
            act_diff = (acts1[0] - acts2[0]).abs().max().item()
            print(f"   - Activation difference: {act_diff:.2e}")
            print(f"   - Activations match: {act_diff < 1e-5}")

            # Check sparsity
            sparsity1 = (acts1[0] > 0).float().mean().item()
            sparsity2 = (acts2[0] > 0).float().mean().item()
            print(f"   - Model 1 sparsity: {sparsity1:.4f}")
            print(f"   - Model 2 sparsity: {sparsity2:.4f}")

    # Clean up
    os.remove(save_path)
    print("\n7. Test completed!")


def check_checkpoint_theta_state(checkpoint_path: str, device: torch.device):
    """Check theta state in an existing checkpoint."""

    print(f"\n=== Checking Theta State in Checkpoint ===")
    print(f"Checkpoint: {checkpoint_path}")

    # Load config
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "cfg.json")
        consolidated_path = os.path.join(checkpoint_path, "model.safetensors")
    else:
        print("ERROR: Only directory checkpoints are supported")
        return

    if not os.path.exists(config_path):
        print(f"ERROR: Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    print(f"\n1. Model config:")
    print(f"   - Activation function: {config_dict.get('activation_fn')}")
    print(f"   - BatchTopK k: {config_dict.get('batchtopk_k')}")
    print(f"   - Num features: {config_dict.get('num_features')}")
    print(f"   - Num layers: {config_dict.get('num_layers')}")

    if not os.path.exists(consolidated_path):
        print(f"\nERROR: Model file not found at {consolidated_path}")
        return

    # Load state dict directly
    print(f"\n2. Loading state dict from {consolidated_path}...")
    state_dict = load_file(consolidated_path, device="cpu")  # Load to CPU first

    print(f"   - Total keys in state dict: {len(state_dict)}")

    # Look for theta-related keys
    theta_keys = [k for k in state_dict.keys() if "theta" in k.lower() or "threshold" in k.lower()]
    print(f"\n3. Theta-related keys found: {len(theta_keys)}")
    for key in theta_keys:
        tensor = state_dict[key]
        print(f"   - {key}:")
        print(f"     Shape: {tensor.shape}")
        print(f"     Dtype: {tensor.dtype}")
        print(f"     Mean: {tensor.mean().item():.6f}")
        print(f"     Std: {tensor.std().item():.6f}")
        print(f"     Min: {tensor.min().item():.6f}")
        print(f"     Max: {tensor.max().item():.6f}")

        if "log" in key:
            print(f"     Exp mean: {tensor.exp().mean().item():.6f}")

    # Create model and load to verify
    print("\n4. Creating model and loading state...")
    clt_config = CLTConfig(**config_dict)
    model = CrossLayerTranscoder(clt_config, process_group=None, device=device)

    # Move state dict to device
    state_dict_device = {k: v.to(device) for k, v in state_dict.items()}
    model.load_state_dict(state_dict_device)

    print("   - Model loaded successfully")

    # Check model's theta state
    print("\n5. Checking model's theta state after loading:")
    if hasattr(model, "theta_manager") and model.theta_manager is not None:
        if hasattr(model.theta_manager, "log_threshold") and model.theta_manager.log_threshold is not None:
            log_theta = model.theta_manager.log_threshold
            print(f"   - Model has log_threshold: shape={log_theta.shape}")
            print(f"   - log_threshold mean: {log_theta.mean().item():.6f}")
            print(f"   - log_threshold std: {log_theta.std().item():.6f}")
            print(f"   - theta (exp) mean: {log_theta.exp().mean().item():.6f}")
            print(f"   - theta (exp) std: {log_theta.exp().std().item():.6f}")
        else:
            print("   - Model does not have log_threshold (might be converted to JumpReLU)")
    else:
        print("   - Model does not have theta_manager")


def main():
    parser = argparse.ArgumentParser(description="Debug BatchTopK state save/load")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to existing checkpoint to check")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.checkpoint:
        # Check existing checkpoint
        check_checkpoint_theta_state(args.checkpoint, device)
    else:
        # Run basic save/load test
        test_batchtopk_save_load(device)


if __name__ == "__main__":
    main()
