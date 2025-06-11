#!/usr/bin/env python3
"""
Debug script to compare weights at three stages:
A. In-memory after training (before saving)
B. Loaded from .distcp files
C. Loaded from merged safetensors file

This will help identify where the weight corruption occurs.
"""

import os
import sys
import json
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
from typing import Dict, Any
import subprocess

# Imports for distributed checkpoint loading
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from safetensors.torch import save_file as save_safetensors_file
from safetensors.torch import load_file as load_safetensors_file

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer
from clt.models.clt import CrossLayerTranscoder
from clt.training.evaluator import CLTEvaluator
from clt.training.data.activation_store_factory import create_activation_store


def get_weight_samples(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Extract sample weights from key layers for comparison."""
    samples = {}
    
    # Get samples from encoders
    for i in range(min(3, len(model.encoder_module.encoders))):
        encoder = model.encoder_module.encoders[i]
        # Sample a 5x5 patch from the weight matrix
        weight_sample = encoder.weight.data[:5, :5].cpu().clone()
        samples[f"{prefix}encoder_{i}_weight"] = weight_sample
        
        # Also get bias if it exists
        if hasattr(encoder, 'bias') and encoder.bias is not None and hasattr(encoder.bias, 'data'):
            bias_sample = encoder.bias.data[:5].cpu().clone()
            samples[f"{prefix}encoder_{i}_bias"] = bias_sample
    
    # Get samples from decoders
    decoder_keys = list(model.decoder_module.decoders.keys())[:3]  # First 3 decoders
    for key in decoder_keys:
        decoder = model.decoder_module.decoders[key]
        weight_sample = decoder.weight.data[:5, :5].cpu().clone()
        samples[f"{prefix}decoder_{key}_weight"] = weight_sample
        
        if hasattr(decoder, 'bias_param') and decoder.bias_param is not None:
            bias_sample = decoder.bias_param.data[:5].cpu().clone()
            samples[f"{prefix}decoder_{key}_bias"] = bias_sample
    
    # Get theta_log if it exists (for JumpReLU/BatchTopK)
    if hasattr(model, 'theta_module') and model.theta_module is not None:
        for i in range(min(3, len(model.theta_module.theta_logs))):
            theta_log = model.theta_module.theta_logs[i]
            if theta_log is not None:
                theta_sample = theta_log.data.flatten()[:10].cpu().clone()
                samples[f"{prefix}theta_log_{i}"] = theta_sample
    
    return samples


def compare_weight_samples(samples1: Dict[str, torch.Tensor], samples2: Dict[str, torch.Tensor], 
                          label1: str, label2: str, rank: int = 0) -> bool:
    """Compare two sets of weight samples and report differences."""
    all_match = True
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Comparing {label1} vs {label2}")
        print(f"{'='*60}")
    
    for key in sorted(set(samples1.keys()) | set(samples2.keys())):
        if key not in samples1:
            if rank == 0:
                print(f"❌ {key}: Missing in {label1}")
            all_match = False
            continue
            
        if key not in samples2:
            if rank == 0:
                print(f"❌ {key}: Missing in {label2}")
            all_match = False
            continue
        
        w1 = samples1[key]
        w2 = samples2[key]
        
        if w1.shape != w2.shape:
            if rank == 0:
                print(f"❌ {key}: Shape mismatch! {label1}={w1.shape}, {label2}={w2.shape}")
            all_match = False
            continue
        
        # Check if values match
        matches = torch.allclose(w1, w2, rtol=1e-5, atol=1e-6)
        max_diff = torch.max(torch.abs(w1 - w2)).item()
        
        if rank == 0:
            if matches:
                print(f"✅ {key}: Match (max diff: {max_diff:.2e})")
            else:
                print(f"❌ {key}: MISMATCH! Max diff: {max_diff:.2e}")
                print(f"   {label1} sample: {w1.flatten()[:5].tolist()}")
                print(f"   {label2} sample: {w2.flatten()[:5].tolist()}")
                all_match = False
    
    return all_match


def evaluate_model(model: CrossLayerTranscoder, activation_path: str, 
                   rank: int, world_size: int, device: torch.device) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    # Create activation store for evaluation
    from clt.config import TrainingConfig
    
    eval_config = TrainingConfig(
        activation_source="local_manifest",
        activation_path=activation_path,
        train_batch_size_tokens=1024,
        normalization_method="auto",
        activation_dtype="float16",
    )
    
    activation_store = create_activation_store(
        training_config=eval_config,
        model_config=model.config,
        rank=rank,
        world_size=world_size,
        device=device,
        shard_data=(world_size > 1),  # Important for TP
    )
    
    # Create evaluator
    evaluator = CLTEvaluator(
        activation_store=activation_store,
        compute_l0=True,
        compute_density=True,
        explained_variance_method="simple",
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(model, num_batches=10)
    
    return metrics


def main():
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"
    num_features = 8192
    training_steps = 10  # Much shorter for quick test
    
    # CLT configuration
    clt_config = CLTConfig(
        num_features=num_features,
        num_layers=12,  # GPT-2
        d_model=768,    # GPT-2
        activation_fn="batchtopk",
        batchtopk_k=200,
        model_name="gpt2",
        # Don't convert model weights to fp16, let AMP handle it
        clt_dtype="float32",
    )
    
    # Training configuration - matching the working config
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=training_steps,
        train_batch_size_tokens=1024,
        activation_source="local_manifest",
        activation_path=activation_path,
        activation_dtype="float16",
        normalization_method="auto",
        sparsity_lambda=0.0,
        sparsity_c=0.0,
        preactivation_coef=0.0,
        aux_loss_factor=0.03125,
        apply_sparsity_penalty_to_batchtopk=False,
        optimizer="adamw",
        optimizer_beta2=0.98,
        lr_scheduler="linear_final20",
        precision="fp16",
        seed=42,
        sampling_strategy="sequential",
        log_interval=50,
        eval_interval=1000,
        checkpoint_interval=200,  # Less frequent to save space
        dead_feature_window=10000,
        enable_wandb=False,
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "debug_weights"
        
        if rank == 0:
            print(f"\n{'='*60}")
            print("STAGE A: Training model and capturing in-memory weights")
            print(f"{'='*60}")
        
        # Train model
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=str(log_dir),
            device=device,
            distributed=(world_size > 1),
        )
        
        # Train
        trained_model = trainer.train()
        
        # A. Capture in-memory weights
        samples_A = get_weight_samples(trained_model, prefix="A_")
        
        # Evaluate in-memory model
        if rank == 0:
            print("\nEvaluating in-memory model...")
        metrics_A = evaluate_model(trained_model, activation_path, rank, world_size, device)
        if rank == 0:
            print(f"In-memory model: NMSE={metrics_A['nmse']:.4f}, EV={metrics_A['ev']:.4f}")
        
        # The trainer already saved the checkpoint
        checkpoint_dir = log_dir / "latest"
        
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print("STAGE B: Loading model from .distcp files")
            print(f"{'='*60}")
        
        # B. Load from distributed checkpoint
        # Load config
        config_path = log_dir / "cfg.json"
        with open(config_path, "r") as f:
            loaded_config_dict = json.load(f)
        loaded_config = CLTConfig(**loaded_config_dict)
        
        # Create new model instance
        loaded_model_B = CrossLayerTranscoder(
            loaded_config, 
            process_group=dist.group.WORLD if world_size > 1 else None, 
            device=device
        )
        loaded_model_B.eval()
        
        # Load distributed checkpoint
        state_dict_B = loaded_model_B.state_dict()
        load_state_dict(
            state_dict=state_dict_B,
            storage_reader=FileSystemReader(str(checkpoint_dir)),
            planner=DefaultLoadPlanner(),
            no_dist=False,
        )
        loaded_model_B.load_state_dict(state_dict_B)
        
        # Capture weights from loaded model
        samples_B = get_weight_samples(loaded_model_B, prefix="B_")
        
        # Compare A vs B
        match_A_B = compare_weight_samples(samples_A, samples_B, "In-memory (A)", "Loaded from distcp (B)", rank)
        
        # Evaluate loaded model
        if rank == 0:
            print("\nEvaluating model loaded from distcp...")
        metrics_B = evaluate_model(loaded_model_B, activation_path, rank, world_size, device)
        if rank == 0:
            print(f"Loaded from distcp: NMSE={metrics_B['nmse']:.4f}, EV={metrics_B['ev']:.4f}")
        
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print("STAGE C: Merging checkpoint and loading from safetensors")
            print(f"{'='*60}")
        
        # C. Merge checkpoint (only if distributed)
        if world_size > 1:
            merged_path = checkpoint_dir / "merged_model.safetensors"
            
            # Run merge script
            merge_script = project_root / "scripts" / "merge_tp_checkpoint.py"
            merge_cmd = [
                "torchrun", f"--nproc-per-node={world_size}",
                str(merge_script),
                "--ckpt-dir", str(checkpoint_dir),
                "--cfg-json", str(config_path),
                "--output", str(merged_path)
            ]
            
            if rank == 0:
                print(f"Running merge command: {' '.join(merge_cmd)}")
                result = subprocess.run(merge_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Merge failed! stderr: {result.stderr}")
                    sys.exit(1)
                else:
                    print("Merge completed successfully")
            
            # Wait for merge to complete
            dist.barrier()
            
            # Load merged model (single GPU)
            if rank == 0:
                loaded_model_C = CrossLayerTranscoder(
                    loaded_config,
                    process_group=None,  # Single GPU
                    device=device
                )
                loaded_model_C.eval()
                
                # Load merged safetensors
                state_dict_C = load_safetensors_file(str(merged_path))
                loaded_model_C.load_state_dict(state_dict_C)
                
                # Capture weights
                samples_C = get_weight_samples(loaded_model_C, prefix="C_")
                
                # Compare B vs C
                match_B_C = compare_weight_samples(samples_B, samples_C, "Loaded from distcp (B)", "Loaded from merged (C)", rank)
                
                # Also compare A vs C
                match_A_C = compare_weight_samples(samples_A, samples_C, "In-memory (A)", "Loaded from merged (C)", rank)
                
                # Evaluate merged model
                print("\nEvaluating merged model...")
                metrics_C = evaluate_model(loaded_model_C, activation_path, 0, 1, device)  # Single GPU eval
                print(f"Loaded from merged: NMSE={metrics_C['nmse']:.4f}, EV={metrics_C['ev']:.4f}")
        
        # Final summary
        if rank == 0:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"In-memory (A):        NMSE={metrics_A['nmse']:.4f}, EV={metrics_A['ev']:.4f}")
            print(f"Loaded distcp (B):    NMSE={metrics_B['nmse']:.4f}, EV={metrics_B['ev']:.4f}")
            if world_size > 1:
                print(f"Loaded merged (C):    NMSE={metrics_C['nmse']:.4f}, EV={metrics_C['ev']:.4f}")
                print(f"\nWeight comparisons:")
                print(f"A vs B (in-memory vs distcp): {'✅ MATCH' if match_A_B else '❌ MISMATCH'}")
                print(f"B vs C (distcp vs merged):    {'✅ MATCH' if match_B_C else '❌ MISMATCH'}")
                print(f"A vs C (in-memory vs merged): {'✅ MATCH' if match_A_C else '❌ MISMATCH'}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()