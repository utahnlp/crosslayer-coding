#!/usr/bin/env python3
"""
Simplified script to compare weights at three stages:
A. In-memory after training (before saving)  
B. Loaded from .distcp files
C. Loaded from merged safetensors file

This focuses only on weight comparison without evaluation.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
import subprocess
from typing import Dict, Any

# Imports for distributed checkpoint loading
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from safetensors.torch import load_file as load_safetensors_file

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer
from clt.models.clt import CrossLayerTranscoder


def get_weight_summary(model: CrossLayerTranscoder, prefix: str = "") -> Dict[str, Any]:
    """Get summary statistics for key weights."""
    summary = {}
    
    # Sample a few key parameters
    key_params = [
        ("encoder_0", model.encoder_module.encoders[0].weight if len(model.encoder_module.encoders) > 0 else None),
        ("decoder_0->0", model.decoder_module.decoders["0->0"].weight if "0->0" in model.decoder_module.decoders else None),
        ("decoder_0->1", model.decoder_module.decoders["0->1"].weight if "0->1" in model.decoder_module.decoders else None),
    ]
    
    for name, param in key_params:
        if param is None:
            continue
        
        data = param.data.cpu().float().numpy()
        
        # Get a 5x5 sample and statistics
        sample = data[:5, :5] if data.ndim > 1 else data[:5]
        
        summary[f"{prefix}{name}"] = {
            "shape": list(param.shape),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "sample_5x5": sample.tolist(),
            "checksum": float(np.sum(np.abs(data)))  # Simple checksum
        }
    
    return summary


def compare_summaries(sum1: Dict[str, Any], sum2: Dict[str, Any], label1: str, label2: str):
    """Compare two weight summaries."""
    print(f"\n{'='*60}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'='*60}")
    
    for key in sorted(set(sum1.keys()) | set(sum2.keys())):
        if key not in sum1:
            print(f"❌ {key}: Missing in {label1}")
            continue
        if key not in sum2:
            print(f"❌ {key}: Missing in {label2}")
            continue
            
        s1 = sum1[key]
        s2 = sum2[key]
        
        # Compare shapes
        if s1["shape"] != s2["shape"]:
            print(f"❌ {key}: Shape mismatch! {s1['shape']} vs {s2['shape']}")
            continue
            
        # Compare checksums
        checksum_diff = abs(s1["checksum"] - s2["checksum"]) / max(s1["checksum"], 1e-10)
        
        if checksum_diff < 1e-5:
            print(f"✅ {key}: Match (checksum diff: {checksum_diff:.2e})")
        else:
            print(f"❌ {key}: MISMATCH!")
            print(f"   Shape: {s1['shape']}")
            print(f"   Mean: {s1['mean']:.6f} vs {s2['mean']:.6f}")
            print(f"   Std: {s1['std']:.6f} vs {s2['std']:.6f}")
            print(f"   Checksum: {s1['checksum']:.6f} vs {s2['checksum']:.6f} (diff: {checksum_diff:.2%})")
            print(f"   Sample [0,0:5]: {s1['sample_5x5'][0][:5]}")
            print(f"            vs: {s2['sample_5x5'][0][:5]}")


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
    
    # Use simple configuration
    output_dir = Path("./debug_weight_check")
    
    # CLT config
    clt_config = CLTConfig(
        num_features=8192,
        num_layers=12,
        d_model=768,
        activation_fn="batchtopk",
        batchtopk_k=200,
        model_name="gpt2",
        clt_dtype="float32",
    )
    
    # Training config
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=10,
        train_batch_size_tokens=1024,
        activation_source="local_manifest",
        activation_path="./activations_local_100M/gpt2/pile-uncopyrighted_train",
        activation_dtype="float16",
        normalization_method="auto",
        sparsity_lambda=0.0,
        aux_loss_factor=0.03125,
        apply_sparsity_penalty_to_batchtopk=False,
        optimizer="adamw",
        optimizer_beta2=0.98,
        lr_scheduler="linear_final20",
        precision="fp16",
        log_interval=10,
        eval_interval=1000,
        checkpoint_interval=10,
        enable_wandb=False,
    )
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("STAGE A: Training model and capturing in-memory weights")
        print(f"{'='*60}")
    
    # Train model
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=str(output_dir),
        device=device,
        distributed=(world_size > 1),
    )
    
    trained_model = trainer.train()
    
    # A. Get in-memory weights
    summary_A = get_weight_summary(trained_model, "A_")
    
    if rank == 0:
        print("\nIn-memory model weight summary:")
        for key, val in summary_A.items():
            print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
    
    # Wait for all ranks to finish training
    if world_size > 1:
        # The trainer destroys the process group, so we need to check if it's still initialized
        if not dist.is_initialized():
            # Reinitialize process group for the rest of the script
            dist.init_process_group(backend="nccl")
    
    checkpoint_dir = output_dir / "latest"
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("STAGE B: Loading model from .distcp files")
        print(f"{'='*60}")
    
    # B. Load from distributed checkpoint
    config_path = output_dir / "cfg.json"
    with open(config_path, "r") as f:
        loaded_config_dict = json.load(f)
    loaded_config = CLTConfig(**loaded_config_dict)
    
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
    
    # Get weights from loaded model
    summary_B = get_weight_summary(loaded_model_B, "B_")
    
    # Compare A vs B
    if rank == 0:
        compare_summaries(summary_A, summary_B, "In-memory (A)", "Loaded from distcp (B)")
    
    # C. Merge and load (only if distributed)
    if world_size > 1:
        if rank == 0:
            print(f"\n{'='*60}")
            print("STAGE C: Merging checkpoint and loading from safetensors")
            print(f"{'='*60}")
        
        dist.barrier()
        
        # Run merge
        merged_path = checkpoint_dir / "merged_model.safetensors"
        merge_script = project_root / "scripts" / "merge_tp_checkpoint.py"
        
        if rank == 0:
            # First, ensure any existing merged file is removed
            if merged_path.exists():
                merged_path.unlink()
        
        dist.barrier()
        
        # Only rank 0 runs the merge script with torchrun
        if rank == 0:
            print(f"Running merge script with torchrun...")
            
            merge_cmd = [
                "torchrun",
                f"--nproc-per-node={world_size}",
                str(merge_script),
                "--ckpt-dir", str(checkpoint_dir),
                "--cfg-json", str(config_path),
                "--output", str(merged_path)
            ]
            
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Merge failed!")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
            else:
                print(f"Merge completed successfully")
        
        dist.barrier()
        
        # Only rank 0 loads and compares the merged model
        if rank == 0 and merged_path.exists():
            print("\nLoading merged model...")
            
            # Create single-GPU model
            single_model = CrossLayerTranscoder(
                loaded_config,
                process_group=None,
                device=device
            )
            single_model.eval()
            
            # Load merged checkpoint
            state_dict_C = load_safetensors_file(str(merged_path))
            single_model.load_state_dict(state_dict_C)
            
            # Get weights
            summary_C = get_weight_summary(single_model, "C_")
            
            # Compare B vs C
            compare_summaries(summary_B, summary_C, "Loaded from distcp (B)", "Loaded from merged (C)")
            
            # Also compare A vs C
            compare_summaries(summary_A, summary_C, "In-memory (A)", "Loaded from merged (C)")
            
            # Check the consolidated model.safetensors file that was saved during training
            print(f"\n{'='*60}")
            print("BONUS: Checking consolidated model.safetensors from training")
            print(f"{'='*60}")
            
            consolidated_path = checkpoint_dir / "model.safetensors"
            if consolidated_path.exists():
                # Load consolidated checkpoint
                state_dict_consolidated = load_safetensors_file(str(consolidated_path))
                
                # Check shapes
                print("\nConsolidated checkpoint shapes:")
                for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                    if key in state_dict_consolidated:
                        print(f"  {key}: {state_dict_consolidated[key].shape}")
                
                # Compare with expected shapes
                print("\nExpected shapes (from merged model):")
                for key in ["encoder_module.encoders.0.weight", "decoder_module.decoders.0->0.weight"]:
                    if key in state_dict_C:
                        print(f"  {key}: {state_dict_C[key].shape}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Weight comparison completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()