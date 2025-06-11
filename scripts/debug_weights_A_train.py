#!/usr/bin/env python3
"""
Script A: Train model and capture in-memory weights.
Saves weight summaries to a JSON file for comparison.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
import numpy as np
from typing import Dict, Any

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
    
    # Print for ALL ranks to verify they're different
    print(f"\nRank {rank} - In-memory model weight summary:")
    for key, val in summary_A.items():
        print(f"  {key}: shape={val['shape']}, checksum={val['checksum']:.6f}")
    
    # Synchronize before saving to ensure all ranks have printed
    if world_size > 1:
        dist.barrier()
    
    # Save summaries to files for each rank
    summary_file = output_dir / f"weight_summary_A_rank{rank}.json"
    with open(summary_file, "w") as f:
        json.dump(summary_A, f, indent=2)
    
    if rank == 0:
        print(f"\nSaved weight summary to {summary_file}")
        print(f"\n{'='*60}")
        print("Stage A completed! Checkpoint saved to debug_weight_check/latest")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()