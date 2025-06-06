#!/usr/bin/env python3
"""Debug script to test saving and loading of tensor-parallel CLT models.

This script:
1. Trains a tiny CLT model for a few steps
2. Evaluates it in-memory
3. Saves it in distributed checkpoint format
4. Loads it back
5. Compares evaluations before and after save/load
"""

import torch
import torch.distributed as dist
import os
import json
import tempfile
from typing import Dict

from clt.config import CLTConfig, TrainingConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.trainer import CLTTrainer
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.evaluator import CLTEvaluator
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

# Initialize distributed even for single GPU
if not dist.is_initialized():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def evaluate_model(model: CrossLayerTranscoder, activation_path: str, num_batches: int = 5) -> Dict[str, float]:
    """Evaluate a model and return metrics."""
    # Create activation store
    store = LocalActivationStore(
        dataset_path=activation_path,
        train_batch_size_tokens=512,
        device=device,
        dtype="float16",
        rank=0,  # All ranks see same data for TP
        world=1,
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=False,  # Critical for TP
    )

    # Create evaluator
    evaluator = CLTEvaluator(
        model=model,
        device=device,
        mean_tg=getattr(store, "mean_tg", {}),
        std_tg=getattr(store, "std_tg", {}),
    )

    # Evaluate
    total_nmse = 0.0
    total_ev = 0.0
    count = 0

    iterator = iter(store)
    for _ in range(num_batches):
        try:
            inputs, targets = next(iterator)

            # Use autocast to match training
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                with torch.no_grad():
                    reconstructions = model(inputs)
                    metrics = evaluator._compute_reconstruction_metrics(targets, reconstructions)

            if rank == 0:  # Only accumulate on rank 0
                total_nmse += metrics["reconstruction/normalized_mean_reconstruction_error"]
                total_ev += metrics["reconstruction/explained_variance"]
                count += 1
        except StopIteration:
            break

    store.close()

    if rank == 0 and count > 0:
        return {"nmse": total_nmse / count, "ev": total_ev / count, "batches": count}
    else:
        return {"nmse": 0.0, "ev": 0.0, "batches": 0}


def main():
    if rank == 0:
        print(f"Running debug script with world_size={world_size}")
        print(f"Device: {device}")

    # Use a small existing activation dataset
    activation_path = "./activations_local_100M/gpt2/pile-uncopyrighted_train"

    if not os.path.exists(activation_path):
        if rank == 0:
            print(f"ERROR: Activation path not found: {activation_path}")
            print("Please ensure you have generated activations first.")
        dist.destroy_process_group()
        return

    # Create a small CLT config
    clt_config = CLTConfig(
        num_features=32768,
        num_layers=12,
        d_model=768,
        activation_fn="batchtopk",
        batchtopk_k=200,
        batchtopk_straight_through=True,
        clt_dtype="float32",
    )

    # Create training config for minimal training
    training_config = TrainingConfig(
        learning_rate=1e-4,
        training_steps=10,  # Just 10 steps
        seed=42,
        activation_source="local_manifest",
        activation_path=activation_path,
        activation_dtype="float16",
        train_batch_size_tokens=512,
        sampling_strategy="sequential",
        normalization_method="auto",
        sparsity_lambda=0.0,
        sparsity_c=0.0,
        preactivation_coef=0.0,
        aux_loss_factor=0.03125,
        apply_sparsity_penalty_to_batchtopk=False,
        optimizer="adamw",
        lr_scheduler="linear_final20",
        log_interval=1,
        eval_interval=100,  # Don't eval during training
        checkpoint_interval=100,  # Don't checkpoint during training
        enable_wandb=False,
        precision="fp16",  # Use mixed precision
    )

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = os.path.join(temp_dir, "debug_logs")

        if rank == 0:
            print(f"\n=== Step 1: Training model for {training_config.training_steps} steps ===")

        # Train model
        trainer = CLTTrainer(
            clt_config=clt_config,
            training_config=training_config,
            log_dir=log_dir,
            device=device,
            distributed=True,
        )

        # Train for a few steps
        trained_model = trainer.train(eval_every=1000)  # Don't eval during training

        if rank == 0:
            print("\n=== Step 2: Evaluating in-memory model ===")

        # Evaluate the in-memory model
        metrics_before = evaluate_model(trained_model, activation_path)

        if rank == 0:
            print(f"In-memory model metrics: NMSE={metrics_before['nmse']:.4f}, EV={metrics_before['ev']:.4f}")

        # Get model state for comparison
        if rank == 0:
            # Sample some weights for comparison
            encoder0_weight_sample = list(trained_model.encoder_module.encoders)[0].weight.data[:5, :5].cpu().clone()
            decoder0_0_weight_sample = (
                list(trained_model.decoder_module.decoders.values())[0].weight.data[:5, :5].cpu().clone()
            )
            print(f"\nSample encoder[0] weights before save:\n{encoder0_weight_sample}")
            print(f"\nSample decoder[0->0] weights before save:\n{decoder0_0_weight_sample}")

        dist.barrier()

        if rank == 0:
            print("\n=== Step 3: Model saved to distributed checkpoint (automatic) ===")
            print(f"Checkpoint saved to: {log_dir}/final/")

        # The trainer already saved the model in distributed format
        # Now load it back
        checkpoint_dir = os.path.join(log_dir, "final")

        if rank == 0:
            print("\n=== Step 4: Loading model from distributed checkpoint ===")

        # Load config
        config_path = os.path.join(checkpoint_dir, "cfg.json")
        with open(config_path, "r") as f:
            loaded_config_dict = json.load(f)
        loaded_config = CLTConfig(**loaded_config_dict)

        # Create new model instance
        loaded_model = CrossLayerTranscoder(loaded_config, process_group=dist.group.WORLD, device=device)
        loaded_model.eval()

        # Load distributed checkpoint
        state_dict = loaded_model.state_dict()
        load_state_dict(
            state_dict=state_dict,
            storage_reader=FileSystemReader(checkpoint_dir),
            planner=DefaultLoadPlanner(),
            no_dist=False,
        )
        loaded_model.load_state_dict(state_dict)

        if rank == 0:
            print("Model loaded from distributed checkpoint")

            # Compare weights
            encoder0_weight_after = loaded_model.encoder_module.encoders[0].weight.data[:5, :5].cpu()
            decoder0_weight_after = loaded_model.decoder_module.decoders["0->0"].weight.data[:5, :5].cpu()
            print(f"\nSample encoder[0] weights after load:\n{encoder0_weight_after}")
            print(f"\nSample decoder[0->0] weights after load:\n{decoder0_weight_after}")

            # Check if weights match
            encoder_match = torch.allclose(encoder0_weight_sample, encoder0_weight_after, rtol=1e-5)
            decoder_match = torch.allclose(decoder0_0_weight_sample, decoder0_weight_after, rtol=1e-5)
            print(f"\nEncoder weights match: {encoder_match}")
            print(f"Decoder weights match: {decoder_match}")

        if rank == 0:
            print("\n=== Step 5: Evaluating loaded model ===")

        # Evaluate the loaded model
        metrics_after = evaluate_model(loaded_model, activation_path)

        if rank == 0:
            print(f"Loaded model metrics: NMSE={metrics_after['nmse']:.4f}, EV={metrics_after['ev']:.4f}")

            print("\n=== Comparison ===")
            print(f"NMSE change: {metrics_before['nmse']:.4f} -> {metrics_after['nmse']:.4f}")
            print(f"EV change: {metrics_before['ev']:.4f} -> {metrics_after['ev']:.4f}")

            # Check if metrics are similar
            nmse_similar = abs(metrics_before["nmse"] - metrics_after["nmse"]) < 0.1
            ev_similar = abs(metrics_before["ev"] - metrics_after["ev"]) < 0.05

            if nmse_similar and ev_similar:
                print("\n✓ SUCCESS: Metrics are similar before and after save/load")
            else:
                print("\n✗ FAILURE: Metrics differ significantly after save/load")
                print("This suggests an issue with the save/load process")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\nDebug script complete.")


if __name__ == "__main__":
    main()
