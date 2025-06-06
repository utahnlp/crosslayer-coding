#!/usr/bin/env python3
"""Evaluate NMSE / EV on an *un-merged* tensor-parallel CLT checkpoint.

Fixed version that properly handles mixed precision and dtypes.

Usage (example for 2-way TP):

    torchrun --standalone --nproc_per_node=2 scripts/eval_tp_nmse_fixed.py \
        --ckpt-dir  clt_training_logs/gpt2_batchtopk/step_90000 \
        --config    clt_training_logs/gpt2_batchtopk/cfg.json \
        --activation-data  ./activations_local_100M/gpt2/pile-uncopyrighted_train \
        --norm-stats       ./activations_local_100M/gpt2/pile-uncopyrighted_train/norm_stats.json \
        --device   cuda \
        --dtype    float16 \
        --batches  50 \
        --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

# Project imports
from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.evaluator import CLTEvaluator


def override_norm_stats(
    store: LocalActivationStore, stats_path: Optional[Path]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Inject *stats_path* into *store* so evaluator can de-normalise outputs."""
    if stats_path is None:
        return store.mean_tg, store.std_tg

    with stats_path.open() as f:
        stats_json = json.load(f)

    mean_tg: Dict[int, torch.Tensor] = {}
    std_tg: Dict[int, torch.Tensor] = {}
    mean_in: Dict[int, torch.Tensor] = {}
    std_in: Dict[int, torch.Tensor] = {}

    for layer_idx_str, stats in stats_json.items():
        li = int(layer_idx_str)
        if "inputs" in stats:
            mean_in[li] = torch.tensor(stats["inputs"]["mean"], dtype=torch.float32, device=store.device).unsqueeze(0)
            std_in[li] = (
                torch.tensor(stats["inputs"]["std"], dtype=torch.float32, device=store.device) + 1e-6
            ).unsqueeze(0)
        if "targets" in stats:
            mean_tg[li] = torch.tensor(stats["targets"]["mean"], dtype=torch.float32, device=store.device).unsqueeze(0)
            std_tg[li] = (
                torch.tensor(stats["targets"]["std"], dtype=torch.float32, device=store.device) + 1e-6
            ).unsqueeze(0)

    store.mean_in, store.std_in = mean_in, std_in
    store.mean_tg, store.std_tg = mean_tg, std_tg
    store.apply_normalization = True
    return mean_tg, std_tg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="Directory that holds *.distcp shards and .metadata")
    p.add_argument("--config", required=True, help="Path to cfg.json used during training")
    p.add_argument("--activation-data", required=True, help="Directory with index.bin & chunks")
    p.add_argument("--norm-stats", default=None, help="Optional training norm_stats.json for de-normalisation")
    p.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | mps (auto if None)")
    p.add_argument("--dtype", default="float16", help="Activation dtype to load (float16/float32/bfloat16)")
    p.add_argument("--batches", type=int, default=50, help="Number of batches to evaluate")
    p.add_argument("--batch-size", type=int, default=512, help="Tokens per batch (should match training)")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    return p.parse_args()


def init_dist() -> Tuple[int, int, int]:
    """Initialise (or reuse) torch.distributed default group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, local_rank, world_size


def main() -> None:
    args = parse_args()

    rank, local_rank, world_size = init_dist()

    if args.device is None:
        # Auto-select: CUDA with local rank if available, else MPS, else CPU
        if torch.cuda.is_available():
            device_str = f"cuda:{local_rank}"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        # User passed --device.  If they said just "cuda", expand to cuda:<local_rank>
        if args.device.lower() == "cuda":
            device_str = f"cuda:{local_rank}"
        else:
            device_str = args.device  # trust they know what they're doing

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if rank == 0:
        print(f"Using world_size={world_size}, device per rank: {device}")

    # --- load config & TP model ---
    cfg = CLTConfig.from_json(args.config)
    if rank == 0:
        print(
            f"Model config: activation_fn={cfg.activation_fn}, num_features={cfg.num_features}, d_model={cfg.d_model}"
        )
        if cfg.activation_fn == "batchtopk":
            print(f"BatchTopK settings: k={cfg.batchtopk_k}")

    # CRITICAL FIX: Override the model dtype to match training
    # During training with --precision fp16, the model uses float16 computations
    original_clt_dtype = cfg.clt_dtype
    cfg.clt_dtype = args.dtype  # Use the activation dtype for model dtype
    if rank == 0:
        print(f"Overriding model dtype from {original_clt_dtype} to {cfg.clt_dtype} to match training")

    model = CrossLayerTranscoder(cfg, process_group=dist.group.WORLD, device=device)
    model.eval()

    # load sharded checkpoint into model.state_dict()
    tp_state = model.state_dict()
    load_state_dict(
        state_dict=tp_state,
        storage_reader=FileSystemReader(args.ckpt_dir),
        planner=DefaultLoadPlanner(),
        no_dist=False,  # we *are* running distributed
    )
    model.load_state_dict(tp_state)
    if rank == 0:
        print("Loaded TP checkpoint")

    # Create activation store - CRITICAL: all ranks must see the same data for TP
    store = LocalActivationStore(
        dataset_path=args.activation_data,
        train_batch_size_tokens=args.batch_size,
        device=device,
        dtype=args.dtype,
        rank=0,  # All ranks use rank 0's data
        world=1,  # Treat as single process for data loading
        seed=42,
        sampling_strategy="sequential",
        normalization_method="auto",
        shard_data=False,  # CRITICAL: Don't shard data across ranks in TP mode
    )

    # Override norm stats if provided
    mean_tg, std_tg = override_norm_stats(store, Path(args.norm_stats) if args.norm_stats else None)
    evaluator = CLTEvaluator(model=model, device=device, mean_tg=mean_tg, std_tg=std_tg)

    iterator = iter(store)
    total_ev, total_nmse, cnt = 0.0, 0.0, 0

    # Debug first batch
    debug_first_batch = args.debug

    for batch_idx in range(args.batches):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            if rank == 0:
                print("Activation store exhausted early.")
            break

        # Debug output for first batch
        if debug_first_batch and batch_idx == 0 and rank == 0:
            print("\n--- Debug info for first batch ---")
            print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            print(f"Input dtypes: {[(k, v.dtype) for k, v in inputs.items()][:3]}")
            print(f"Model dtype: {next(model.parameters()).dtype}")

        # All ranks process the same batch
        with torch.no_grad():
            # Use autocast to match training behavior
            # During training, forward passes were done with fp16 autocast
            autocast_device_type = device.type if device.type in ["cuda", "mps"] else "cpu"
            autocast_enabled = (args.dtype == "float16" and device.type == "cuda") or (
                args.dtype == "bfloat16" and device.type in ["cuda", "cpu"]
            )
            autocast_dtype = (
                torch.float16
                if args.dtype == "float16"
                else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
            )

            with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype, enabled=autocast_enabled):
                # Get reconstructions
                reconstructions = model(inputs)

                if debug_first_batch and batch_idx == 0:
                    # Check feature activations
                    feature_acts = model.get_feature_activations(inputs)
                    if rank == 0:
                        print(f"\nFeature activation shapes: {[(k, v.shape) for k, v in feature_acts.items()][:3]}")
                        print(f"Feature activation dtypes: {[(k, v.dtype) for k, v in feature_acts.items()][:3]}")

                metrics = evaluator._compute_reconstruction_metrics(targets, reconstructions)

        # Only rank 0 accumulates metrics to avoid double counting
        if rank == 0:
            total_ev += metrics["reconstruction/explained_variance"]
            total_nmse += metrics["reconstruction/normalized_mean_reconstruction_error"]
            cnt += 1

            if debug_first_batch and batch_idx == 0:
                print(
                    f"\nBatch 0 metrics - NMSE: {metrics['reconstruction/normalized_mean_reconstruction_error']:.4f}, EV: {metrics['reconstruction/explained_variance']:.4f}"
                )

    # Only rank 0 reports results
    if rank == 0:
        if cnt == 0:
            print("No batches evaluated.")
        else:
            print(f"\nEvaluated {cnt} batches")
            print(f"Avg NMSE : {total_nmse / cnt:.4f}")
            print(f"Avg EV   : {total_ev / cnt:.4f}")

    store.close()

    # Barrier so all ranks wait until rank0 prints
    dist.barrier()
    if rank == 0:
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
