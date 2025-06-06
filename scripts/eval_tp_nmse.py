#!/usr/bin/env python3
"""Evaluate NMSE / EV on an *un-merged* tensor-parallel CLT checkpoint.

Usage (example for 2-way TP):

    torchrun --standalone --nproc_per_node=2 scripts/eval_tp_nmse.py \
        --ckpt-dir  clt_training_logs/gpt2_batchtopk/step_90000 \
        --config    clt_training_logs/gpt2_batchtopk/cfg.json \
        --activation-data  ./activations_local_100M/gpt2/pile-uncopyrighted_train \
        --norm-stats       ./activations_local_100M/gpt2/pile-uncopyrighted_train/norm_stats.json \
        --device   cuda \
        --dtype    float16 \
        --batches  50 \
        --batch-size 1024

Only rank 0 iterates over the activation store and prints results; other ranks just
participate in tensor-parallel computation.
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
    p.add_argument("--batch-size", type=int, default=1024, help="Tokens per batch when reading activations")
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

    # --- evaluation only on rank 0 to avoid duplicate data I/O ---
    if rank == 0:
        store = LocalActivationStore(
            dataset_path=args.activation_data,
            train_batch_size_tokens=args.batch_size,
            device=device,
            dtype=args.dtype,
            rank=0,
            world=1,
            seed=42,
            sampling_strategy="sequential",
            normalization_method="auto",
        )
        mean_tg, std_tg = override_norm_stats(store, Path(args.norm_stats) if args.norm_stats else None)
        evaluator = CLTEvaluator(model=model, device=device, mean_tg=mean_tg, std_tg=std_tg)

        iterator = iter(store)
        total_ev, total_nmse, cnt = 0.0, 0.0, 0
        for _ in range(args.batches):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                print("Activation store exhausted early.")
                break
            metrics = evaluator._compute_reconstruction_metrics(targets, model(inputs))
            total_ev += metrics["reconstruction/explained_variance"]
            total_nmse += metrics["reconstruction/normalized_mean_reconstruction_error"]
            cnt += 1
        if cnt == 0:
            print("No batches evaluated.")
        else:
            print(f"\nEvaluated {cnt} batches (rank 0)")
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
