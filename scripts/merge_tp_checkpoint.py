#!/usr/bin/env python3
"""Merge tensor-parallel CLT checkpoints into a single consolidated file.

Run this script with exactly the same number of processes (`world_size`) that
was used during training, e.g. for 2-way tensor parallelism:

    torchrun --standalone --nproc_per_node=2 \
        scripts/merge_tp_checkpoint.py \
        --ckpt-dir /path/to/step_1234 \
        --cfg-json /path/to/cfg.json \
        --output /path/to/full_model.safetensors

Only rank 0 writes the final `.safetensors` file.  Other ranks exit after
gathering their tensor shards.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path *before* importing from clt
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from safetensors.torch import save_file as save_safetensors_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load_state_dict

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder


def gather_tensor_parallel_param(param: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """Gather shards of a tensor-parallel parameter along *dim*.

    Each rank passes its local shard (same shape) and receives a list with
    *world_size* shards.  Rank 0 concatenates them along *dim* and returns the
    full tensor; other ranks return an empty tensor (they do not need to keep
    the full copy).
    """
    gathered: List[torch.Tensor] = [torch.empty_like(param) for _ in range(world_size)]
    dist.all_gather(gathered, param)
    if dist.get_rank() == 0:
        return torch.cat(gathered, dim=dim).cpu()
    return torch.tensor([])  # placeholder on non-zero ranks


def merge_state_dict(tp_model: CrossLayerTranscoder, num_features: int, d_model: int) -> Dict[str, torch.Tensor]:
    """Collect the full (non-sharded) state_dict on rank 0."""
    world_size = dist.get_world_size()
    full_state: Dict[str, torch.Tensor] = {}
    rank = dist.get_rank()

    for name, param in tp_model.state_dict().items():
        # Column-parallel weight: [num_features/world, d_model]
        if param.ndim == 2 and param.shape[0] * world_size == num_features and param.shape[1] == d_model:
            gathered = gather_tensor_parallel_param(param, dim=0, world_size=world_size)
            if rank == 0:
                full_state[name] = gathered
        # Row-parallel weight: [d_model, num_features/world]
        elif param.ndim == 2 and param.shape[0] == d_model and param.shape[1] * world_size == num_features:
            gathered = gather_tensor_parallel_param(param, dim=1, world_size=world_size)
            if rank == 0:
                full_state[name] = gathered
        # NEW: Handle log_threshold, sharded on feature dimension
        elif name.endswith("log_threshold") and param.ndim == 2 and param.shape[1] * world_size == num_features:
            gathered = gather_tensor_parallel_param(param, dim=1, world_size=world_size)
            if rank == 0:
                full_state[name] = gathered
        # Bias or vector split along features: [num_features/world]
        elif param.ndim == 1 and param.shape[0] * world_size == num_features:
            gathered = gather_tensor_parallel_param(param, dim=0, world_size=world_size)
            if rank == 0:
                full_state[name] = gathered
        else:
            # Replicated parameters – take rank 0 copy
            if rank == 0:
                full_state[name] = param.cpu()
    return full_state


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True, help="Directory that holds *.distcp shards and .metadata")
    parser.add_argument("--cfg-json", required=True, help="Path to cfg.json that was saved during training")
    parser.add_argument("--output", required=True, help="Path to write consolidated .safetensors file (rank 0)")
    parser.add_argument("--device", default=None, help="Device per rank (default: cuda:<local_rank> or cpu)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Initialise distributed
    # ------------------------------------------------------------------
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(
        args.device if args.device is not None else (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if rank == 0:
        print(f"Running merge with world_size={world_size} on device={device}")

    # ------------------------------------------------------------------
    # Re-create model in TP mode and load sharded checkpoint
    # ------------------------------------------------------------------
    cfg = CLTConfig.from_json(args.cfg_json)
    model = CrossLayerTranscoder(cfg, process_group=dist.group.WORLD, device=device)
    model.eval()

    # Sharded load (each rank gets its part)
    tp_state = model.state_dict()  # template (sharded)
    load_state_dict(
        state_dict=tp_state,
        storage_reader=FileSystemReader(args.ckpt_dir),
        planner=DefaultLoadPlanner(),
        no_dist=False,  # must be False when running with TP ranks
    )
    model.load_state_dict(tp_state)

    # ------------------------------------------------------------------
    # Gather shards → rank 0 builds full state_dict
    # ------------------------------------------------------------------
    full_state = merge_state_dict(model, cfg.num_features, cfg.d_model)

    # ------------------------------------------------------------------
    # Rank 0 writes consolidated file
    # ------------------------------------------------------------------
    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors_file(full_state, str(out_path))
        print(f"✅ Saved merged model to {out_path} (features = {cfg.num_features})")

    dist.barrier()  # ensure file is written before other ranks exit
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
