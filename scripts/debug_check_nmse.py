#!/usr/bin/env python3
"""Interactive investigation: compute NMSE / EV for a (possibly tensor-parallel
or merged) CLT checkpoint *without* any JumpReLU conversion.

Open the file in VS Code or another IDE that supports `# %%` cells and run the
cells one by one.

Adjust the default paths below to point at your files.  You can also run the
script non-interactively:

    python scripts/debug_check_nmse.py \
        --ckpt-path /path/to/full_model.safetensors \
        --config     /path/to/cfg.json \
        --activation-data /path/to/activation_dir \
        --norm-stats  /path/to/training_norm_stats.json \
        --device mps --batches 50
"""

# %% imports -----------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.evaluator import CLTEvaluator

# %% helper to override norm stats --------------------------------------------


def override_norm_stats(
    store: LocalActivationStore, stats_path: Optional[Path]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Load *stats_path* and inject it into *store* so that inputs/targets are
    normalised the same way as during training.  Returns (mean_tg, std_tg) so
    the evaluator can de-normalise reconstructions with the **same** stats.
    """
    if stats_path is None:
        return store.mean_tg, store.std_tg  # whatever the store already has

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


# %% CLI ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt-path", required=True, help="Path to .safetensors or .pt model checkpoint file")
    p.add_argument("--config", required=True, help="Path to cfg.json used for training")
    p.add_argument("--activation-data", required=True, help="Directory that contains index.bin & chunks")
    p.add_argument("--norm-stats", default=None, help="norm_stats.json from training run (optional but recommended)")
    p.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | mps (auto detects if None)")
    p.add_argument("--dtype", default="float16", help="dtype to load activations (float16/float32/bfloat16)")
    p.add_argument("--batches", type=int, default=50, help="Number of batches to evaluate")
    p.add_argument("--batch-size", type=int, default=1024, help="Tokens per batch when reading activations")
    return p.parse_args()


# %% main ---------------------------------------------------------------------


def main():
    args = parse_args()
    device_str = args.device or (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device = torch.device(device_str)
    print(f"Device: {device}")

    cfg = CLTConfig.from_json(args.config)

    # --- load checkpoint ---
    ckpt_path = Path(args.ckpt_path)
    state: Dict[str, torch.Tensor]

    print("Loading single-file checkpoint ...")
    if ckpt_path.is_dir():
        print(f"ERROR: --ckpt-path must be a file, but got a directory: {ckpt_path}")
        print("Please merge sharded checkpoints with `scripts/merge_tp_checkpoint.py` first.")
        return

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path), device=device.type)
    else:
        state = torch.load(str(ckpt_path), map_location=device)

    model = CrossLayerTranscoder(cfg, process_group=None, device=device)
    model.load_state_dict(state)
    model.eval()

    # --- activation store ---
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
            print("Store exhausted before reaching requested number of batches.")
            break
        metrics = evaluator._compute_reconstruction_metrics(targets, model(inputs))
        total_ev += metrics["reconstruction/explained_variance"]
        total_nmse += metrics["reconstruction/normalized_mean_reconstruction_error"]
        cnt += 1

    if cnt == 0:
        print("No batches evaluated.")
    else:
        print(f"\nEvaluated {cnt} batches")
        print(f"Avg   NMSE : {total_nmse / cnt:.4f}")
        print(f"Avg   EV   : {total_ev / cnt:.4f}")

    store.close()


if __name__ == "__main__":
    main()
