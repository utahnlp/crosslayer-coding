#!/usr/bin/env python3
"""Compare two norm_stats.json files layer-by-layer.

Usage:
    python scripts/compare_norm_stats.py path/to/a/norm_stats.json path/to/b/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np


def _load_norm(path: Path) -> Dict[int, Dict[str, Any]]:
    with open(path) as f:
        raw = json.load(f)
    # cast layer keys to int for convenient lookup
    norm: Dict[int, Dict[str, Any]] = {int(k): v for k, v in raw.items()}
    return norm


def _diff_stats(
    a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]]
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Return {layer: {"inputs_mean": (abs_diff, rel_diff%), ...}}"""
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for layer in sorted(set(a) | set(b)):
        layer_res: Dict[str, Tuple[float, float]] = {}
        for section in ("inputs", "targets"):
            for field in ("mean", "std"):
                key = f"{section}_{field}"
                if layer in a and layer in b and section in a[layer] and section in b[layer]:
                    vec_a = np.asarray(a[layer][section][field], dtype=np.float64)
                    vec_b = np.asarray(b[layer][section][field], dtype=np.float64)
                    if vec_a.shape != vec_b.shape:
                        layer_res[key] = (float("nan"), float("nan"))
                        continue
                    abs_diff = float(np.mean(np.abs(vec_a - vec_b)))
                    denom = np.mean(np.abs(vec_a)) + 1e-12
                    rel_diff = float((abs_diff / denom) * 100.0)
                    layer_res[key] = (abs_diff, rel_diff)
                else:
                    layer_res[key] = (float("nan"), float("nan"))
        out[layer] = layer_res
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare two norm_stats.json files")
    parser.add_argument("file_a", type=Path)
    parser.add_argument("file_b", type=Path)
    parser.add_argument(
        "--top-n", type=int, default=5, help="Show detailed stats for top-N layers with biggest mean differences"
    )
    args = parser.parse_args()

    norm_a = _load_norm(args.file_a)
    norm_b = _load_norm(args.file_b)

    diffs = _diff_stats(norm_a, norm_b)

    print(f"Compared {len(diffs)} layers\n")
    worst_layers = sorted(diffs.items(), key=lambda kv: np.nan_to_num(kv[1]["inputs_mean"][0], nan=0.0), reverse=True)

    print("Layer  |  inputs_mean  |  targets_mean  |  inputs_std  |  targets_std  (abs diff / % rel diff)")
    print("------- | ------------- | -------------- | ------------ | ------------")
    for layer, stats in worst_layers[: args.top_n]:
        im = stats["inputs_mean"]
        tm = stats["targets_mean"]
        isd = stats["inputs_std"]
        tsd = stats["targets_std"]
        print(
            f"{layer:5d} | {im[0]:10.4g} / {im[1]:6.2f}% | {tm[0]:10.4g} / {tm[1]:6.2f}% | "
            f"{isd[0]:10.4g} / {isd[1]:6.2f}% | {tsd[0]:10.4g} / {tsd[1]:6.2f}%"
        )

    print(
        "\nTip: large relative differences (>5-10 %) mean you should use the training norm_stats.json during evaluation."
    )


if __name__ == "__main__":
    main()
