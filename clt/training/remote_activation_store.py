"""RemoteActivationStore – manifest‑driven exactly‑once sampler.

Replaces the old stateless random‑batch client.  The workflow is:

1.  On startup, download `metadata.json`, `index.bin`, optional
    `norm_stats.json`.
2.  Build a `ShardedIndexSampler` that shuffles the manifest every epoch
    and yields *contiguous slices* of `batch_size` rows (may span
    multiple chunks).  Each GPU rank consumes a disjoint strided subset.
3.  For each batch:
    * Group the next `B` manifest entries by `chunk_id`.
    * For each chunk request `/slice?chunk=X&rows=i,j,k`.
    * Parse the raw bf16 bytes into tensors: \[layers\] → inputs, targets.
4.  Apply normalization and return `Dict[layer → Tensor]`.

This module requires the server refactor (`/slice` endpoint).
"""

from __future__ import annotations

import os, io, math, json, time, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import torch
import requests
from urllib.parse import urljoin, quote
from torch.utils.data import Sampler
from threading import Thread, Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Base abstract class for compatibility with trainer
from .data import BaseActivationStore


# ---------------------------------------------------------------------------
# Helper: ShardedIndexSampler
# ---------------------------------------------------------------------------
class ShardedIndexSampler(Sampler):
    def __init__(
        self,
        idx: np.ndarray,
        batch_size: int,
        seed: int,
        epoch: int,
        rank: int,
        world: int,
    ):
        self.batch = batch_size
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(len(idx))
        self.local = idx[perm][rank::world]  # disjoint slice per GPU

    def __iter__(self):
        for i in range(0, len(self.local), self.batch):
            yield self.local[i : i + self.batch]

    def __len__(self):
        return math.ceil(len(self.local) / self.batch)


# ---------------------------------------------------------------------------
# RemoteActivationStore
# ---------------------------------------------------------------------------
ActivationBatch = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


class RemoteActivationStore(BaseActivationStore):
    def __init__(
        self,
        server_url: str,
        dataset_id: str,
        train_batch_size_tokens: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        prefetch_batches: int = 4,
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
        timeout: int = 60,
    ):
        self.server = server_url.rstrip("/") + "/"
        self.did_enc = quote(dataset_id, safe="")
        self.did_raw = dataset_id
        self.batch_tok = train_batch_size_tokens
        self.timeout = timeout
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # --- fetch metadata & manifest ---
        self._meta = self._get_json("info")
        if self._meta is None:
            raise RuntimeError(
                "Failed to fetch dataset metadata from server. Is the server running and dataset uploaded?"
            )

        # Get dtype from metadata, defaulting to bfloat16 if missing
        self.dtype_str = self._meta.get("dtype", "bfloat16")
        try:
            self.dtype = getattr(torch, self.dtype_str)
            logger.info(f"Using activation dtype from metadata: {self.dtype_str}")
        except AttributeError:
            logger.warning(
                f"Metadata specified invalid dtype '{self.dtype_str}'. Defaulting to bfloat16."
            )
            self.dtype = torch.bfloat16

        # BaseActivationStore required attributes
        self.train_batch_size_tokens = train_batch_size_tokens

        self.rank, self.world, self.seed = rank, world, seed
        # --- fetch metadata & manifest ---
        self.num_layers = self._meta["num_layers"]
        self.d_model = self._meta["d_model"]
        self.chunk_tokens = self._meta["chunk_tokens"]
        self.total_tokens = self._meta["total_tokens"]

        # layer indices list for each layer
        self.layer_indices = list(range(self.num_layers))

        self.manifest = self._download_manifest()
        # norm stats
        self.norm = self._get_json("norm_stats", required=False) or {}
        if self.norm:
            self._prep_norm()
        # epoch / sampler
        self.epoch = 0
        self.sampler_iter = iter(
            ShardedIndexSampler(
                self.manifest,
                self.batch_tok,
                self.seed,
                self.epoch,
                self.rank,
                self.world,
            )
        )

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------
    def _get_json(
        self, endpoint: str, required: bool = True
    ) -> Optional[Dict[str, Any]]:
        url = urljoin(self.server, f"datasets/{self.did_enc}/{endpoint}")
        r = requests.get(url, timeout=30)
        if not r.ok:
            if required:
                r.raise_for_status()
            else:
                return None
        return r.json()

    def _download_manifest(self) -> np.ndarray:
        url = urljoin(self.server, f"datasets/{self.did_enc}/manifest")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype="<u4").reshape(-1, 2)
        return data

    # ------------------------------------------------------------------
    def _prep_norm(self):
        self.mean_in: Dict[int, torch.Tensor] = {}
        self.std_in: Dict[int, torch.Tensor] = {}
        self.mean_tg: Dict[int, torch.Tensor] = {}
        self.std_tg: Dict[int, torch.Tensor] = {}
        for k, d in self.norm.items():
            lid = int(k)
            self.mean_in[lid] = torch.tensor(
                d["inputs"]["mean"], device=self.device, dtype=torch.float32
            )
            self.std_in[lid] = torch.tensor(
                d["inputs"]["std"], device=self.device, dtype=torch.float32
            )
            self.mean_tg[lid] = torch.tensor(
                d["targets"]["mean"], device=self.device, dtype=torch.float32
            )
            self.std_tg[lid] = torch.tensor(
                d["targets"]["std"], device=self.device, dtype=torch.float32
            )

    # ------------------------------------------------------------------
    def _fetch_slice(self, chunk: int, rows: List[int]) -> bytes:
        row_str = ",".join(map(str, rows))
        url = urljoin(
            self.server, f"datasets/{self.did_enc}/slice?chunk={chunk}&rows={row_str}"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.content

    # ------------------------------------------------------------------
    def get_batch(self) -> ActivationBatch:
        try:
            idxs = next(self.sampler_iter)  # (batch, 2) np.ndarray
        except StopIteration:
            # new epoch
            self.epoch += 1
            self.sampler_iter = iter(
                ShardedIndexSampler(
                    self.manifest,
                    self.batch_tok,
                    self.seed,
                    self.epoch,
                    self.rank,
                    self.world,
                )
            )
            idxs = next(self.sampler_iter)
        # group by chunk
        by_chunk: Dict[int, List[int]] = defaultdict(list)
        for chunk_id, row in idxs:
            by_chunk[int(chunk_id)].append(int(row))
        layer_inputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        layer_targets: Dict[int, List[torch.Tensor]] = defaultdict(list)
        # download slices
        for chunk_id, rows in by_chunk.items():
            buf = self._fetch_slice(chunk_id, rows)
            slice_tok = len(rows)
            # --- Calculate byte offsets based on actual dtype --- #
            bytes_per_element = torch.finfo(self.dtype).bits // 8
            bytes_per_row = self.d_model * bytes_per_element
            bytes_per_tensor = (
                slice_tok * bytes_per_row
            )  # for one tensor (input OR target)
            per_layer_bytes = bytes_per_tensor * 2  # for both input and target
            # ---------------------------------------------------- #
            for li in range(self.num_layers):
                start = li * per_layer_bytes
                mid = start + bytes_per_tensor  # End of input tensor
                end = start + per_layer_bytes
                inp_buf = memoryview(buf)[start:mid]
                tgt_buf = memoryview(buf)[mid:end]
                inp = (
                    torch.frombuffer(inp_buf, dtype=self.dtype)
                    .reshape(slice_tok, self.d_model)
                    .to(self.device)
                )
                tgt = (
                    torch.frombuffer(tgt_buf, dtype=self.dtype)
                    .reshape(slice_tok, self.d_model)
                    .to(self.device)
                )
                if self.norm:
                    inp = (inp - self.mean_in[li]) / (self.std_in[li] + 1e-6)
                    tgt = (tgt - self.mean_tg[li]) / (self.std_tg[li] + 1e-6)
                layer_inputs[li].append(inp)
                layer_targets[li].append(tgt)
        # concat pieces from multiple chunks if any
        batch_inp = {
            lid: torch.cat(tensors, dim=0) for lid, tensors in layer_inputs.items()
        }
        batch_tgt = {
            lid: torch.cat(tensors, dim=0) for lid, tensors in layer_targets.items()
        }
        return batch_inp, batch_tgt

    # for trainer compatibility
    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    # ------------------------------------------------------------------
    # BaseActivationStore compatibility helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return minimal state needed to resume iteration."""
        return {
            "store_type": "RemoteActivationStore",
            "epoch": self.epoch,
            "seed": self.seed,
            # Note: we don't serialize sampler iterator position for simplicity.
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state created by `state_dict`. Resets sampler to saved epoch."""
        self.epoch = int(state_dict.get("epoch", 0))
        self.seed = int(state_dict.get("seed", self.seed))
        # Re‑create sampler iterator from saved epoch
        self.sampler_iter = iter(
            ShardedIndexSampler(
                self.manifest,
                self.batch_tok,
                self.seed,
                self.epoch,
                self.rank,
                self.world,
            )
        )

    def __len__(self):
        """Rough estimate of number of batches in the dataset."""
        if self.total_tokens <= 0 or self.train_batch_size_tokens <= 0:
            return 0
        return (
            self.total_tokens + self.train_batch_size_tokens - 1
        ) // self.train_batch_size_tokens
