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
    * Parse the raw bf16 bytes into tensors: [layers] → inputs, targets.
4.  Apply normalization and return `Dict[layer → Tensor]`.

This module requires the server refactor (`/slice` endpoint).
"""

from __future__ import annotations

import logging
import json
from typing import Dict, Optional, Any

import numpy as np
import torch
import requests
from urllib.parse import urljoin, quote
import time

from .manifest_activation_store import ManifestActivationStore

logger = logging.getLogger(__name__)


class RemoteActivationStore(ManifestActivationStore):
    """
    Activation store that fetches data from a remote slice server using
    a manifest file for deterministic, sharded sampling.
    Inherits common logic from ManifestActivationStore.
    """

    def __init__(
        self,
        server_url: str,
        dataset_id: str,
        train_batch_size_tokens: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = "bfloat16",
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
        timeout: int = 60,
        sampling_strategy: str = "sequential",
        normalization_method: str = "none",
        shard_data: bool = True,
    ):
        self.server = server_url.rstrip("/") + "/"
        self.did_enc = quote(dataset_id, safe="")
        self.did_raw = dataset_id
        self.timeout = timeout

        super().__init__(
            train_batch_size_tokens=train_batch_size_tokens,
            device=device,
            dtype=dtype,
            rank=rank,
            world=world,
            seed=seed,
            sampling_strategy=sampling_strategy,
            normalization_method=normalization_method,
            shard_data=shard_data,
        )

        logger.info(
            "RemoteActivationStore initialized for dataset '%s' at %s "
            "(Rank %d/%d, Seed %d, Batch %d, Device %s, Dtype %s, Strategy '%s')",
            self.did_raw,
            self.server,
            self.rank,
            self.world,
            self.seed,
            self.train_batch_size_tokens,
            self.device,
            self.dtype,
            self.sampling_strategy,
        )

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        return self._get_json("info", required=True, retries=3)

    def _load_manifest(self) -> Optional[np.ndarray]:
        max_retries = 3
        base_delay = 2
        url = urljoin(self.server, f"datasets/{self.did_enc}/manifest")
        logger.info(f"Downloading manifest from {url}")

        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=self.timeout)
                r.raise_for_status()
                data = np.frombuffer(r.content, dtype=np.uint32).reshape(-1, 2)
                logger.info(f"Manifest downloaded ({len(data)} rows, {len(r.content) / 1024:.1f} KiB).")
                return data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to download manifest from {url}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Final attempt failed to download manifest. Returning None.")
                    return None
                else:
                    delay = base_delay * (2**attempt)
                    logger.info(f"Retrying manifest download in {delay:.1f} seconds...")
                    time.sleep(delay)
            except ValueError as e:
                logger.error(f"Error reshaping manifest data (expected Nx2 shape): {e}")
                return None
        return None

    def _load_norm_stats(self) -> Optional[Dict[str, Any]]:
        return self._get_json("norm_stats", required=False, retries=1)

    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:
        if row_indices.dtype != np.uint32:
            logger.warning(f"Row indices dtype is {row_indices.dtype}, expected uint32. Casting.")
            row_indices = row_indices.astype(np.uint32)
        rows_list = row_indices.tolist()
        url = urljoin(
            self.server,
            f"datasets/{self.did_enc}/slice?chunk={chunk_id}",
        )
        try:
            r = requests.post(url, json={"rows": rows_list}, timeout=self.timeout)
            r.raise_for_status()
            return r.content
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching slice from {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching slice from {url}: {e}")
            raise RuntimeError(f"Failed to fetch slice for chunk {chunk_id}") from e

    def _get_json(self, endpoint: str, required: bool = True, retries: int = 1) -> Optional[Dict[str, Any]]:
        base_delay = 1
        url = urljoin(self.server, f"datasets/{self.did_enc}/{endpoint}")
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=self.timeout)
                if not r.ok:
                    if r.status_code == 404 and not required:
                        logger.info(f"Optional resource not found at {url} (404)")
                        return None
                    else:
                        r.raise_for_status()
                data = r.json()
                logger.info(f"Successfully fetched JSON from {url} on attempt {attempt + 1}")
                return data
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed to fetch JSON from {url}: {e}")
                if attempt + 1 == retries:
                    logger.error(f"Final attempt failed to fetch JSON from {url}. Returning None or raising error.")
                    if required:
                        raise RuntimeError(
                            f"Failed to fetch required resource {endpoint} after {retries} attempts"
                        ) from e
                    else:
                        return None
                else:
                    delay = base_delay * (2**attempt)
                    logger.info(f"Retrying JSON fetch from {url} in {delay:.1f} seconds...")
                    time.sleep(delay)
        if required:
            raise RuntimeError(f"Failed to fetch required resource {endpoint} after {retries} attempts (logic error)")
        return None
