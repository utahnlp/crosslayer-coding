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

import logging
import json
from typing import Dict, Optional, Any  # Keep Dict, Optional, Any

# Removed unused: os, io, time, Path, Tuple, defaultdict, Thread, Event

import numpy as np
import torch
import requests
from urllib.parse import urljoin, quote

# Import the new base class
from .manifest_activation_store import ManifestActivationStore

# Removed unused imports from previous version

logger = logging.getLogger(__name__)

# Removed unused import: BaseActivationStore (now inherited via ManifestActivationStore)
# Removed unused import: Sampler (now in manifest_activation_store)


# ---------------------------------------------------------------------------
# RemoteActivationStore
# ---------------------------------------------------------------------------
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
        dtype: torch.dtype | str = "bfloat16",  # Match base class default
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
        timeout: int = 60,
    ):
        """
        Initializes the RemoteActivationStore.

        Args:
            server_url: Base URL of the activation server (e.g., "http://localhost:8000").
            dataset_id: Identifier for the dataset on the server (e.g., "gpt2/pile_train").
            train_batch_size_tokens: Number of tokens per training batch.
            device: Device to place tensors on.
            dtype: Desired torch dtype for activations.
            rank: Rank of the current process in distributed training.
            world: Total number of processes in distributed training.
            seed: Random seed for the sampler.
            timeout: HTTP request timeout in seconds.
        """
        self.server = server_url.rstrip("/") + "/"
        self.did_enc = quote(dataset_id, safe="")  # URL-encoded dataset ID
        self.did_raw = dataset_id  # Raw dataset ID for logging/errors
        self.timeout = timeout

        # Initialize the base class - this will call the _load_* methods below
        super().__init__(
            train_batch_size_tokens=train_batch_size_tokens,
            device=device,
            dtype=dtype,
            rank=rank,
            world=world,
            seed=seed,
        )

        logger.info(
            f"RemoteActivationStore initialized for dataset '{self.did_raw}' at {self.server} "
            f"(Rank {self.rank}/{self.world}, Seed {self.seed}, Batch {self.train_batch_size_tokens}, "
            f"Device {self.device}, Dtype {self.dtype})"
        )

    # --- Implementation of abstract methods for remote fetching --- #

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Fetches metadata.json from the server."""
        return self._get_json("info", required=True)

    def _load_manifest(self) -> Optional[np.ndarray]:
        """Downloads index.bin (manifest) from the server."""
        url = urljoin(self.server, f"datasets/{self.did_enc}/manifest")
        logger.info(f"Downloading manifest from {url}")
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            # Manifest is stored as flat uint32 pairs (chunk_id, row_id)
            data = np.frombuffer(r.content, dtype=np.uint32).reshape(-1, 2)
            logger.info(
                f"Manifest downloaded ({len(data)} rows, {r.content.__sizeof__() / 1024:.1f} KiB)."
            )
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download manifest from {url}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Error reshaping manifest data (expected Nx2 shape): {e}")
            return None

    def _load_norm_stats(self) -> Optional[Dict[str, Any]]:
        """Fetches norm_stats.json from the server (optional)."""
        # required=False means it won't raise an error if the file doesn't exist (404)
        return self._get_json("norm_stats", required=False)

    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:
        """
        Fetches raw bytes for specific rows from a chunk via the /slice endpoint.
        Assumes row_indices are sorted uint32.
        """
        if row_indices.dtype != np.uint32:
            logger.warning(
                f"Row indices dtype is {row_indices.dtype}, expected uint32. Casting."
            )
            row_indices = row_indices.astype(np.uint32)

        # Convert numpy array to comma-separated string efficiently
        # Using a simple loop might be faster for small arrays than np.array2string
        row_str = ",".join(map(str, row_indices))

        url = urljoin(
            self.server,
            f"datasets/{self.did_enc}/slice?chunk={chunk_id}&rows={row_str}",
        )
        # Consider adding retry logic here for transient network issues
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()  # Check for HTTP errors
            return r.content
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching slice from {url}")
            raise  # Re-raise Timeout
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching slice from {url}: {e}")
            # You might want custom exceptions here to distinguish network vs server errors
            raise RuntimeError(f"Failed to fetch slice for chunk {chunk_id}") from e

    # --- Helper methods specific to remote fetching --- #

    def _get_json(
        self, endpoint: str, required: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Helper to fetch JSON data from a specific dataset endpoint."""
        url = urljoin(self.server, f"datasets/{self.did_enc}/{endpoint}")
        try:
            r = requests.get(url, timeout=self.timeout)
            if not r.ok:
                if r.status_code == 404 and not required:
                    logger.info(f"Optional resource not found at {url} (404)")
                    return None
                else:
                    # Raise detailed error
                    r.raise_for_status()

            return r.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching JSON from {url}")
            if required:
                raise
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching JSON from {url}: {e}")
            if required:
                raise  # Re-raise if required
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {url}: {e}")
            if required:
                raise
            return None

    # state_dict and load_state_dict are handled by the base class
    # __len__ is handled by the base class
    # __iter__ and __next__ are handled by the base class
