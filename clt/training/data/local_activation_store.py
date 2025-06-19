from __future__ import annotations

from functools import lru_cache
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch

# Import the base class and the HDF5 opener
# This import will be correct once manifest_activation_store.py is also in this directory.
from .manifest_activation_store import ManifestActivationStore, _open_h5

logger = logging.getLogger(__name__)

# Helper to map torch dtypes to numpy dtypes for conversion
TORCH_TO_NUMPY_DTYPE_MAP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.float32,  # NumPy doesn't have bfloat16, so we use float32 as an intermediate
}


class LocalActivationStore(ManifestActivationStore):
    """
    Activation store that reads data from local HDF5 files using a
    manifest file for deterministic, sharded sampling.
    Inherits common logic from ManifestActivationStore.
    Requires `index.bin`, `metadata.json`, and HDF5 chunk files.
    Optionally uses `norm_stats.json`.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        train_batch_size_tokens: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = "bfloat16",
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
        sampling_strategy: str = "sequential",
        normalization_method: str = "none",
        shard_data: bool = True,
    ):
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

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
            "LocalActivationStore initialized for dataset at %s "
            "(Rank %d/%d, Seed %d, Batch %d, Device %s, Dtype %s, Strategy '%s')",
            self.dataset_path,
            self.rank,
            self.world,
            self.seed,
            self.train_batch_size_tokens,
            self.device,
            self.dtype,
            self.sampling_strategy,
        )
        logger.info(f"Found {self.num_chunks} chunks based on manifest.")

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        path = self.dataset_path / "metadata.json"
        if not path.exists():
            logger.error(f"Metadata file not found: {path}")
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {path}: {e}")
            return None
        except OSError as e:
            logger.error(f"Error reading metadata file {path}: {e}")
            return None

    def _load_manifest(self) -> Optional[np.ndarray]:
        path = self.dataset_path / "index.bin"
        if not path.exists():
            logger.error(f"Manifest file not found: {path}")
            return None
        try:
            file_size_bytes = path.stat().st_size
            # Heuristic: older 2-field format is 8 bytes per entry (two uint32),
            # newer 3-field format is 16 bytes per entry (int32, int32, int64).
            if file_size_bytes % 16 == 0:
                # New format with 3 fields (chunk_id, num_tokens, offset)
                manifest_dtype = np.dtype([("chunk_id", np.int32), ("num_tokens", np.int32), ("offset", np.int64)])
                data_structured = np.fromfile(path, dtype=manifest_dtype)
                logger.info(
                    f"Manifest loaded (3-field format) from {path} ({data_structured.shape[0]} chunks). Expanding to per-row entries."
                )

                # --- Consistency check against metadata --- #
                expected_total_tokens: Optional[int] = None
                if hasattr(self, "_meta") and isinstance(self._meta, dict):
                    try:
                        expected_total_tokens = int(self._meta.get("total_tokens", -1))
                    except (ValueError, TypeError):
                        expected_total_tokens = None

                # Expand into per-row entries expected by downstream (chunk_id, row_in_chunk)
                chunk_ids = data_structured["chunk_id"].astype(np.uint32)
                num_tokens_arr = data_structured["num_tokens"].astype(np.uint32)

                # If the sum of num_tokens does not match metadata total_tokens (when available),
                # this file is very likely a *legacy* per-row manifest whose byte-length happens to be
                # divisible by 16 (e.g. an even number of rows).  In that case we fall back to the
                # legacy 2-field parsing logic.
                if expected_total_tokens is not None and expected_total_tokens > 0:
                    parsed_total = int(num_tokens_arr.sum())
                    if parsed_total != expected_total_tokens:
                        logger.warning(
                            "3-field manifest parse produced total_rows=%d but metadata reports %d tokens. "
                            "Falling back to legacy 2-field manifest parsing.",
                            parsed_total,
                            expected_total_tokens,
                        )
                        # Legacy 2-field format already matches expected shape
                        data = np.fromfile(path, dtype=np.uint32).reshape(-1, 2)
                        logger.info(
                            "Manifest re-loaded (legacy 2-field format) from %s (%d rows).",
                            path,
                            data.shape[0],
                        )
                        return data

                # --- Proceed with 3-field expansion --- #
                # Compute total rows
                total_rows = int(num_tokens_arr.sum())
                logger.info(f"Expanding manifest: total rows = {total_rows}")
                # Pre-allocate array
                data = np.empty((total_rows, 2), dtype=np.uint32)
                row_ptr = 0
                for cid, ntok in zip(chunk_ids, num_tokens_arr):
                    if ntok == 0:
                        continue  # Skip empty chunks to avoid broadcast errors
                    data[row_ptr : row_ptr + ntok, 0] = cid  # chunk_id column
                    data[row_ptr : row_ptr + ntok, 1] = np.arange(ntok, dtype=np.uint32)  # row index within chunk
                    row_ptr += ntok
            elif file_size_bytes % 8 == 0:
                # Legacy 2-field format already matches expected shape
                data = np.fromfile(path, dtype=np.uint32).reshape(-1, 2)
                logger.info(f"Manifest loaded (legacy 2-field format) from {path} ({data.shape[0]} rows).")
            else:
                logger.error(
                    f"Manifest file size ({file_size_bytes} bytes) is not compatible with known formats (8 or 16 bytes per row)."
                )
                return None
            return data
        except ValueError as e:
            logger.error(f"Error parsing manifest data from {path}: {e}")
            return None
        except OSError as e:
            logger.error(f"Error reading manifest file {path}: {e}")
            return None

    def _load_norm_stats(self) -> Optional[Dict[str, Any]]:
        path = self.dataset_path / "norm_stats.json"
        if not path.exists():
            logger.info(f"Optional normalization stats file not found: {path}")
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Normalization stats loaded from {path}.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {path}: {e}")
            return None
        except OSError as e:
            logger.error(f"Error reading norm_stats file {path}: {e}")
            return None

    @lru_cache(maxsize=64)
    def _load_chunk(self, chunk_path: str, layer_key: str, data_type: str) -> np.ndarray:
        """Loads entire HDF5 chunk from disk and caches"""

        logger.debug(f"Fetching chunk {chunk_path} / {layer_key} / {data_type}")

        try:
            return _open_h5(chunk_path)[layer_key][data_type][:]
        except FileNotFoundError:
            logger.error(f"Chunk file not found for fetch: {chunk_path}")
            raise
        except KeyError as e:
            raise RuntimeError(
                f"Missing 'inputs' or 'targets' dataset in layer group '{layer_key}' of chunk {chunk_path}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to open chunk at {chunk_path}: {e}")
            raise RuntimeError(f"Failed to access chunk HDF5 file: {chunk_path}") from e

    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:

        chunk_path = self.dataset_path / f"chunk_{chunk_id}.h5"
        if not chunk_path.exists():
            # Fall back to .hdf5 extension (newer generator default)
            alt_path = self.dataset_path / f"chunk_{chunk_id}.hdf5"
            if alt_path.exists():
                chunk_path = alt_path
            else:
                # Provide clearer error message before _open_h5 raises
                logger.error(
                    "Chunk file for chunk_id %d not found with either .h5 or .hdf5 extension in %s",
                    chunk_id,
                    self.dataset_path,
                )

        hf = _open_h5(chunk_path)

        try:

            def _layer_sort_key(name: str) -> int:
                try:
                    return int(name.split("_")[1])
                except (IndexError, ValueError):
                    return 1_000_000

            layer_keys = sorted([k for k in hf.keys() if k.startswith("layer_")], key=_layer_sort_key)
            if not layer_keys:
                raise ValueError(f"No layer groups found in chunk {chunk_id} at {chunk_path}")
            if len(layer_keys) != self.num_layers:
                logger.warning(
                    f"Chunk {chunk_id}: Number of layer groups ({len(layer_keys)}) doesn't match metadata ({self.num_layers}). Using layers found in chunk."
                )

            bufs = []
            if row_indices.dtype != np.uint32:
                row_indices_h5 = row_indices.astype(np.uint32)
            else:
                row_indices_h5 = row_indices

            # PyTorch doesn't support from_buffer for bfloat16, so we can't create bfloat16 bytes.
            # The conversion to bfloat16 must happen after the tensor is created.
            # If the requested dtype is bfloat16, we'll load the data as float32 bytes.
            if self.dtype == torch.bfloat16:
                target_np_dtype = np.float32
            else:
                # e.g., str(torch.float32) -> "torch.float32" -> "float32"
                dtype_str = str(self.dtype).split(".")[-1]
                target_np_dtype = np.dtype(dtype_str)

            for i, lk in enumerate(layer_keys):
                input_data = self._load_chunk(chunk_path, lk, "inputs")[row_indices_h5, :]
                target_data = self._load_chunk(chunk_path, lk, "targets")[row_indices_h5, :]

                # Convert to the target numpy dtype before getting bytes
                input_data_converted = input_data.astype(target_np_dtype)
                target_data_converted = target_data.astype(target_np_dtype)

                bufs.append(input_data_converted.tobytes())
                bufs.append(target_data_converted.tobytes())
            return b"".join(bufs)
        except KeyError as e:
            logger.error(f"Error accessing data within chunk {chunk_id} at {chunk_path}: Missing key {e}")
            raise RuntimeError(f"Data structure error in HDF5 chunk {chunk_id}") from e
        except Exception as e:
            logger.error(
                f"Error processing data from chunk {chunk_id} at {chunk_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to read data slice from chunk {chunk_id}") from e
