from __future__ import annotations

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
            with open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint32).reshape(-1, 2)
            logger.info(f"Manifest loaded from {path} ({len(data)} rows).")
            return data
        except ValueError as e:
            logger.error(f"Error reshaping manifest data from {path} (expected Nx2): {e}")
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

    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:
        chunk_path = self.dataset_path / f"chunk_{chunk_id}.h5"
        try:
            hf = _open_h5(chunk_path)
        except FileNotFoundError:
            logger.error(f"Chunk file not found for fetch: {chunk_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to open chunk {chunk_id} at {chunk_path}: {e}")
            raise RuntimeError(f"Failed to access chunk HDF5 file: {chunk_path}") from e

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

            for i, lk in enumerate(layer_keys):
                layer_group = hf[lk]
                if "inputs" not in layer_group or "targets" not in layer_group:
                    raise KeyError(f"Missing 'inputs' or 'targets' dataset in layer group '{lk}' of chunk {chunk_id}")

                input_data = layer_group["inputs"][row_indices_h5, :]
                target_data = layer_group["targets"][row_indices_h5, :]
                bufs.append(input_data.tobytes())
                bufs.append(target_data.tobytes())
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
