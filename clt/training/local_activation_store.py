from __future__ import annotations

import logging
import json

# import os # Unused
from pathlib import Path

# Removed unused List
from typing import Dict, Optional, Any  # Keep needed types

import numpy as np
import torch

# import h5py # Unused in this file directly

# Import the base class and the HDF5 opener
from .manifest_activation_store import ManifestActivationStore, _open_h5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LocalActivationStore Implementation
# ---------------------------------------------------------------------------
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
        dtype: torch.dtype | str = "bfloat16",  # Match base class default
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
        sampling_strategy: str = "sequential",
        normalization_method: str = "none",
    ):
        """
        Initializes the LocalActivationStore.

        Args:
            dataset_path: Path to the directory containing metadata.json,
                          index.bin, norm_stats.json (optional), and chunk_*.h5 files.
            train_batch_size_tokens: Number of tokens per training batch.
            device: Device to place tensors on.
            dtype: Desired torch dtype for activations.
            rank: Rank of the current process in distributed training.
            world: Total number of processes in distributed training.
            seed: Random seed for the sampler.
            sampling_strategy: 'sequential' or 'random_chunk'.
            normalization_method: 'none' or 'mean_std'.
        """
        self.dataset_path = Path(dataset_path).resolve()  # Ensure absolute path
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        # Initialize the base class - this will call the _load_* methods below
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

    # --- Implementation of abstract methods for local fetching --- #

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads metadata.json from the dataset directory."""
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
        """Loads index.bin (manifest) from the dataset directory."""
        path = self.dataset_path / "index.bin"
        if not path.exists():
            logger.error(f"Manifest file not found: {path}")
            return None
        try:
            # Read binary file and interpret as uint32, then reshape
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
        """Loads norm_stats.json from the dataset directory (optional)."""
        path = self.dataset_path / "norm_stats.json"
        if not path.exists():
            logger.info(f"Optional normalization stats file not found: {path}")
            return None  # It's optional, so just return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Normalization stats loaded from {path}.")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {path}: {e}")
            return None  # Treat as missing if corrupt
        except OSError as e:
            logger.error(f"Error reading norm_stats file {path}: {e}")
            return None  # Treat as missing if unreadable

    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:
        """
        Fetches raw bytes for specific rows from a local HDF5 chunk file.
        Uses the LRU-cached _open_h5 helper.
        Assumes row_indices are sorted uint32.
        """
        chunk_path = self.dataset_path / f"chunk_{chunk_id}.h5"

        try:
            # Use the cached HDF5 file opener
            hf = _open_h5(chunk_path)
        except FileNotFoundError:
            logger.error(f"Chunk file not found for fetch: {chunk_path}")
            raise  # Re-raise the specific error
        except Exception as e:
            # Catch other potential h5py errors during open
            logger.error(f"Failed to open chunk {chunk_id} at {chunk_path}: {e}")
            # Invalidate cache for this path if open failed?
            # _open_h5.cache_clear() # Use carefully, maybe only for specific errors
            raise RuntimeError(f"Failed to access chunk HDF5 file: {chunk_path}") from e

        try:
            # --- Get layer keys dynamically --- #
            # Use *numeric* sort so that layer_10, layer_11 come after layer_9
            # This keeps the per-layer byte layout identical to the generator
            # (`_write_chunk`) which iterates over integer-sorted `layer_ids`.
            def _layer_sort_key(name: str) -> int:
                try:
                    return int(name.split("_")[1])
                except (IndexError, ValueError):
                    # Fallback – keep original string order if parsing fails
                    return 1_000_000  # push unparsable names to the end

            layer_keys = sorted([k for k in hf.keys() if k.startswith("layer_")], key=_layer_sort_key)
            if not layer_keys:
                raise ValueError(f"No layer groups found in chunk {chunk_id} at {chunk_path}")
            if len(layer_keys) != self.num_layers:
                logger.warning(
                    f"Chunk {chunk_id}: Number of layer groups ({len(layer_keys)}) doesn't match metadata ({self.num_layers}). Using layers found in chunk."
                )
                # Adjust num_layers used for byte calculation ONLY for this fetch?
                # For simplicity, we'll assume the metadata num_layers is correct for byte calc,
                # but only fetch existing layers. The byte check in base class might fail.

            # --- Efficiently read and concatenate bytes --- #
            # Initialize with an empty list and append
            bufs = []

            # Ensure indices are uint32 for h5py fancy indexing
            if row_indices.dtype != np.uint32:
                row_indices_h5 = row_indices.astype(np.uint32)
            else:
                row_indices_h5 = row_indices

            for i, lk in enumerate(layer_keys):
                layer_group = hf[lk]
                # Check if datasets exist before accessing
                if "inputs" not in layer_group or "targets" not in layer_group:
                    raise KeyError(f"Missing 'inputs' or 'targets' dataset in layer group '{lk}' of chunk {chunk_id}")

                # h5py fancy indexing with a sorted list/array is efficient
                input_data = layer_group["inputs"][row_indices_h5, :]
                target_data = layer_group["targets"][row_indices_h5, :]

                # Convert numpy array slices to bytes
                # Using .tobytes() might be slightly faster than BytesIO for contiguous arrays
                # Append to the list
                bufs.append(input_data.tobytes())
                bufs.append(target_data.tobytes())

            # Concatenate all byte strings into one final bytes object
            return b"".join(bufs)

        except KeyError as e:
            logger.error(f"Error accessing data within chunk {chunk_id} at {chunk_path}: Missing key {e}")
            raise RuntimeError(f"Data structure error in HDF5 chunk {chunk_id}") from e
        except Exception as e:
            # Catch other potential h5py or numpy errors during data access/conversion
            logger.error(
                f"Error processing data from chunk {chunk_id} at {chunk_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to read data slice from chunk {chunk_id}") from e

    # state_dict and load_state_dict are handled by the base class
    # __len__ is handled by the base class
    # __iter__ and __next__ are handled by the base class
