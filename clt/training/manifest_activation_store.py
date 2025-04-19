from __future__ import annotations

import logging
import time
from collections import defaultdict

# import json # Unused
from abc import ABC, abstractmethod
from pathlib import Path

# Removed unused List, Generator
from typing import Dict, Tuple, Optional, Any, List

# Removed unused defaultdict
# from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Sampler
import h5py  # Needed for _open_h5 cache type hint

# Import BaseActivationStore from the original data module
from .data import BaseActivationStore

logger = logging.getLogger(__name__)

# Type hint for the generator output & batch format
ActivationBatch = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


# ---------------------------------------------------------------------------
# Helper: ChunkRowSampler (Moved from remote_activation_store.py)
# ---------------------------------------------------------------------------
class ChunkRowSampler(Sampler):
    """
    Shuffle chunk order each epoch; inside each chunk iterate rows
    sequentially (already random due to shuffle at generation).
    Strides by GPU rank so there is no overlap. Yields (chunk_id, row_id) pairs.
    """

    def __init__(
        self,
        chunk_sizes: Dict[int, int] | np.ndarray,
        num_chunks: int,
        batch: int,
        seed: int,
        epoch: int,
        rank: int,
        world: int,
    ):
        # Handle chunk_sizes as either a dict mapping chunk_id → size or an array of sizes
        if isinstance(chunk_sizes, dict):
            self.chunk_sizes = chunk_sizes
        else:
            # Create dict from array (index → size)
            self.chunk_sizes = {i: int(size) for i, size in enumerate(chunk_sizes) if size > 0}

        self.batch = batch
        self.rank = rank
        self.world = world
        self.num_chunks = num_chunks
        self.seed = seed
        self.epoch = epoch

        if not self.chunk_sizes:
            raise ValueError("chunk_sizes cannot be empty")
        if self.batch <= 0:
            raise ValueError("batch size must be positive")
        if not (0 <= self.rank < self.world):
            raise ValueError(f"Invalid rank/world: {rank}/{world}")

        # Initialize generator here to set chunk_order for the first epoch
        self._reset_generator()

    def _reset_generator(self):
        """Resets the numpy random generator and shuffles chunk order for a new epoch."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.chunk_order = rng.permutation(self.num_chunks)

        # Create chunk_id → row_ids mapping for each chunk
        self.rows_by_chunk = {}
        for chunk_id, chunk_size in self.chunk_sizes.items():
            if chunk_id >= self.num_chunks:
                continue  # Skip if chunk_id is beyond what we consider in this epoch

            # For each chunk, get the subset of rows assigned to this rank
            chunk_rows = np.arange(chunk_size, dtype=np.uint32)
            self.rows_by_chunk[chunk_id] = chunk_rows[self.rank :: self.world]

        # Count total rows across all chunks for this rank
        total_rows = sum(len(rows) for rows in self.rows_by_chunk.values())
        self.total_batches_this_rank = total_rows // self.batch

        self.current_chunk_idx_in_order = 0
        self.current_row_offset = 0

    def __iter__(self):
        self.current_chunk_idx_in_order = 0
        self.current_row_offset = 0
        return self

    def __next__(self):
        if self.current_chunk_idx_in_order >= len(self.chunk_order):
            # End of epoch reached
            raise StopIteration

        # Get the actual chunk ID for this step
        current_chunk_id = self.chunk_order[self.current_chunk_idx_in_order]

        # Chunk doesn't exist or has no rows assigned to this rank?
        if current_chunk_id not in self.rows_by_chunk or not len(self.rows_by_chunk[current_chunk_id]):
            # Skip to next chunk and try again
            self.current_chunk_idx_in_order += 1
            self.current_row_offset = 0
            return self.__next__()

        # Get the rows for this chunk assigned to this rank
        rows_for_this_chunk = self.rows_by_chunk[current_chunk_id]

        # Determine batch slice
        start_row_offset = self.current_row_offset
        end_row_offset = min(start_row_offset + self.batch, len(rows_for_this_chunk))
        batch_rows = rows_for_this_chunk[start_row_offset:end_row_offset]

        # If we didn't get enough rows for a full batch, move to next chunk
        if len(batch_rows) < self.batch:
            self.current_chunk_idx_in_order += 1
            self.current_row_offset = 0
            return self.__next__()  # Try next chunk

        # Prepare output: (batch, 2) numpy array [chunk_id, row_id]
        batch_output = np.stack(
            [np.full(len(batch_rows), current_chunk_id, dtype=np.uint32), batch_rows],
            axis=1,
        )

        # Update offset for the next batch within the current chunk
        if end_row_offset >= len(rows_for_this_chunk):
            self.current_chunk_idx_in_order += 1
            self.current_row_offset = 0
        else:
            self.current_row_offset = end_row_offset

        return batch_output

    def __len__(self):
        """Return the total number of batches this rank will process in an epoch."""
        # Count total valid rows for this rank across all chunks
        total_rows = 0
        for chunk_id, rows in self.rows_by_chunk.items():
            if chunk_id < self.num_chunks:  # Only consider chunks in our range
                total_rows += len(rows)

        # Calculate batches (integer division drops partial batches)
        return total_rows // self.batch

    def set_epoch(self, epoch: int):
        """Sets the epoch for this sampler, resetting the RNG and chunk order."""
        self.epoch = epoch
        self._reset_generator()


# ---------------------------------------------------------------------------
# ManifestActivationStore - Base Class
# ---------------------------------------------------------------------------
class ManifestActivationStore(BaseActivationStore, ABC):
    """
    Base class for activation stores that use a manifest (`index.bin`)
    for deterministic, sharded, exactly-once sampling via `ChunkRowSampler`.

    Subclasses must implement fetching mechanisms for metadata, manifest,
    norm stats, and the core `_fetch_slice` method.
    """

    def __init__(
        self,
        train_batch_size_tokens: int = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = "bfloat16",  # Default to bf16 as per new generator standard
        rank: int = 0,
        world: int = 1,
        seed: int = 42,
    ):
        self.train_batch_size_tokens = train_batch_size_tokens  # From Base
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = 0  # Initial epoch

        # Device setup
        _device_input = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(_device_input) if isinstance(_device_input, str) else _device_input

        # Dtype setup
        if isinstance(dtype, str):
            try:
                self.dtype = getattr(torch, dtype)
            except AttributeError:
                logger.warning(f"Invalid dtype string '{dtype}'. Defaulting to torch.bfloat16.")
                self.dtype = torch.bfloat16
        elif isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            logger.warning(f"Invalid dtype type '{type(dtype)}'. Defaulting to torch.bfloat16.")
            self.dtype = torch.bfloat16

        # --- Load Core Metadata (must be implemented by subclass) ---
        self._meta = self._load_metadata()
        if not self._meta:
            raise RuntimeError("Failed to load dataset metadata.")

        # --- Populate BaseActivationStore attributes from metadata ---
        try:
            self.num_layers = int(self._meta["num_layers"])
            self.d_model = int(self._meta["d_model"])
            self.chunk_tokens = int(self._meta.get("chunk_tokens", -1))  # Target size
            self.total_tokens = int(self._meta["total_tokens"])
            # Use metadata dtype if available and consistent, else warn
            meta_dtype_str = self._meta.get("dtype")
            if meta_dtype_str:
                try:
                    meta_dtype = getattr(torch, meta_dtype_str)
                    if meta_dtype != self.dtype:
                        logger.warning(
                            f"Metadata dtype ({meta_dtype_str}) differs from requested dtype ({self.dtype}). Using requested dtype."
                        )
                except AttributeError:
                    logger.warning(f"Metadata contains invalid dtype string '{meta_dtype_str}'. Ignoring.")
            else:
                logger.info(f"No dtype specified in metadata, using requested/default: {self.dtype}")

        except KeyError as e:
            raise ValueError(f"Metadata dictionary missing required key: {e}")
        except ValueError as e:
            raise ValueError(f"Metadata contains invalid numeric value: {e}")

        # BaseActivationStore required attributes
        self.layer_indices = list(range(self.num_layers))  # Simple range for now

        # --- Load Manifest (must be implemented by subclass) ---
        self.manifest = self._load_manifest()
        if self.manifest is None or self.manifest.size == 0:
            raise RuntimeError("Failed to load or manifest is empty.")
        if self.manifest.ndim != 2 or self.manifest.shape[1] != 2:
            raise ValueError(f"Manifest must be Nx2 array, got shape {self.manifest.shape}")
        if self.manifest.dtype != np.uint32:
            logger.warning(f"Manifest dtype is {self.manifest.dtype}, expected uint32. Casting.")
            self.manifest = self.manifest.astype(np.uint32)

        # Determine number of chunks and evaluate actual rows per chunk from manifest
        self.num_chunks = int(self.manifest[:, 0].max()) + 1

        counts = np.bincount(self.manifest[:, 0])
        if len(counts) == 0:
            raise ValueError("Manifest appears to be empty after bincount.")

        median_rows = int(np.median(counts[counts > 0]))
        min_rows = int(counts[counts > 0].min())

        # Store full array of chunk sizes for the sampler
        # This enables per-chunk handling of sizes
        self.chunk_sizes = counts

        # Decide which value to trust for rows_per_chunk.
        # If metadata chunk_tokens differs from the manifest median by more than
        # 10 %, assume the metadata contained only the *threshold* used by the
        # generator rather than the true chunk size and fall back to the
        # manifest‑derived value.
        if self.chunk_tokens > 0:
            rel_diff = abs(self.chunk_tokens - median_rows) / max(median_rows, 1)
            if rel_diff > 0.10:
                logger.warning(
                    "'chunk_tokens' in metadata (%s) differs from median chunk "
                    "size in manifest (%s) by more than 10%% – treating it as a "
                    "threshold and using the manifest value instead.",
                    self.chunk_tokens,
                    median_rows,
                )
                self.rows_per_chunk = min_rows  # guarantee within all chunks
            else:
                # Ensure we don't exceed the smallest chunk
                if self.chunk_tokens > min_rows:
                    logger.warning(
                        "Reducing rows_per_chunk from %s to %s (smallest chunk) to avoid out‑of‑range indexing.",
                        self.chunk_tokens,
                        min_rows,
                    )
                    self.rows_per_chunk = min_rows
                else:
                    self.rows_per_chunk = self.chunk_tokens
        else:
            logger.info(
                "No valid 'chunk_tokens' in metadata – using actual chunk sizes from manifest (median: %s).",
                median_rows,
            )
            self.rows_per_chunk = min_rows

        # --- Load Normalization Stats (optional, subclass responsibility) ---
        self.norm_stats_data = self._load_norm_stats()
        self.apply_normalization = bool(self.norm_stats_data)
        if self.apply_normalization:
            self._prep_norm()
        else:
            # Initialize empty dicts with types for linter
            self.mean_in: Dict[int, torch.Tensor] = {}
            self.std_in: Dict[int, torch.Tensor] = {}
            self.mean_tg: Dict[int, torch.Tensor] = {}
            self.std_tg: Dict[int, torch.Tensor] = {}

        # --- Setup Sampler ---
        self.sampler = ChunkRowSampler(
            chunk_sizes=self.chunk_sizes,
            num_chunks=self.num_chunks,
            batch=self.train_batch_size_tokens,
            seed=self.seed,
            epoch=self.epoch,
            rank=self.rank,
            world=self.world,
        )
        self.sampler_iter = iter(self.sampler)

        # --- Precompute byte offsets based on dtype and d_model ---
        self.bytes_per_element = torch.finfo(self.dtype).bits // 8
        self.bytes_per_row = self.d_model * self.bytes_per_element
        # bytes_per_tensor = slice_tok * bytes_per_row (depends on batch size)
        # per_layer_bytes = bytes_per_tensor * 2 (input + target)

        # --- Previous chunk tracking for logging ---
        self.last_processed_chunk_ids: Optional[set[int]] = None

    @abstractmethod
    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load dataset metadata (e.g., from metadata.json)."""
        pass

    @abstractmethod
    def _load_manifest(self) -> Optional[np.ndarray]:
        """Load the manifest file (index.bin) as an Nx2 uint32 numpy array."""
        pass

    @abstractmethod
    def _load_norm_stats(self) -> Optional[Dict[str, Any]]:
        """Load normalization statistics (e.g., from norm_stats.json)."""
        pass

    @abstractmethod
    def _fetch_slice(self, chunk_id: int, row_indices: np.ndarray) -> bytes:
        """
        Fetch the raw bytes corresponding to the specified rows from the given chunk.
        The returned bytes should contain concatenated data for all layers:
        Layer0_Inputs | Layer0_Targets | Layer1_Inputs | Layer1_Targets | ...
        """
        pass

    def _prep_norm(self):
        """Prepare normalization tensors from loaded JSON data."""
        self.mean_in: Dict[int, torch.Tensor] = {}
        self.std_in: Dict[int, torch.Tensor] = {}
        self.mean_tg: Dict[int, torch.Tensor] = {}
        self.std_tg: Dict[int, torch.Tensor] = {}

        if not self.norm_stats_data:
            logger.warning("Normalization prep called but no stats data loaded.")
            self.apply_normalization = False
            return

        missing_layers = set(self.layer_indices)

        try:
            for layer_idx_str, stats in self.norm_stats_data.items():
                layer_idx = int(layer_idx_str)
                if layer_idx not in self.layer_indices:
                    logger.warning(f"Normalization stats contain unknown layer index {layer_idx}. Skipping.")
                    continue

                missing_layers.discard(layer_idx)

                # Inputs
                if "inputs" in stats and "mean" in stats["inputs"] and "std" in stats["inputs"]:
                    self.mean_in[layer_idx] = torch.tensor(
                        stats["inputs"]["mean"],
                        device=self.device,
                        dtype=torch.float32,  # Compute in float32
                    ).unsqueeze(
                        0
                    )  # Add batch dim
                    self.std_in[layer_idx] = (
                        torch.tensor(
                            stats["inputs"]["std"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        + 1e-6
                    ).unsqueeze(
                        0
                    )  # Add batch dim and epsilon
                else:
                    logger.warning(f"Missing input mean/std for layer {layer_idx} in norm stats.")

                # Targets
                if "targets" in stats and "mean" in stats["targets"] and "std" in stats["targets"]:
                    self.mean_tg[layer_idx] = torch.tensor(
                        stats["targets"]["mean"],
                        device=self.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    self.std_tg[layer_idx] = (
                        torch.tensor(
                            stats["targets"]["std"],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        + 1e-6
                    ).unsqueeze(0)
                else:
                    logger.warning(f"Missing target mean/std for layer {layer_idx} in norm stats.")

            if missing_layers:
                logger.warning(f"Normalization stats missing for layers: {sorted(list(missing_layers))}")

            logger.info("Normalization statistics prepared.")
            self.apply_normalization = True  # Confirm application

            if self.apply_normalization:
                log_msg = "Normalization statistics prepared successfully. Example shapes: "
                example_layer = self.layer_indices[0]
                if example_layer in self.mean_in:
                    log_msg += f"mean_in[{example_layer}]: {self.mean_in[example_layer].shape}, "
                if example_layer in self.std_in:
                    log_msg += f"std_in[{example_layer}]: {self.std_in[example_layer].shape}"
                logger.info(log_msg)
            else:
                logger.warning("Normalization statistics FAILED to load or were incomplete. Normalization disabled.")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(
                f"Error processing normalization stats: {e}. Disabling normalization.",
                exc_info=True,
            )
            self.apply_normalization = False
            self.mean_in, self.std_in, self.mean_tg, self.std_tg = {}, {}, {}, {}

    def get_batch(self) -> ActivationBatch:
        """Fetches the next batch based on the manifest sampler."""
        start_time = time.monotonic()
        try:
            idxs = next(self.sampler_iter)
        except StopIteration:
            self.epoch += 1
            logger.info(f"Epoch {self.epoch} finished. Resetting sampler.")
            self.sampler.set_epoch(self.epoch)
            try:
                idxs = next(self.sampler_iter)
            except StopIteration:
                logger.error("Sampler immediately exhausted even after epoch increment...")
                raise StopIteration("Dataset exhausted.")

        fetch_start_time = time.monotonic()
        unique_chunks, inverse_indices = np.unique(idxs[:, 0], return_inverse=True)
        rows_by_chunk: Dict[int, np.ndarray] = {}
        for i, chunk_id in enumerate(unique_chunks):
            rows_by_chunk[chunk_id] = idxs[inverse_indices == i, 1]

        # ---> ADDED CHUNK TRANSITION LOGGING <---
        current_chunk_ids = set(unique_chunks)
        if self.last_processed_chunk_ids != current_chunk_ids:
            if self.last_processed_chunk_ids is None:
                logger.debug(f"Starting fetch. Accessing chunk(s): {sorted(list(current_chunk_ids))}")
            else:
                logger.debug(
                    f"Chunk transition detected. Now accessing chunk(s): {sorted(list(current_chunk_ids))} "
                    f"(Previous was: {sorted(list(self.last_processed_chunk_ids))})"
                )
            self.last_processed_chunk_ids = current_chunk_ids
        # ---> END ADDED CHUNK TRANSITION LOGGING <---

        raw_bytes_by_chunk: Dict[int, bytes] = {}
        fetch_errors = []
        for chunk_id, row_indices_for_chunk in rows_by_chunk.items():
            try:
                # Use a sorted COPY for efficient fetching while preserving the
                # original order so we can restore it later without per‑row loops.
                sorted_rows = np.sort(row_indices_for_chunk)
                raw_bytes_by_chunk[chunk_id] = self._fetch_slice(chunk_id, sorted_rows)
                expected_bytes = len(sorted_rows) * self.bytes_per_row * 2 * self.num_layers
                actual_bytes = len(raw_bytes_by_chunk[chunk_id])
                if actual_bytes != expected_bytes:
                    logger.error(
                        f"Chunk {chunk_id}: Fetched byte size mismatch. Expected {expected_bytes}, got {actual_bytes}"
                    )
                    raise ValueError(f"Incorrect byte size fetched for chunk {chunk_id}")
            except Exception as e:
                logger.error(f"Failed to fetch slice for chunk {chunk_id}: {e}", exc_info=True)
                fetch_errors.append(chunk_id)

        if fetch_errors:
            raise RuntimeError(f"Failed to fetch data for chunks: {fetch_errors}")
        fetch_duration = time.monotonic() - fetch_start_time
        parse_start_time = time.monotonic()

        # Use the list append + torch.cat strategy from the old implementation,
        # which works across devices (CPU, CUDA, MPS).
        layer_inputs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        layer_targets: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for chunk_id, row_indices_original_order in rows_by_chunk.items():
            raw_bytes = raw_bytes_by_chunk[chunk_id]
            slice_tok = len(row_indices_original_order)
            bytes_per_tensor = slice_tok * self.bytes_per_row
            per_layer_bytes_in_slice = bytes_per_tensor * 2

            # Compute permutation to map fetched (sorted) order back to original
            # This is still needed to ensure the concatenated tensors have the
            # rows in the order yielded by the sampler.
            sorted_rows = np.sort(row_indices_original_order)
            reorder_idx = np.searchsorted(sorted_rows, row_indices_original_order)
            # Use torch tensor for indexing GPU tensors directly
            reorder_idx_tensor = torch.as_tensor(reorder_idx, dtype=torch.long)

            # Process layer by layer
            for li in self.layer_indices:
                layer_start_offset = li * per_layer_bytes_in_slice
                inp_start = layer_start_offset
                inp_end = layer_start_offset + bytes_per_tensor
                tgt_start = inp_end
                tgt_end = tgt_start + bytes_per_tensor

                # Create tensors from buffer (no copy yet if memoryview is used)
                #    Reshape to (slice_tok, d_model)
                inp_tensor_slice = torch.frombuffer(memoryview(raw_bytes)[inp_start:inp_end], dtype=self.dtype).reshape(
                    slice_tok, self.d_model
                )
                tgt_tensor_slice = torch.frombuffer(memoryview(raw_bytes)[tgt_start:tgt_end], dtype=self.dtype).reshape(
                    slice_tok, self.d_model
                )

                # Re-order the rows according to the original sampler order
                #    This operation copies data to the target device (CPU or GPU)
                inp_reordered = inp_tensor_slice[reorder_idx_tensor].to(self.device)
                tgt_reordered = tgt_tensor_slice[reorder_idx_tensor].to(self.device)

                # Append the re-ordered slice for this chunk to the list for this layer
                layer_inputs[li].append(inp_reordered)
                layer_targets[li].append(tgt_reordered)

        # Concatenate chunk pieces from lists
        final_batch_inputs: Dict[int, torch.Tensor] = {li: torch.cat(tensors) for li, tensors in layer_inputs.items()}
        final_batch_targets: Dict[int, torch.Tensor] = {li: torch.cat(tensors) for li, tensors in layer_targets.items()}

        # 5. Apply Normalization (if enabled)
        if self.apply_normalization:
            log_stats_this_batch = {}
            for li in self.layer_indices:
                if li == 0 and final_batch_inputs[li].numel() > 0:
                    inp_before = final_batch_inputs[li]
                    log_stats_this_batch["inp_mean_before"] = inp_before.float().mean().item()
                    log_stats_this_batch["inp_std_before"] = inp_before.float().std().item()
                    if li in self.mean_in:
                        log_stats_this_batch["target_mean_in"] = self.mean_in[li].mean().item()
                        log_stats_this_batch["target_std_in"] = self.std_in[li].mean().item()

                if li in self.mean_in and li in self.std_in:
                    final_batch_inputs[li] = (final_batch_inputs[li].float() - self.mean_in[li]) / self.std_in[li]
                    final_batch_inputs[li] = final_batch_inputs[li].to(self.dtype)
                if li in self.mean_tg and li in self.std_tg:
                    final_batch_targets[li] = (final_batch_targets[li].float() - self.mean_tg[li]) / self.std_tg[li]
                    final_batch_targets[li] = final_batch_targets[li].to(self.dtype)

                if li == 0 and final_batch_inputs[li].numel() > 0:
                    inp_after = final_batch_inputs[li]
                    log_stats_this_batch["inp_mean_after"] = inp_after.float().mean().item()
                    log_stats_this_batch["inp_std_after"] = inp_after.float().std().item()

            if log_stats_this_batch:
                logger.debug(f"Normalization Stats (Layer 0): {log_stats_this_batch}")

        parse_duration = time.monotonic() - parse_start_time
        total_duration = time.monotonic() - start_time
        logger.debug(
            f"get_batch completed in {total_duration:.4f}s (fetch: {fetch_duration:.4f}s, parse: {parse_duration:.4f}s)"
        )

        return final_batch_inputs, final_batch_targets

    def state_dict(self) -> Dict[str, Any]:
        """Return minimal state needed to resume iteration."""
        # We only need to save the epoch to reconstruct the sampler state
        return {
            "store_type": self.__class__.__name__,  # Include specific type
            "epoch": self.epoch,
            "seed": self.seed,
            # Sampler state (chunk order, position) is implicitly restored
            # by re-initializing ChunkRowSampler with the saved epoch and seed.
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state created by `state_dict`. Resets sampler to saved epoch."""
        if state_dict.get("store_type") != self.__class__.__name__:
            logger.warning(
                f"Attempting to load state from incompatible store type '{state_dict.get('store_type')}'. Expected '{self.__class__.__name__}'."
            )

        loaded_epoch = int(state_dict.get("epoch", 0))
        loaded_seed = int(state_dict.get("seed", self.seed))

        if loaded_seed != self.seed:
            logger.warning(
                f"Loading state with different seed ({loaded_seed}) than current ({self.seed}). Sampler sequence will differ."
            )
            self.seed = loaded_seed  # Update own seed to match loaded state

        self.epoch = loaded_epoch

        # Re‑create sampler iterator starting from the loaded epoch
        logger.info(f"Resetting sampler to epoch {self.epoch} with seed {self.seed}.")
        self.sampler = ChunkRowSampler(
            chunk_sizes=self.chunk_sizes,
            num_chunks=self.num_chunks,
            batch=self.train_batch_size_tokens,
            seed=self.seed,
            epoch=self.epoch,
            rank=self.rank,
            world=self.world,
        )
        self.sampler_iter = iter(self.sampler)

    # __len__ is already implemented in BaseActivationStore using total_tokens
    # and train_batch_size_tokens, which ManifestActivationStore populates.
    # We can override if a more precise calculation based on the sampler is needed,
    # but the base implementation provides a reasonable estimate.

    # Make the store iterable (standard iterator protocol)
    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()


# --- LRU Cache for HDF5 Files (used by Local variant) ---
# Place it here so LocalActivationStore can use it. Needs h5py.
@lru_cache(maxsize=128)
def _open_h5(path: Path) -> h5py.File:
    """Cached HDF5 file opener."""
    if not path.exists():
        # Raise specific error if file doesn't exist to prevent caching failures
        raise FileNotFoundError(f"HDF5 file not found at: {path}")
    try:
        # 'swmr=True' (Single Writer Multiple Reader) might improve concurrency
        # if chunks are ever written while being read, but typically not needed here.
        return h5py.File(path, "r")
    except OSError as e:
        # Catch potential file corruption errors during open
        logger.error(f"Failed to open HDF5 file {path}: {e}")
        raise  # Re-raise after logging
