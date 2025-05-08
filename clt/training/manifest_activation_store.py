from __future__ import annotations

import logging
import time
from collections import defaultdict
import threading
import queue

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
import random  # For jitter
import requests  # <-- Add import for requests

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
    Samples (chunk_id, row_id) pairs for deterministic, sharded iteration.

    Supports two strategies:
    1. 'sequential' (default): Shuffles chunk order each epoch; iterates rows sequentially
       within each chunk (rows are already random due to generation shuffling).
    2. 'random_chunk': Selects a random chunk for each batch from chunks that still
       have unused rows for the current rank in the current epoch. Rows within
       a chunk are yielded in a pre-shuffled order for that epoch.

    Strides by GPU rank so there is no overlap. Yields (batch, 2) numpy arrays
    containing [chunk_id, row_id].
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
        sampling_strategy: str = "sequential",  # Added sampling strategy
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

        # Added: Validate and store sampling strategy
        if sampling_strategy not in ["sequential", "random_chunk"]:
            raise ValueError(
                f"Invalid sampling_strategy: '{sampling_strategy}'. Must be 'sequential' or 'random_chunk'."
            )
        self.sampling_strategy = sampling_strategy

        if not self.chunk_sizes:
            raise ValueError("chunk_sizes cannot be empty")
        if self.batch <= 0:
            raise ValueError("batch size must be positive")
        if not (0 <= self.rank < self.world):
            raise ValueError(f"Invalid rank/world: {rank}/{world}")

        # Initialize generator here to set state for the first epoch
        self._reset_generator()

    def _reset_generator(self):
        """Resets the numpy random generator and internal state for a new epoch."""
        self.rng = np.random.default_rng(self.seed + self.epoch)  # Use instance RNG

        # --- State common to both strategies ---
        # Create chunk_id → row_ids mapping for each chunk FOR THIS RANK
        self.rows_by_chunk_for_epoch: Dict[int, np.ndarray] = {}
        for chunk_id, chunk_size in self.chunk_sizes.items():
            if chunk_id >= self.num_chunks:
                continue  # Skip chunks beyond considered range

            # Get rows assigned to this rank
            chunk_rows = np.arange(chunk_size, dtype=np.uint32)
            rows_for_rank = chunk_rows[self.rank :: self.world]
            # Shuffle rows *within* the chunk for this rank for the epoch
            self.rng.shuffle(rows_for_rank)
            self.rows_by_chunk_for_epoch[chunk_id] = rows_for_rank

        # Count total rows across all chunks for this rank (used for __len__)
        self.total_rows_this_rank = sum(len(rows) for rows in self.rows_by_chunk_for_epoch.values())
        self.total_batches_this_rank = self.total_rows_this_rank // self.batch

        # --- Strategy-specific state reset ---
        if self.sampling_strategy == "sequential":
            # Shuffle chunk order for sequential iteration
            self.chunk_order = self.rng.permutation(
                # Only shuffle chunks that actually have rows for this rank
                [cid for cid, rows in self.rows_by_chunk_for_epoch.items() if len(rows) > 0]
            )
            self.current_chunk_idx_in_order = 0
            self.current_row_offset_in_chunk = 0
        elif self.sampling_strategy == "random_chunk":
            # Track next starting row offset for each chunk independently
            self.next_row_offset_by_chunk = {chunk_id: 0 for chunk_id in self.rows_by_chunk_for_epoch}
            self.batches_yielded_this_epoch = 0

    def __iter__(self):
        # Reset iteration state based on strategy
        if self.sampling_strategy == "sequential":
            self.current_chunk_idx_in_order = 0
            self.current_row_offset_in_chunk = 0
        elif self.sampling_strategy == "random_chunk":
            self.batches_yielded_this_epoch = 0
        return self

    def __next__(self):
        if self.sampling_strategy == "sequential":
            return self._next_sequential()
        elif self.sampling_strategy == "random_chunk":
            return self._next_random_chunk()
        else:
            # Should not happen due to validation in __init__
            raise RuntimeError(f"Internal Error: Unknown sampling strategy '{self.sampling_strategy}'")

    def _next_sequential(self):
        """Get the next batch using the sequential chunk strategy."""
        if self.current_chunk_idx_in_order >= len(self.chunk_order):
            # End of epoch reached
            raise StopIteration

        # Get the actual chunk ID for this step
        current_chunk_id = self.chunk_order[self.current_chunk_idx_in_order]

        # This check should be redundant now as chunk_order only includes valid chunks
        # if current_chunk_id not in self.rows_by_chunk_for_epoch:
        #     # This should ideally not happen if chunk_order is built correctly
        #     self.current_chunk_idx_in_order += 1
        #     self.current_row_offset_in_chunk = 0
        #     return self._next_sequential() # Try next

        # Get the pre-shuffled rows for this chunk assigned to this rank for this epoch
        rows_for_this_chunk = self.rows_by_chunk_for_epoch[current_chunk_id]

        # Determine batch slice start/end within the current chunk
        start_row_offset = self.current_row_offset_in_chunk
        end_row_offset = min(start_row_offset + self.batch, len(rows_for_this_chunk))

        # Check if we have enough rows left in *this chunk* for a full batch
        if (end_row_offset - start_row_offset) < self.batch:
            # Not enough rows left in this chunk for a full batch, move to the next chunk
            self.current_chunk_idx_in_order += 1
            self.current_row_offset_in_chunk = 0
            return self._next_sequential()  # Try next chunk

        # We have a full batch from this chunk
        batch_rows = rows_for_this_chunk[start_row_offset:end_row_offset]

        # Prepare output: (batch, 2) numpy array [chunk_id, row_id]
        batch_output = np.stack(
            [np.full(len(batch_rows), current_chunk_id, dtype=np.uint32), batch_rows],
            axis=1,
        )

        # Update offset for the next potential batch within the current chunk
        self.current_row_offset_in_chunk = end_row_offset

        # If we've exhausted rows in the current chunk, move to the next chunk index for the *next* call
        if self.current_row_offset_in_chunk >= len(rows_for_this_chunk):
            self.current_chunk_idx_in_order += 1
            self.current_row_offset_in_chunk = 0  # Reset row offset for the new chunk

        return batch_output

    def _next_random_chunk(self):
        """Get the next batch using the random chunk strategy."""
        if self.batches_yielded_this_epoch >= self.total_batches_this_rank:
            raise StopIteration  # Epoch complete

        # Find chunks that have enough remaining rows for a full batch for this rank
        available_chunks = [
            chunk_id
            for chunk_id, rows in self.rows_by_chunk_for_epoch.items()
            if self.next_row_offset_by_chunk[chunk_id] + self.batch <= len(rows)
        ]

        if not available_chunks:
            # No single chunk has enough rows left for a full batch.
            # This indicates the end of the epoch based on full batches.
            # Note: Some rows might remain unused if total_rows % batch != 0.
            raise StopIteration

        # Randomly select one of the available chunks
        selected_chunk_id = self.rng.choice(available_chunks)

        # Get the rows and current offset for the selected chunk
        rows_for_selected_chunk = self.rows_by_chunk_for_epoch[selected_chunk_id]
        start_offset = self.next_row_offset_by_chunk[selected_chunk_id]
        end_offset = start_offset + self.batch

        # Extract the batch rows (already shuffled during _reset_generator)
        batch_rows = rows_for_selected_chunk[start_offset:end_offset]

        # Update the offset for the selected chunk for the next time it's picked
        self.next_row_offset_by_chunk[selected_chunk_id] = end_offset

        # Prepare output
        batch_output = np.stack(
            [np.full(len(batch_rows), selected_chunk_id, dtype=np.uint32), batch_rows],
            axis=1,
        )

        # Increment the count of batches yielded this epoch
        self.batches_yielded_this_epoch += 1

        return batch_output

    def __len__(self):
        """Return the total number of *full* batches this rank will process in an epoch."""
        # Calculation remains the same regardless of strategy
        return self.total_batches_this_rank

    def set_epoch(self, epoch: int):
        """Sets the epoch for this sampler, resetting the RNG and internal state."""
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
        sampling_strategy: str = "sequential",  # Added sampling strategy
        normalization_method: str = "none",  # Added normalization method
        prefetch_batches: int = 1,  # Number of batches to prefetch (1 means no prefetching)
    ):
        self.train_batch_size_tokens = train_batch_size_tokens  # From Base
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = 0  # Initial epoch
        # Added: Store sampling strategy
        self.prefetch_batches = max(1, prefetch_batches)  # Ensure at least 1
        self.sampling_strategy = sampling_strategy

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
        # # Log the loaded data immediately for debugging
        # if self.norm_stats_data is None:
        #     logger.debug("Loaded norm_stats_data is None.")
        # else:
        #     logger.debug(f"Loaded norm_stats_data type: {type(self.norm_stats_data)}, Is empty? {not bool(self.norm_stats_data)}")
        #     # Optionally log the keys if it's a dict and not empty, to verify content further
        #     if isinstance(self.norm_stats_data, dict) and self.norm_stats_data:
        #         logger.debug(f"Loaded norm_stats_data keys: {list(self.norm_stats_data.keys())}")

        self.apply_normalization = False
        if normalization_method == "none":
            self.apply_normalization = False
        else:
            self.apply_normalization = bool(self.norm_stats_data)
        # logger.debug(f"Post-decision self.apply_normalization = {self.apply_normalization}") # Log the result

        if self.apply_normalization:
            # logger.debug("Entering block to call _prep_norm...") # Log entry to IF block
            # logger.info(f"Preparing normalization stats. Expecting layers: {self.layer_indices}") # Keep this?
            self._prep_norm()
        else:
            # logger.debug("Skipping _prep_norm because self.apply_normalization is False.") # Log entry to ELSE block
            # Initialize empty dicts with types for linter
            self.mean_in: Dict[int, torch.Tensor] = {}
            self.std_in: Dict[int, torch.Tensor] = {}
            self.mean_tg: Dict[int, torch.Tensor] = {}
            self.std_tg: Dict[int, torch.Tensor] = {}

        # --- Setup Sampler ---
        # Pass the sampling_strategy to the sampler
        self.sampler = ChunkRowSampler(
            chunk_sizes=self.chunk_sizes,
            num_chunks=self.num_chunks,
            batch=self.train_batch_size_tokens,
            seed=self.seed,
            epoch=self.epoch,
            rank=self.rank,
            world=self.world,
            sampling_strategy=self.sampling_strategy,  # Pass the strategy
        )
        self.sampler_iter = iter(self.sampler)

        # --- Precompute byte offsets based on dtype and d_model ---
        self.bytes_per_element = torch.finfo(self.dtype).bits // 8
        self.bytes_per_row = self.d_model * self.bytes_per_element
        # bytes_per_tensor = slice_tok * bytes_per_row (depends on batch size)
        # per_layer_bytes = bytes_per_tensor * 2 (input + target)

        # --- Setup Prefetching (if enabled) ---
        self.prefetch_queue: Optional[queue.Queue] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch: Optional[threading.Event] = None
        self._prefetch_error: Optional[Exception] = None  # To store errors from thread

        if self.prefetch_batches > 1:
            self._initialize_prefetching()

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
                    try:
                        self.mean_in[layer_idx] = torch.tensor(
                            stats["inputs"]["mean"],
                            device=self.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        # Create std tensor, add epsilon
                        std_tensor_in = (
                            torch.tensor(
                                stats["inputs"]["std"],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            + 1e-6
                        )
                    except (ValueError, TypeError) as e:
                        # logger.warning(f"Layer {layer_idx} input mean/std failed tensor conversion: {e}. Disabling normalization.")
                        self.apply_normalization = False
                        break  # Exit loop

                    # Check for non-positive values AFTER adding epsilon
                    if torch.any(std_tensor_in <= 0):
                        logger.warning(
                            f"Layer {layer_idx} input std contains non-positive values after adding epsilon. Disabling normalization."
                        )
                        self.apply_normalization = False
                        break  # Exit the loop over layers if an issue is found
                    self.std_in[layer_idx] = std_tensor_in.unsqueeze(0)  # Add batch dim
                else:
                    # Log which keys might be missing
                    missing_keys_in = []
                    if "inputs" not in stats:
                        missing_keys_in.append("'inputs'")
                    elif "mean" not in stats["inputs"]:
                        missing_keys_in.append("'inputs.mean'")
                    elif "std" not in stats["inputs"]:
                        missing_keys_in.append("'inputs.std'")
                    # logger.warning(f"Missing structure ({', '.join(missing_keys_in)}) for layer {layer_idx} inputs in norm stats. Disabling normalization.")
                    self.apply_normalization = False  # Disable if structure is wrong
                    break  # Exit loop

                # Targets
                if "targets" in stats and "mean" in stats["targets"] and "std" in stats["targets"]:
                    try:
                        self.mean_tg[layer_idx] = torch.tensor(
                            stats["targets"]["mean"],
                            device=self.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        # Create std tensor, add epsilon
                        std_tensor_tg = (
                            torch.tensor(
                                stats["targets"]["std"],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            + 1e-6
                        )
                    except (ValueError, TypeError) as e:
                        # logger.warning(f"Layer {layer_idx} target mean/std failed tensor conversion: {e}. Disabling normalization.")
                        self.apply_normalization = False
                        break  # Exit loop

                    # Check for non-positive values AFTER adding epsilon
                    if torch.any(std_tensor_tg <= 0):
                        logger.warning(
                            f"Layer {layer_idx} target std contains non-positive values after adding epsilon. Disabling normalization."
                        )
                        self.apply_normalization = False
                        break  # Exit loop
                    self.std_tg[layer_idx] = std_tensor_tg.unsqueeze(0)
                else:
                    # Log which keys might be missing
                    missing_keys_tg = []
                    if "targets" not in stats:
                        missing_keys_tg.append("'targets'")
                    elif "mean" not in stats["targets"]:
                        missing_keys_tg.append("'targets.mean'")
                    elif "std" not in stats["targets"]:
                        missing_keys_tg.append("'targets.std'")
                    # logger.warning(f"Missing structure ({', '.join(missing_keys_tg)}) for layer {layer_idx} targets in norm stats. Disabling normalization.")
                    self.apply_normalization = False  # Disable if structure is wrong
                    break  # Exit loop

            # Log state of missing_layers *after* the loop
            # logger.debug(f"_prep_norm loop finished. Final missing_layers: {missing_layers}")

            if not self.apply_normalization:  # Check if loop was broken
                # Clear out potentially partially filled stats if we broke early
                self.mean_in, self.std_in, self.mean_tg, self.std_tg = {}, {}, {}, {}
            elif missing_layers:  # Check if loop completed but layers were missing
                # Add specific logging here
                logger.warning(
                    f"Normalization stats missing for layers: {sorted(list(missing_layers))}. Disabling normalization."
                )
                self.apply_normalization = False  # Explicitly disable if layers are missing

            # Only log success and set True if NO errors occurred AND no layers were missing
            if self.apply_normalization:  # Check if it hasn't been set to False above
                logger.info("Normalization statistics prepared successfully.")
                # Optional: Add shape logging only on success
                # log_msg = "Example shapes: "
                # example_layer = self.layer_indices[0]
                # if example_layer in self.mean_in:
                #     log_msg += f"mean_in[{example_layer}]: {self.mean_in[example_layer].shape}, "
                # if example_layer in self.std_in:
                #     log_msg += f"std_in[{example_layer}]: {self.std_in[example_layer].shape}"
                # logger.info(log_msg)
            # else: # Already logged the reason for failure above (missing layers or exception)
            #    logger.warning("Normalization statistics FAILED to load or were incomplete. Normalization disabled.")

        except (KeyError, ValueError, TypeError) as e:
            # Add specific logging here
            logger.error(
                f"Error processing normalization stats (e.g., invalid format, wrong keys): {e}. Disabling normalization.",
                exc_info=True,  # Include traceback for debugging
            )
            self.apply_normalization = False
            self.mean_in, self.std_in, self.mean_tg, self.std_tg = {}, {}, {}, {}

        # Add final log statement regardless of path taken
        logger.debug(f"_prep_norm finished. Final self.apply_normalization = {self.apply_normalization}")

    def _initialize_prefetching(self):
        """Initializes the prefetching queue, thread, and events."""
        if not self.prefetch_batches > 1:
            return

        logger.info(f"Rank {self.rank}: Initializing prefetching with queue size {self.prefetch_batches - 1}")
        # Queue holds pre-fetched batches. Size is num_to_prefetch - 1 because
        # one batch is always being processed by the main thread.
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_batches - 1)
        self._stop_prefetch = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            name=f"PrefetchThread-Rank{self.rank}",
            daemon=True,  # Allow program exit even if thread is running
        )
        self._prefetch_thread.start()

    def _prefetch_loop(self):
        """Target function for the prefetch thread."""
        logger.debug(f"Prefetch thread started for rank {self.rank}.")
        try:
            while not self._stop_prefetch.is_set():  # type: ignore[union-attr]
                try:
                    # Get next indices from the shared sampler iterator
                    # Note: The main thread __next__ will handle epoch bumps if needed
                    # when it consumes StopIteration from the queue.
                    # This thread focuses solely on filling the queue.
                    idxs = next(self.sampler_iter)

                    # Log time taken for fetch/parse
                    fetch_parse_start_time = time.monotonic()
                    # Fetch and parse the batch using the refactored logic
                    batch = self._fetch_and_parse_batch(idxs)
                    fetch_parse_duration = time.monotonic() - fetch_parse_start_time
                    logger.debug(f"Rank {self.rank}: _fetch_and_parse_batch took {fetch_parse_duration:.4f}s")

                    # Put the fetched batch onto the queue, blocking if full.
                    # Use a timeout to periodically check the stop signal.
                    while not self._stop_prefetch.is_set():  # type: ignore[union-attr]
                        try:
                            self.prefetch_queue.put(batch, timeout=0.5)  # type: ignore[union-attr]
                            logger.debug(f"Rank {self.rank}: Prefetch thread put batch onto queue (current qsize={self.prefetch_queue.qsize()})")  # type: ignore[union-attr]
                            break  # Successfully put batch
                        except queue.Full:
                            continue  # Queue is full, loop and check stop signal again

                except StopIteration:
                    logger.info(f"Rank {self.rank}: Prefetch thread detected end of sampler iteration.")
                    # Signal end of data by putting None onto the queue
                    self.prefetch_queue.put(None, block=True)  # type: ignore[union-attr] # Block until space
                    break  # Exit loop, epoch finished

                except Exception as e:
                    # Log error and store it for the main thread
                    logger.error(f"Rank {self.rank}: Error in prefetch loop: {e}", exc_info=True)
                    self._prefetch_error = e  # Store the exception
                    # Signal error by putting the exception itself onto the queue? Or None?
                    # Putting None signals end-of-data, let's put the exception
                    self.prefetch_queue.put(e, block=True)  # type: ignore[union-attr]
                    break  # Exit loop on error

        except Exception as e:
            # Catch errors during thread setup/loop entry
            logger.error(f"Rank {self.rank}: Unhandled exception in _prefetch_loop setup: {e}", exc_info=True)
            self._prefetch_error = e
            # Attempt to signal error via queue if possible
            if self.prefetch_queue:
                try:
                    self.prefetch_queue.put(e, block=False)
                except queue.Full:
                    logger.error("Prefetch queue full, cannot signal error.")

        finally:
            logger.debug(f"Prefetch thread stopping for rank {self.rank}.")

    def _fetch_and_parse_batch(self, idxs: np.ndarray) -> ActivationBatch:
        """Fetches raw bytes for the given indices and parses into tensors."""
        # --- Existing Fetch/Parse Logic from get_batch ---
        fetch_start_time = time.monotonic()
        unique_chunks, inverse_indices = np.unique(idxs[:, 0], return_inverse=True)
        # ---> ADDED: Retry config moved here <---
        max_fetch_retries = 3
        fetch_initial_backoff = 0.5
        fetch_max_backoff = 10.0
        rows_by_chunk: Dict[int, np.ndarray] = {}
        for i, chunk_id in enumerate(unique_chunks):
            rows_by_chunk[chunk_id] = idxs[inverse_indices == i, 1]

        raw_bytes_by_chunk: Dict[int, bytes] = {}
        fetch_errors = []
        for chunk_id, row_indices_for_chunk in rows_by_chunk.items():
            # --> ADDED: Retry Loop for _fetch_slice <--
            chunk_fetch_success = False
            for attempt in range(max_fetch_retries):
                try:
                    # Use a sorted COPY for efficient fetching
                    sorted_rows = np.sort(row_indices_for_chunk)
                    fetched_bytes = self._fetch_slice(chunk_id, sorted_rows)
                    raw_bytes_by_chunk[chunk_id] = fetched_bytes  # Store fetched bytes

                    # Check byte size immediately after fetch
                    expected_bytes = len(sorted_rows) * self.bytes_per_row * 2 * self.num_layers
                    actual_bytes = len(fetched_bytes)
                    if actual_bytes != expected_bytes:
                        # Raise specific error for size mismatch
                        raise ValueError(
                            f"Incorrect byte size fetched for chunk {chunk_id}. Expected {expected_bytes}, got {actual_bytes}"
                        )

                    # If fetch and size check successful
                    chunk_fetch_success = True
                    logger.debug(f"Successfully fetched chunk {chunk_id} slice on attempt {attempt + 1}")
                    break  # Exit retry loop for this chunk

                except (ValueError, RuntimeError, requests.exceptions.RequestException) as e:
                    # Catch fetch errors (RuntimeError from _fetch_slice, RequestException, ValueError for size mismatch)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_fetch_retries} failed to fetch/validate slice for chunk {chunk_id}: {e}"
                    )
                    if attempt + 1 == max_fetch_retries:
                        logger.error(f"Final attempt failed for chunk {chunk_id}. Adding to fetch_errors.")
                        fetch_errors.append(chunk_id)
                        # Break inner loop, error recorded
                        break
                    else:
                        backoff_time = min(fetch_initial_backoff * (2**attempt), fetch_max_backoff)
                        jitter = backoff_time * 0.1
                        sleep_time = backoff_time + random.uniform(-jitter, jitter)
                        logger.info(f"Retrying fetch for chunk {chunk_id} in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                except Exception as e:
                    # Catch unexpected errors
                    logger.error(
                        f"Unexpected error fetching slice for chunk {chunk_id} on attempt {attempt + 1}: {e}",
                        exc_info=True,
                    )
                    fetch_errors.append(chunk_id)
                    # Treat unexpected errors as fatal for this chunk fetch
                    break  # Exit retry loop for this chunk
            # --> END ADDED: Retry Loop <--

            # If a chunk failed all retries, we stop processing this batch
            if not chunk_fetch_success and chunk_id in fetch_errors:
                logger.error(f"Stopping batch processing because chunk {chunk_id} failed all fetch attempts.")
                break  # Exit the outer loop over chunks

        # If any chunk failed permanently after retries, raise an error for the whole batch
        if fetch_errors:
            # ---> MODIFIED: Raise RuntimeError again <---
            failed_chunks_str = ", ".join(map(str, sorted(fetch_errors)))
            logger.error(
                f"Permanently failed to fetch data for chunk(s): {failed_chunks_str} after {max_fetch_retries} retries."
            )
            # Raise runtime error to be caught by __next__
            raise RuntimeError(f"Failed to fetch data for chunk(s): {sorted(fetch_errors)} after retries.")
            # ---> END MODIFIED <---

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
        total_duration = time.monotonic() - fetch_start_time
        logger.debug(
            f"get_batch completed in {total_duration:.4f}s (fetch: {fetch_duration:.4f}s, parse: {parse_duration:.4f}s)"
        )

        return final_batch_inputs, final_batch_targets

    # Add concrete get_batch to satisfy BaseActivationStore abstract method
    def get_batch(self) -> ActivationBatch:
        """Compatibility method to satisfy BaseActivationStore abstract requirement."""
        try:
            # Delegate to the __next__ method which handles prefetching queue or direct call
            return self.__next__()
        except StopIteration:
            # Ensure StopIteration is propagated correctly if __next__ raises it
            raise

    def state_dict(self) -> Dict[str, Any]:
        """Return minimal state needed to resume iteration."""
        # We only need to save the epoch to reconstruct the sampler state
        return {
            "store_type": self.__class__.__name__,  # Include specific type
            "epoch": self.epoch,
            "seed": self.seed,
            "sampling_strategy": self.sampling_strategy,
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
        # --> Added: Load sampling strategy, default to sequential if missing <--
        loaded_strategy = state_dict.get("sampling_strategy", "sequential")

        if loaded_seed != self.seed:
            logger.warning(
                f"Loading state with different seed ({loaded_seed}) than current ({self.seed}). Sampler sequence will differ."
            )
            self.seed = loaded_seed  # Update own seed to match loaded state

        self.epoch = loaded_epoch
        # --> Added: Update own strategy to match loaded state <--
        # Check if the loaded strategy is different and log a warning maybe?
        if loaded_strategy != self.sampling_strategy:
            logger.warning(
                f"Loading state with different sampling strategy ('{loaded_strategy}') than current ('{self.sampling_strategy}'). Using loaded strategy."
            )
            self.sampling_strategy = loaded_strategy

        # Re‑create sampler iterator starting from the loaded epoch, seed, and strategy
        logger.info(
            f"Resetting sampler to epoch {self.epoch} with seed {self.seed} and strategy '{self.sampling_strategy}'."
        )
        self.sampler = ChunkRowSampler(
            chunk_sizes=self.chunk_sizes,
            num_chunks=self.num_chunks,
            batch=self.train_batch_size_tokens,
            seed=self.seed,
            epoch=self.epoch,
            rank=self.rank,
            world=self.world,
            sampling_strategy=self.sampling_strategy,  # Use loaded/updated strategy
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
        # --> MODIFIED: Implement skip logic <--
        try:
            # If prefetching, get from queue, else call get_batch()
            if self.prefetch_queue is not None:
                # Block until an item is available from the prefetch thread
                item = self.prefetch_queue.get(block=True)
                if isinstance(item, Exception):  # Check if thread sent an error
                    logger.error(f"Rank {self.rank}: Error received from prefetch thread: {item}")
                    raise item  # Re-raise the error from the thread
                elif item is None:  # Check for end-of-data sentinel
                    raise StopIteration("Prefetch thread signalled end of data.")
                else:
                    logger.debug(f"Rank {self.rank}: Retrieved batch from prefetch queue (previous qsize approx: {self.prefetch_queue.qsize() + 1})")  # type: ignore[union-attr]
                    return item  # Return the prefetched batch
            else:
                # Get indices and call fetch/parse directly
                idxs = next(self.sampler_iter)
                return self._fetch_and_parse_batch(idxs)
        except RuntimeError as e:
            # This exception is raised by get_batch() if fetching fails permanently
            logger.warning(f"Batch fetch failed ({e}), attempting to skip and fetch the next one...")
            # Skip logic needs adjustment for prefetching - how to tell prefetch thread to skip?
            # For now, let's keep the skip logic simpler and assume it works mostly for non-prefetch
            # or handles errors raised *after* getting from the queue.
            try:
                # Try getting the *subsequent* batch
                idxs = next(self.sampler_iter)  # Need to get new indices
                next_batch = self._fetch_and_parse_batch(idxs)
                logger.info("Successfully fetched subsequent batch after skipping failed one.")
                return next_batch
            except RuntimeError as e2:
                logger.error(f"Failed to fetch subsequent batch after skip ({e2}). Stopping iteration.")
                raise StopIteration("Consecutive batch fetches failed.") from e2
            except StopIteration:
                # If the store was exhausted when trying to get the next batch
                logger.info("Store exhausted while trying to fetch subsequent batch after skip.")
                raise  # Re-raise StopIteration
        # Note: StopIteration from the initial self.get_batch() call propagates normally
        # --> END MODIFIED <--

    def close(self):
        """Shuts down the prefetch thread if it's running."""
        logger.info(f"Rank {self.rank}: Closing activation store...")
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.debug(f"Rank {self.rank}: Signalling prefetch thread to stop.")
            self._stop_prefetch.set()  # type: ignore[union-attr]

            # Put a sentinel value to potentially unblock the thread from queue.put()
            # Or unblock the main thread if it's waiting on queue.get()
            if self.prefetch_queue:
                try:
                    self.prefetch_queue.put(None, block=False)
                except queue.Full:
                    pass

            self._prefetch_thread.join(timeout=5.0)  # Wait for thread to finish
            logger.debug(f"Rank {self.rank}: Prefetch thread joined.")


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
