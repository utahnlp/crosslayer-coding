from __future__ import annotations

from functools import lru_cache
import logging
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.distributed as dist

from .manifest_activation_store import ManifestActivationStore
from ...activation_generation.generator import ActivationConfig, ActivationGenerator
from ...nnsight.extractor import ActivationExtractorCLT

logger = logging.getLogger(__name__)

DIR = '/uufs/chpc.utah.edu/common/home/u1472283/scr/crosslayer-coding/data/activations/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_float32_1000000toks_1000chunks'
DIR = '/scratch/general/vast/u1472283/crosslayer-coding/data/activations/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_float32_1000000toks_1000chunks'
DIR = Path(DIR)

class StreamingActivationStore(ManifestActivationStore):

    def __init__(
        self,
        activation_cfg: ActivationConfig,
        activation_extractor: ActivationExtractorCLT,
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
        
        self.extractor = activation_extractor
        self.activation_cfg = activation_cfg
        cfg = activation_cfg
        self.stream = self.extractor.stream_activations(
            dataset_path=cfg.dataset_path,
            dataset_split=cfg.dataset_split,
            dataset_text_column=cfg.dataset_text_column,
            dataset_skip=cfg.dataset_skip,
            streaming=cfg.streaming,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            cache_path=cfg.cache_path,
        )

        # buffer
        self.idx = 0
        self.inp = None
        self.tgt = None
        
        if cfg.activation_dtype == 'float32':
            self.act_dtype = torch.float32
        elif cfg.activation_dtype == 'float16':
            self.act_dtype = torch.float16
        elif cfg.activation_dtype == 'bfloat16':
            self.act_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported activation_dtype: {cfg.activation_dtype}")

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
            "StreamingActivationStore initialized for dataset at %s "
            "(Rank %d/%d, Seed %d, Batch %d, Device %s, Dtype %s, Strategy '%s')",
            DIR,
            self.rank,
            self.world,
            self.seed,
            self.train_batch_size_tokens,
            self.device,
            self.dtype,
            self.sampling_strategy,
        )
        logger.info(f"Found {self.num_chunks} chunks based on manifest.")

        self.layer_indices = list(range(self.extractor.num_layers))
        self.d_model = getattr(self.extractor, "d_model", None)
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype


    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        path = DIR / "metadata.json"
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
        path = DIR / "index.bin"
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
        path = DIR / "norm_stats.json"
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
        raise Exception('this shouldn\'t be called')

    def __next__(self):
        # Choose the comm group (WORLD if none was provided)
        group = getattr(self, "group", None)
        if group is None:
            group = dist.group.WORLD

        batch_size = self.train_batch_size_tokens
        def need_to_replenish_buffer():
            return (self.inp is None) or (self.idx + batch_size > self.inp[0].shape[0])

        def replenish_buffer():
            buffer_size = 0
            if self.inp is None:
                logger.info('replenish buffer')
            else:
                buffer_size = self.inp[0].shape[0] - self.idx
                logger.info(f'replenish buffer, buffer size {buffer_size}, less than batch size {batch_size}')

            # get new batch from stream
            new_inp, new_tgt = next(self.stream)
            new_stream_size = new_inp[0].shape[0]
            
            # concat new batch to remaining buffer, reset index
            if self.inp is None: # first iteration
                self.inp = {li: x.to(self.act_dtype) for li, x in new_inp.items()}
                self.tgt = {li: x.to(self.act_dtype) for li, x in new_tgt.items()}
            else: # all other iterations, inp maps layer idx to tensor
                for li in new_inp.keys():
                    self.inp[li] = torch.cat((self.inp[li][self.idx:], new_inp[li].to(self.act_dtype)), dim=0)
                    self.tgt[li] = torch.cat((self.tgt[li][self.idx:], new_tgt[li].to(self.act_dtype)), dim=0)
            self.idx = 0

            new_buffer_size = self.inp[0].shape[0]
            logger.info(f'new buffer size is {buffer_size} (old) + {new_stream_size} (new) = {new_buffer_size}')


        # replenish buffer until large enough to retrieve next batch
        while need_to_replenish_buffer():
            replenish_buffer()
        
        # set start and end indices
        start = self.idx
        end = self.idx + batch_size
        logger.info(f'buffer size {self.inp[0].shape[0]}, batch size {batch_size}, retrieving idxs {start}-{end}')
        
        # get batch from buffer
        batch_inp = {}
        batch_tgt = {}
        for li in self.inp.keys():
            batch_inp[li] = self.inp[li][start:end]
            batch_tgt[li] = self.tgt[li][start:end]
        
        # update index for next iteration
        self.idx = self.idx + batch_size

        return batch_inp, batch_tgt

