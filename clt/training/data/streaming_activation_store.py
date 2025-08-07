from __future__ import annotations

from functools import lru_cache
import logging
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch


from .manifest_activation_store import ManifestActivationStore
from ...activation_generation.generator import ActivationConfig, ActivationGenerator
from ...nnsight.extractor import ActivationExtractorCLT

logger = logging.getLogger(__name__)

DIR = '/uufs/chpc.utah.edu/common/home/u1472283/scr/crosslayer-coding/data/activations/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_float32_1000000toks_1000chunks'
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
        self.idx = 0
        # self.inps, self.tgts = next(self.stream)

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
        batch_size = self.train_batch_size_tokens

        logger.critical('HELLO')
        logger.info(f'{batch_size=}')

        # 1 easy
        inps, tgts = next(self.stream)
        logger.info(f'before slicing {inps[0].shape=} {tgts[0].shape=}')
        inps = {li: acts[:batch_size, :] for li, acts in inps.items()}
        tgts = {li: acts[:batch_size, :] for li, acts in tgts.items()}
        logger.info(f'after slicing {inps[0].shape=} {tgts[0].shape=}')
        return inps, tgts

        # 2 
        # if self.idx + batch_size > len(self.activations):
        #     inp, tgt = next(self.stream)
        #     self.activations = next(self.stream)
        # return self.activations[self.idx: self.idx + batch_size]

        # 3
        # if self.idx + batch_size > len(self.activations):
        #     # get more activations
        #     inp, tgt = next(self.stream)
        #     self.activations = torch.concatenate(self.activations[self.idx:], )
        #     pass


        # self.idx += batch_size
        # return self.activations[self.idx: self.idx + batch_size]

