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

DIR = Path('/scratch/general/vast/u1472283/crosslayer-coding/data/activations/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_float32_1000000toks_1000chunks')

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
        # Only rank 0 needs the extractor/stream; other ranks just receive tensors.
        if rank == 0:
            assert self.extractor is not None, "Rank 0 must receive an activation_extractor."
            self.stream = self.extractor.stream_activations(
                dataset_path=cfg.dataset_path,
                dataset_split=cfg.dataset_split,
                dataset_text_column=cfg.dataset_text_column,
                dataset_skip=cfg.dataset_skip,
                streaming=cfg.streaming,
                dataset_trust_remote_code=cfg.dataset_trust_remote_code,
                cache_path=cfg.cache_path,
                show_pbar=False
            )
        else:
            self.stream = None

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

        # -------------------------
        # Rank 0: manage CPU-side stream buffer (remote logic)
        # -------------------------
        if self.rank == 0:
            assert self.stream is not None, "Rank 0 must own the activation stream."

            def need_to_replenish_buffer():
                if getattr(self, "inp", None) is None:
                    return True
                # pick any key for shape checks
                any_li = next(iter(self.inp.keys()))
                return (self.idx + batch_size > self.inp[any_li].shape[0])

            def replenish_buffer():
                # debug only
                if getattr(self, "inp", None) is None:
                    logger.debug("replenish buffer (first fill)")
                    current_rem = 0
                else:
                    any_li = next(iter(self.inp.keys()))
                    current_rem = self.inp[any_li].shape[0] - self.idx
                    logger.debug(f"replenish buffer, buffer size {current_rem}, less than batch size {batch_size}")

                # pull a fresh chunk from the stream (likely CPU tensors)
                new_inp, new_tgt = next(self.stream)
                any_new = next(iter(new_inp.keys()))
                new_stream_size = new_inp[any_new].shape[0]

                # cast to the activation dtype on CPU first (keeps VRAM down until broadcast)
                new_inp = {li: x.to(self.act_dtype) for li, x in new_inp.items()}
                new_tgt = {li: x.to(self.act_dtype) for li, x in new_tgt.items()}

                if getattr(self, "inp", None) is None:
                    # first fill: take the whole new chunk
                    self.inp = new_inp
                    self.tgt = new_tgt
                else:
                    # concatenate remaining tail of old buffer with the new chunk
                    # (all on CPU so far)
                    for li in new_inp.keys():
                        self.inp[li] = torch.cat((self.inp[li][self.idx:], new_inp[li]), dim=0)
                        self.tgt[li] = torch.cat((self.tgt[li][self.idx:], new_tgt[li]), dim=0)
                self.idx = 0

                any_li2 = next(iter(self.inp.keys()))
                new_buffer_size = self.inp[any_li2].shape[0]
                logger.debug(f"new buffer size is {current_rem} (old rem) + {new_stream_size} (new) = {new_buffer_size}")

            # ensure buffer has at least one batch
            while need_to_replenish_buffer():
                replenish_buffer()

            # slice a batch (still on CPU)
            start = self.idx
            end   = self.idx + batch_size
            any_li = next(iter(self.inp.keys()))
            logger.debug(f'buffer size {self.inp[any_li].shape[0]}, batch size {batch_size}, retrieving idxs {start}-{end}')

            batch_inp = {li: self.inp[li][start:end] for li in self.inp.keys()}
            batch_tgt = {li: self.tgt[li][start:end] for li in self.tgt.keys()}

            # advance the buffer index for next call
            self.idx = end

            # infer d_model (feature dim) once from the sliced batch
            if self.d_model is None:
                self.d_model = next(iter(batch_inp.values())).shape[1]

            # broadcast d_model as a CUDA tensor for robustness with NCCL
            d_model_t = torch.tensor([self.d_model], device='cuda', dtype=torch.int64)
            dist.broadcast(d_model_t, src=0, group=group)
            self.d_model = int(d_model_t.item())

            # ensure the broadcast sources live on CUDA and match dtype
            batch_inp = {li: x.to(device='cuda', dtype=self.act_dtype) for li, x in batch_inp.items()}
            batch_tgt = {li: x.to(device='cuda', dtype=self.act_dtype) for li, x in batch_tgt.items()}

            # broadcast per-layer tensors
            for li in self.layer_indices:
                dist.broadcast(batch_inp[li], src=0, group=group)
                dist.broadcast(batch_tgt[li], src=0, group=group)

            inps, tgts = batch_inp, batch_tgt

        # -------------------------
        # Non-zero ranks: receive shapes & tensors
        # -------------------------
        else:
            # receive d_model
            d_model_t = torch.zeros(1, device='cuda', dtype=torch.int64)
            dist.broadcast(d_model_t, src=0, group=group)
            self.d_model = int(d_model_t.item())

            # pre-allocate CUDA buffers for this mini-batch
            inps = {li: torch.empty((batch_size, self.d_model), dtype=self.act_dtype, device='cuda')
                    for li in self.layer_indices}
            tgts = {li: torch.empty((batch_size, self.d_model), dtype=self.act_dtype, device='cuda')
                    for li in self.layer_indices}

            # receive the layer tensors
            for li in self.layer_indices:
                dist.broadcast(inps[li], src=0, group=group)
                dist.broadcast(tgts[li], src=0, group=group)

        # Optionally move off CUDA if your trainer/device expects something else
        if self.device is not None and str(self.device) != 'cuda':
            inps = {li: x.to(self.device) for li, x in inps.items()}
            tgts = {li: x.to(self.device) for li, x in tgts.items()}

        any_li = self.layer_indices[0]
        logger.debug(f'after broadcast inps[{any_li}].shape={inps[any_li].shape} tgts[{any_li}].shape={tgts[any_li].shape}')
        return inps, tgts


