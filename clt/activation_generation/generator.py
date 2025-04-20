"""Activation Generator – full version with metadata & norm‑stats

*   Streams activations, writes **row‑shuffled bf16 HDF5** chunks.
*   Builds a global **index.bin** manifest `(chunk, row)` for exactly‑once
    sampling.
*   Optionally uploads chunks in the background.
*   Emits `metadata.json` and `norm_stats.json` compatible with the
    storage server.

Public API is unchanged: create with `ActivationGeneratorConfig`, call
`generate_and_save()`.
"""

from __future__ import annotations

import os
import time
import json
import math
import queue
import random
import shutil
import signal
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple, Any

import torch
import numpy as np
import h5py
from tqdm import tqdm
import requests
from urllib.parse import quote, urljoin

# ––– local imports (keep relative to package root) –––
from clt.nnsight.extractor import ActivationExtractorCLT  # noqa: E402
from clt.config.data_config import ActivationConfig  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ActivationBatch = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]

# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------


def _create_datasets(
    hf: h5py.File,
    layer_ids: List[int],
    rows: int,
    d: int,
    h5py_dtype: str = "float16",
):
    """Row‑chunked bf16 datasets with gzip‑2 compression."""
    for lid in layer_ids:
        g = hf.create_group(f"layer_{lid}")
        for name in ("inputs", "targets"):
            g.create_dataset(
                name,
                shape=(rows, d),
                dtype=h5py_dtype,
                chunks=(1, d),
                compression="gzip",
                compression_opts=2,
            )


def _async_uploader(upload_q: "queue.Queue[Path]", cfg: ActivationConfig):
    dataset_name = os.path.basename(cfg.dataset_path)
    dataset_id = quote(f"{cfg.model_name}/{dataset_name}_{cfg.dataset_split}", safe="")
    # Handle optional remote_server_url
    if cfg.remote_server_url is None:
        logger.error("Remote server URL is not configured, cannot upload.")
        # Drain the queue to prevent deadlock if items were added
        while True:
            try:
                item = upload_q.get_nowait()
                if item is None:
                    upload_q.task_done()
                    break
                # Mark drained items as done too
                logger.warning(f"Draining un-uploadable item: {item.name}")
                upload_q.task_done()
            except queue.Empty:
                break
        return

    base = cfg.remote_server_url.rstrip("/") + "/"
    sess = requests.Session()

    # --> ADDED: Retry/Backoff Configuration <--
    # TODO: Make these configurable via ActivationConfig?
    max_retries_per_chunk = getattr(cfg, "upload_max_retries", 5)  # Default 5 retries
    initial_backoff = getattr(cfg, "upload_initial_backoff", 1.0)  # Default 1 second
    max_backoff = getattr(cfg, "upload_max_backoff", 30.0)  # Default 30 seconds
    # --> END ADDED <--

    while True:
        p = upload_q.get()
        if p is None:
            upload_q.task_done()
            break  # Sentinel value received, terminate thread

        idx = int(p.stem.split("_")[-1])
        url = urljoin(base, f"datasets/{dataset_id}/chunks/{idx}")
        upload_success = False

        # --> ADDED: Retry Loop <--
        for attempt in range(max_retries_per_chunk):
            try:
                print(
                    f"[Uploader Thread Attempt {attempt+1}/{max_retries_per_chunk}] Uploading chunk: {p.name} to {url}"
                )
                with open(p, "rb") as f:
                    files = {"chunk_file": (p.name, f, "application/x-hdf5")}
                    # TODO: Fetch num_tokens and saved_dtype from the chunk file itself or metadata?
                    # For now, using config values, which might be okay if consistent.
                    headers = {
                        "X-Num-Tokens": str(cfg.chunk_token_threshold),
                        "X-Saved-Dtype": cfg.activation_dtype,
                    }
                    r = sess.post(url, files=files, headers=headers, timeout=300)

                    # Check status code for retry logic
                    if r.ok:  # 2xx status codes
                        logger.info(f"Uploaded {p.name} -> {r.status_code} on attempt {attempt + 1}")
                        upload_success = True
                        if cfg.delete_after_upload:
                            try:
                                p.unlink(missing_ok=True)
                                logger.debug(f"Deleted local chunk {p.name} after successful upload.")
                            except OSError as unlink_err:
                                logger.warning(f"Failed to delete chunk {p.name} after upload: {unlink_err}")
                        break  # Exit retry loop on success
                    else:
                        # Non-2xx status codes - decide whether to retry
                        is_retryable = False
                        if r.status_code in {408, 429, 500, 502, 503, 504}:
                            is_retryable = True
                            logger.warning(
                                f"Upload attempt {attempt + 1} for {p.name} failed with retryable status {r.status_code}: {r.text}"
                            )
                        elif r.status_code == 507:
                            logger.error(f"Upload failed for {p.name}: Server out of storage (507). Not retrying.")
                            # Non-retryable error
                        elif 400 <= r.status_code < 500:
                            logger.error(
                                f"Upload failed for {p.name}: Client error {r.status_code}. Not retrying. Response: {r.text}"
                            )
                            # Non-retryable client error
                        else:
                            logger.error(
                                f"Upload attempt {attempt + 1} for {p.name} failed with unexpected status {r.status_code}: {r.text}. Not retrying."
                            )
                        # Unexpected - treat as non-retryable for safety

                        if not is_retryable:
                            break  # Exit retry loop for non-retryable errors
                        # For retryable errors, continue to backoff logic below

            except requests.exceptions.Timeout as e:
                logger.warning(f"Upload attempt {attempt + 1} for {p.name} timed out: {e}")
                is_retryable = True  # Treat timeouts as retryable
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Upload attempt {attempt + 1} for {p.name} failed due to connection error: {e}")
                is_retryable = True  # Treat connection errors as retryable
            except requests.exceptions.RequestException as e:
                # Catch other potential request exceptions (e.g., DNS errors)
                logger.warning(f"Upload attempt {attempt + 1} for {p.name} failed with request exception: {e}")
                is_retryable = True  # Generally treat these as retryable
            except Exception as e:
                # Catch unexpected errors during file open, etc.
                logger.error(f"Unexpected error during upload attempt {attempt + 1} for {p.name}: {e}", exc_info=True)
                is_retryable = False  # Don't retry unexpected code errors
                break  # Exit retry loop

            # --- Backoff logic if retry is needed --- #
            if attempt + 1 < max_retries_per_chunk and is_retryable:
                backoff_time = min(initial_backoff * (2**attempt), max_backoff)
                jitter = backoff_time * 0.1
                sleep_time = backoff_time + random.uniform(-jitter, jitter)
                logger.info(f"Retrying upload of {p.name} in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            elif not is_retryable:
                logger.error(f"Upload failed for {p.name} due to non-retryable error. Stopping attempts.")
                break  # Exit retry loop immediately if error is non-retryable

        # --- After Retry Loop --- #
        if not upload_success:
            logger.error(
                f"Failed to upload chunk {p.name} after {max_retries_per_chunk} attempts. Chunk will remain locally."
            )
            # NOTE: We are NOT re-queuing the item here to prevent potential infinite loops
            # if the error condition persists. The file remains locally.
            # Consider adding logic here later to move failed chunks to a separate directory.

        # --> Moved task_done outside the retry loop <--
        # Ensure task_done is called exactly once per item from the queue,
        # regardless of success, failure, or retries.
        upload_q.task_done()
        # --> END MOVED <--


# ---------------------------------------------------------------------------
# Welford online stats helper
# ---------------------------------------------------------------------------
class _RunningStat:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.M2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        x = x.to(torch.float64)
        cnt = x.shape[0]
        delta = x.mean(0) - self.mean
        new_n = self.n + cnt
        self.mean += delta * (cnt / new_n)
        self.M2 += ((x - self.mean).pow(2)).sum(0)
        self.n = new_n

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        var = self.M2 / max(self.n - 1, 1)
        return self.mean.cpu().numpy().astype("float32"), np.sqrt(var).cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ActivationGenerator:
    def __init__(self, cfg: ActivationConfig, device: torch.device | str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.torch_dtype = getattr(torch, cfg.activation_dtype)
        except AttributeError:
            logger.warning(f"Invalid activation_dtype '{cfg.activation_dtype}' in config. Defaulting to bfloat16.")
            self.torch_dtype = torch.bfloat16
        self.extractor = ActivationExtractorCLT(
            model_name=cfg.model_name,
            mlp_input_module_path_template=cfg.mlp_input_module_path_template,
            mlp_output_module_path_template=cfg.mlp_output_module_path_template,
            device=self.device,
            model_dtype=cfg.model_dtype,
            context_size=cfg.context_size,
            inference_batch_size=cfg.inference_batch_size,
            exclude_special_tokens=cfg.exclude_special_tokens,
            prepend_bos=cfg.prepend_bos,
            nnsight_tracer_kwargs=cfg.nnsight_tracer_kwargs,
            nnsight_invoker_args=cfg.nnsight_invoker_args,
        )
        # Paths
        ds_name = os.path.basename(cfg.dataset_path)
        self.out_dir = Path(cfg.activation_dir) / cfg.model_name / f"{ds_name}_{cfg.dataset_split}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_tmp = self.out_dir / "index.tmp"
        self.manifest_final = self.out_dir / "index.bin"
        # Background uploader
        self.upload_q: "queue.Queue[Path]" = queue.Queue()
        if cfg.remote_server_url:
            self.uploader = threading.Thread(target=_async_uploader, args=(self.upload_q, cfg), daemon=True)
            self.uploader.start()
        else:
            self.uploader = None

        # ------------------------------------------------------------------
        # Storage mode handling
        # ------------------------------------------------------------------

        # Default storage mode is inferred from whether a remote URL is
        # configured. Users can later override via `set_storage_type()`.
        self.storage_type: str = "remote" if cfg.remote_server_url else "local"
        if self.storage_type == "remote" and not cfg.remote_server_url:
            raise ValueError("Storage type is 'remote' but no remote_server_url is configured.")

    # ------------------------------------------------------------------
    def generate_and_save(self):
        cfg = self.cfg
        stream = self.extractor.stream_activations(
            dataset_path=cfg.dataset_path,
            dataset_split=cfg.dataset_split,
            dataset_text_column=cfg.dataset_text_column,
            streaming=cfg.streaming,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            cache_path=cfg.cache_path,
        )

        tgt_tokens = cfg.target_total_tokens
        chunk_tokens = cfg.chunk_token_threshold
        pbar = tqdm(total=tgt_tokens or None, unit="tok", smoothing=0.2)

        # Collect manifest rows in‑memory to avoid pre‑allocation mismatch bugs.
        # Each entry is (chunk_id, local_row).  For 1 M tokens this is only 8 MB.
        manifest_rows: List[np.ndarray] = []

        # Norm‑stat structures
        stats: Dict[int, Dict[str, _RunningStat]] = {}

        g_row = 0
        c_idx = 0
        buf_inp: Dict[int, List[torch.Tensor]] = {}
        buf_tgt: Dict[int, List[torch.Tensor]] = {}
        layer_ids: Optional[List[int]] = None
        d_model = -1
        dtype_str = "unknown"

        for batch_inp, batch_tgt in stream:
            if tgt_tokens and g_row >= tgt_tokens:
                break
            if not batch_inp:
                continue
            if layer_ids is None:
                layer_ids = sorted(batch_inp.keys())
                d_model = batch_inp[layer_ids[0]].shape[-1]
                dtype_str = str(batch_inp[layer_ids[0]].dtype)
                for lid in layer_ids:
                    buf_inp[lid] = []
                    buf_tgt[lid] = []
                    if cfg.compute_norm_stats:
                        stats[lid] = {
                            "inputs": _RunningStat(d_model),
                            "targets": _RunningStat(d_model),
                        }
                logger.info("Layers=%d d_model=%d dtype=%s", len(layer_ids), d_model, dtype_str)

            n_tok = batch_inp[layer_ids[0]].shape[0]
            for lid in layer_ids:
                inp = batch_inp[lid].detach().cpu()
                tgt = batch_tgt[lid].detach().cpu()
                buf_inp[lid].append(inp)
                buf_tgt[lid].append(tgt)
                if cfg.compute_norm_stats:
                    stats[lid]["inputs"].update(inp)
                    stats[lid]["targets"].update(tgt)

            g_row += n_tok
            pbar.update(n_tok)

            # Flush chunk when we've reached the threshold
            cur_rows = sum(t.shape[0] for t in buf_inp[layer_ids[0]])
            if cur_rows >= chunk_tokens:
                # Write *all* accumulated rows so far (variable‑size chunk).
                self._write_chunk(
                    c_idx,
                    buf_inp,
                    buf_tgt,
                    layer_ids,
                    d_model,
                    cur_rows,
                    manifest_rows,
                    g_row - cur_rows,
                )
                c_idx += 1
                # Clear buffers for next chunk
                for lid in layer_ids:
                    buf_inp[lid].clear()
                    buf_tgt[lid].clear()

        # Flush final partial chunk
        if layer_ids and buf_inp[layer_ids[0]]:
            rows = sum(t.shape[0] for t in buf_inp[layer_ids[0]])
            self._write_chunk(
                c_idx,
                buf_inp,
                buf_tgt,
                layer_ids,
                d_model,
                rows,
                manifest_rows,
                g_row - rows,
            )
            c_idx += 1

        # Write manifest to disk (concatenate to keep correct order)
        manifest_arr = np.concatenate(manifest_rows, axis=0)
        manifest_arr.tofile(self.manifest_final)

        # Upload final manifest if remote
        if self.storage_type == "remote" and self.manifest_final.exists():
            try:
                self._upload_binary_file(self.manifest_final, "manifest")
            except Exception as e:
                logger.warning("Failed to upload manifest.bin: %s", e)

        # Write metadata JSON
        meta = {
            "model_name": cfg.model_name,
            "dataset": cfg.dataset_path,
            "split": cfg.dataset_split,
            "num_layers": len(layer_ids or []),
            "d_model": d_model,
            "dtype": cfg.activation_dtype,
            "total_tokens": g_row,
            "chunk_tokens": chunk_tokens,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("metadata.json written")

        # If remote storage – immediately upload metadata so the server
        # registers the dataset before any training jobs start.
        meta_path = self.out_dir / "metadata.json"
        if self.storage_type == "remote" and self.cfg.remote_server_url:
            try:
                self._upload_json(meta_path, "metadata")
                logger.info("metadata.json uploaded to server")
            except Exception as e:
                logger.warning("Failed to upload metadata.json: %s", e)

        # Write norm_stats.json
        if cfg.compute_norm_stats and stats:
            norm: Dict[str, Any] = {}
            for lid in layer_ids:
                m_in, s_in = stats[lid]["inputs"].finalize()
                m_tg, s_tg = stats[lid]["targets"].finalize()
                norm[str(lid)] = {
                    "inputs": {"mean": m_in.tolist(), "std": s_in.tolist()},
                    "targets": {"mean": m_tg.tolist(), "std": s_tg.tolist()},
                }
            with open(self.out_dir / "norm_stats.json", "w") as f:
                json.dump(norm, f)
            logger.info("norm_stats.json written")

            # Upload as well if remote
            norm_path = self.out_dir / "norm_stats.json"
            if self.storage_type == "remote" and self.cfg.remote_server_url:
                try:
                    self._upload_json(norm_path, "norm_stats")
                    logger.info("norm_stats.json uploaded to server")
                except Exception as e:
                    logger.warning("Failed to upload norm_stats.json: %s", e)

        # Finish uploading (only if we are in remote mode)
        if self.storage_type == "remote" and self.uploader:
            self.upload_q.put(None)
            self.upload_q.join()
        logger.info("Finished: %d chunks, %s tokens", c_idx, f"{g_row:,}")

    # ------------------------------------------------------------------
    def _write_chunk(
        self,
        chunk_idx: int,
        buf_inp: Dict[int, List[torch.Tensor]],
        buf_tgt: Dict[int, List[torch.Tensor]],
        layer_ids: List[int],
        d_model: int,
        rows: int,
        manifest_rows: List[np.ndarray],
        offset: int,
    ):
        perm = torch.randperm(rows)
        p = self.out_dir / f"chunk_{chunk_idx}.h5"

        # --- Determine h5py dtype string from torch dtype --- #
        if self.torch_dtype == torch.float32:
            h5py_dtype_str = "float32"
        elif self.torch_dtype == torch.float16:
            h5py_dtype_str = "float16"
        elif self.torch_dtype == torch.bfloat16:
            # h5py doesn't natively support bfloat16, store as uint16
            # Note: Client needs to be aware of this if using bfloat16
            h5py_dtype_str = "uint16"
            logger.warning("Storing bfloat16 as uint16 in HDF5. Ensure client handles conversion.")
        else:
            raise ValueError(f"Unsupported torch_dtype for HDF5: {self.torch_dtype}")
        # ----------------------------------------------------- #

        try:
            with h5py.File(p, "w", libver="latest") as hf:
                _create_datasets(hf, layer_ids, rows, d_model, h5py_dtype=h5py_dtype_str)
                for lid in layer_ids:
                    inp = torch.cat(buf_inp[lid], 0)[perm].to(self.torch_dtype).numpy()
                    tgt = torch.cat(buf_tgt[lid], 0)[perm].to(self.torch_dtype).numpy()
                    # View casting for bfloat16 before saving
                    if h5py_dtype_str == "uint16" and inp.dtype == np.dtype("bfloat16"):
                        inp = inp.view(np.uint16)
                        tgt = tgt.view(np.uint16)

                    hf[f"layer_{lid}/inputs"][:] = inp
                    hf[f"layer_{lid}/targets"][:] = tgt
        except (IOError, OSError) as e:
            logger.error(f"Failed to write HDF5 chunk {p}: {e}", exc_info=True)
            # Attempt to remove potentially corrupted partial file
            try:
                p.unlink(missing_ok=True)
            except OSError:
                logger.warning(f"Failed to remove partial chunk file {p} after write error.")
            # Re-raise to halt generation
            raise RuntimeError(f"Fatal error writing HDF5 chunk {chunk_idx}") from e

        # Append manifest rows for this chunk
        m = np.empty((rows, 2), dtype="<u4")
        m[:, 0] = chunk_idx
        m[:, 1] = np.arange(rows, dtype="<u4")
        manifest_rows.append(m)

        logger.debug("chunk %d written (%d rows)", chunk_idx, rows)
        if self.storage_type == "remote" and self.uploader:
            self.upload_q.put(p)

    # ------------------------------------------------------------------
    def set_storage_type(self, storage_type: str):
        """Explicitly select 'local' or 'remote' storage.

        This mirrors the API of the earlier generator implementation so that
        existing tutorials continue to work. When switching *from* remote to
        local we shut down the background uploader thread to avoid leaking
        resources. When switching *to* remote we lazily start the uploader
        thread if it has not already been created.
        """

        st = storage_type.lower()
        if st not in {"local", "remote"}:
            raise ValueError("storage_type must be 'local' or 'remote'")

        # If nothing changes, we are done.
        if st == self.storage_type:
            return

        # Switching to remote: ensure server URL is configured and uploader is
        # running.
        if st == "remote":
            if self.cfg.remote_server_url is None:
                raise ValueError(
                    "Cannot set storage_type to 'remote' because " "cfg.remote_server_url is not configured."
                )
            if self.uploader is None:
                # Lazily start a new uploader thread
                self.uploader = threading.Thread(
                    target=_async_uploader,
                    args=(self.upload_q, self.cfg),
                    daemon=True,
                )
                self.uploader.start()
        else:  # switching to local
            if self.uploader is not None:
                # Gracefully terminate the uploader thread
                self.upload_q.put(None)
                self.uploader.join()
                self.uploader = None

        self.storage_type = st

    # ------------------------------------------------------------------
    def _upload_json(self, path: Path, endpoint: str):
        """Upload metadata or norm_stats JSON to the server.

        `endpoint` must be either 'metadata' or 'norm_stats'.
        Retries up to 3 times on failure.
        """
        max_retries = 3
        base_delay = 1  # seconds

        if endpoint not in {"metadata", "norm_stats"}:
            raise ValueError("endpoint must be 'metadata' or 'norm_stats'")

        if not self.cfg.remote_server_url:
            raise ValueError("remote_server_url is not configured for upload")

        dataset_name = os.path.basename(self.cfg.dataset_path)
        dataset_id = quote(f"{self.cfg.model_name}/{dataset_name}_{self.cfg.dataset_split}", safe="")
        # Point directly to the root-mounted slice server endpoints
        base = self.cfg.remote_server_url.rstrip("/") + "/"
        url = urljoin(base, f"datasets/{dataset_id}/{endpoint}")

        for attempt in range(max_retries):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                r = requests.post(url, json=data, timeout=60)
                r.raise_for_status()
                logger.info(f"Successfully uploaded {path.name} to {endpoint} endpoint on attempt {attempt + 1}")
                return  # Success
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to upload {path.name} to {endpoint}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Final attempt failed to upload {path.name}. Giving up.")
                    raise RuntimeError(f"Failed to upload {path.name} after {max_retries} attempts") from e
                else:
                    delay = base_delay * (2**attempt)
                    logger.info(f"Retrying upload of {path.name} in {delay:.1f} seconds...")
                    time.sleep(delay)

    # ------------------------------------------------------------------
    def _upload_binary_file(self, path: Path, endpoint: str):
        """Upload a binary file (like index.bin) to the server.

        `endpoint` must be the target path component (e.g., 'manifest').
        Retries up to 3 times on failure.
        """
        max_retries = 3
        base_delay = 2  # seconds, slightly longer for potentially larger files

        if not self.cfg.remote_server_url:
            raise ValueError("remote_server_url is not configured for upload")

        dataset_name = os.path.basename(self.cfg.dataset_path)
        dataset_id = quote(f"{self.cfg.model_name}/{dataset_name}_{self.cfg.dataset_split}", safe="")
        base = self.cfg.remote_server_url.rstrip("/") + "/"  # Point to root
        url = urljoin(base, f"datasets/{dataset_id}/{endpoint}")

        for attempt in range(max_retries):
            try:
                with open(path, "rb") as f:
                    # Use the correct key expected by the server endpoint (e.g., manifest_file)
                    file_key = f"{endpoint}_file"  # Dynamically create key, assuming convention
                    if endpoint == "manifest":  # Explicit mapping if needed
                        file_key = "manifest_file"
                    # Add other endpoint -> key mappings here if necessary

                    files = {file_key: (path.name, f, "application/octet-stream")}
                    r = requests.post(url, files=files, timeout=300)  # Increased timeout
                    r.raise_for_status()
                    logger.info(f"Successfully uploaded {path.name} to {endpoint} endpoint on attempt {attempt + 1}")
                    return  # Success
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to upload {path.name} to {endpoint}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Final attempt failed to upload {path.name}. Giving up.")
                    raise RuntimeError(f"Failed to upload {path.name} after {max_retries} attempts") from e
                else:
                    delay = base_delay * (2**attempt)
                    logger.info(f"Retrying upload of {path.name} in {delay:.1f} seconds...")
                    time.sleep(delay)
            except Exception as e:
                # Catch other potential errors during file handling/request prep
                logger.error(
                    f"Unexpected error during upload attempt {attempt + 1} for {path.name}: {e}", exc_info=True
                )
                # Treat unexpected errors as fatal for now
                raise RuntimeError(f"Unexpected error uploading {path.name}") from e


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg_path = ap.parse_args().config
    with open(cfg_path) as f:
        cfg = ActivationConfig(**yaml.safe_load(f))
    ActivationGenerator(cfg).generate_and_save()
