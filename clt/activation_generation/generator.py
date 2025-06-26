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
import json
import queue
import random
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, DefaultDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import h5py
from tqdm import tqdm
import requests
from urllib.parse import quote, urljoin

# ––– local imports (keep relative to package root) –––
from clt.nnsight.extractor import ActivationExtractorCLT  # noqa: E402
from clt.config.data_config import ActivationConfig  # noqa: E402

# --- Profiling Imports ---
import time  # Keep this one
from contextlib import contextmanager
from collections import defaultdict
import psutil

# Local application imports
# from clt.training.utils import torch_bfloat16_to_numpy_uint16 # Removed unused import

try:
    import GPUtil
except ImportError:
    GPUtil = None  # type: ignore
# --- End Profiling Imports ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ActivationBatch = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


# --- Performance Profiler Class ---
class PerformanceProfiler:
    def __init__(self, chunk_tokens_threshold: int = 1_000_000):
        self.timings: DefaultDict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.chunk_tokens_threshold = chunk_tokens_threshold
        self.system_metrics_log: List[Dict[str, Any]] = []
        self.layer_ids_ref: Optional[List[int]] = None
        self.total_tokens_processed_for_batch_profiling = 0
        self.batch_processing_total_calls = 0

    def set_layer_ids_ref(self, layer_ids: List[int]):
        self.layer_ids_ref = layer_ids

    @contextmanager
    def measure(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        start_mem_vm = psutil.virtual_memory().used
        start_mem_rss = psutil.Process(os.getpid()).memory_info().rss

        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        end_mem_vm = psutil.virtual_memory().used
        end_mem_rss = psutil.Process(os.getpid()).memory_info().rss

        self.timings[name].append(elapsed)
        self.memory_snapshots.append(
            {
                "name": name,
                "timestamp": time.time(),
                "duration_s": elapsed,
                "vm_delta_bytes": end_mem_vm - start_mem_vm,
                "vm_total_bytes": end_mem_vm,
                "rss_delta_bytes": end_mem_rss - start_mem_rss,
                "rss_total_bytes": end_mem_rss,
            }
        )

    def log_system_metrics(self, interval_name: str = "interval"):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking

        # Memory usage
        mem = psutil.virtual_memory()

        # Disk I/O (cumulative, consider diffing for rates)
        disk_io = psutil.disk_io_counters()

        gpu_util = 0.0  # Changed to float for consistency
        gpu_memory_percent = 0.0  # Changed to float
        gpu_memory_used_mib = 0.0  # Changed to float
        gpu_memory_total_mib = 0.0  # Changed to float

        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assuming single GPU, or log for all
                    gpu_util = float(gpu.load * 100)
                    gpu_memory_percent = float(gpu.memoryUtil * 100)
                    gpu_memory_used_mib = float(gpu.memoryUsed)
                    gpu_memory_total_mib = float(gpu.memoryTotal)
            except Exception as e:
                logger.debug(f"Could not get GPU stats: {e}")

        metrics = {
            "interval_name": interval_name,
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": mem.percent,
            "memory_used_gb": mem.used / (1024**3),
            "memory_available_gb": mem.available / (1024**3),
            "disk_read_gb": disk_io.read_bytes / (1024**3) if disk_io else 0,
            "disk_write_gb": disk_io.write_bytes / (1024**3) if disk_io else 0,
            "gpu_util_percent": gpu_util,
            "gpu_memory_percent": gpu_memory_percent,
            "gpu_memory_used_mib": gpu_memory_used_mib,
            "gpu_memory_total_mib": gpu_memory_total_mib,
        }
        self.system_metrics_log.append(metrics)
        return metrics

    def report(self, top_n_ops: Optional[int] = 20):
        logger.info("\n=== Performance Report ===")
        # Sort by total time descending for timings
        # Filter out operations with zero total time before sorting and slicing
        valid_timings = {name: times for name, times in self.timings.items() if sum(times) > 0}
        sorted_timings = sorted(valid_timings.items(), key=lambda item: sum(item[1]), reverse=True)

        if top_n_ops is not None and top_n_ops > 0:
            logger.info(f"--- Showing Top {top_n_ops} Timed Operations (by total time) ---")
            timings_to_show = sorted_timings[:top_n_ops]
        else:
            logger.info("--- Showing All Timed Operations (by total time) ---")
            timings_to_show = sorted_timings

        for name, times in timings_to_show:
            if not times:  # Should be redundant due to pre-filtering but safe
                continue
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            min_time = min(times)
            max_time = max(times)

            logger.info(f"\n--- Operation: {name} ---")
            logger.info(f"  Count: {len(times)}")
            logger.info(f"  Total time: {total_time:.3f}s")
            logger.info(f"  Avg time: {avg_time:.4f}s")
            logger.info(f"  Min time: {min_time:.4f}s")
            logger.info(f"  Max time: {max_time:.4f}s")

            if "chunk_write_total_idx" in name:  # New unique name per chunk
                logger.info(
                    f"  Avg ms/k-tok (for this chunk): {avg_time / self.chunk_tokens_threshold * 1000 * 1000:.2f}"
                )
            elif (
                name == "batch_processing_total"
                and self.batch_processing_total_calls > 0
                and self.total_tokens_processed_for_batch_profiling > 0
            ):
                avg_tok_per_batch_call = (
                    self.total_tokens_processed_for_batch_profiling / self.batch_processing_total_calls
                )
                if avg_tok_per_batch_call > 0:
                    logger.info(
                        f"  Avg ms/k-tok (estimated for batch_processing_total): {avg_time / avg_tok_per_batch_call * 1000 * 1000:.2f}"
                    )

        logger.info("\n=== Memory Snapshots (showing top 10 by RSS delta) ===")
        interesting_mem_snapshots = sorted(
            self.memory_snapshots, key=lambda x: abs(x["rss_delta_bytes"]), reverse=True
        )[:10]
        for snap in interesting_mem_snapshots:
            logger.info(
                f"  {snap['name']} (took {snap['duration_s']:.3f}s): Total RSS {snap['rss_total_bytes'] / (1024**3):.3f} GB (ΔRSS {snap['rss_delta_bytes'] / (1024**3):.3f} GB)"
            )

        logger.info("\n=== System Metrics Log (sample) ===")
        for i, metrics in enumerate(self.system_metrics_log[:5]):  # Print first 5 samples
            logger.info(
                f"  Sample {i} ({metrics['interval_name']}): CPU {metrics['cpu_percent']:.1f}%, Mem {metrics['memory_percent']:.1f}%, GPU {metrics['gpu_util_percent']:.1f}% (Mem {metrics['gpu_memory_percent']:.1f}%)"
            )
        if len(self.system_metrics_log) > 5:
            logger.info("  ...")
            if self.system_metrics_log:  # Check if not empty before accessing last element
                metrics = self.system_metrics_log[-1]
                logger.info(
                    f"  Sample End ({metrics['interval_name']}): CPU {metrics['cpu_percent']:.1f}%, Mem {metrics['memory_percent']:.1f}%, GPU {metrics['gpu_util_percent']:.1f}% (Mem {metrics['gpu_memory_percent']:.1f}%)"
                )


# --- End Performance Profiler Class ---

# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------


def _create_datasets(
    hf: h5py.File,
    layer_ids: List[int],
    rows: int,
    d: int,
    h5py_dtype: str = "float16",
):
    """Row‑chunked datasets with optimized chunking and no compression for speed."""
    # Optimize chunk size for better I/O performance
    # If total rows in this HDF5 file is less than 10,000, make it a single chunk.
    # Otherwise, use a calculated optimal chunk size.
    if rows < 10000:
        optimal_chunk_rows = rows
    else:
        optimal_chunk_rows = min(max(1000, rows // 10), rows)

    for lid in layer_ids:
        g = hf.create_group(f"layer_{lid}")
        for name in ("inputs", "targets"):
            g.create_dataset(
                name,
                shape=(rows, d),
                dtype=h5py_dtype,
                chunks=(optimal_chunk_rows, d),  # Better chunking
                compression=None,  # No compression for 10-20x speedup
                # compression="gzip",  # OLD - this was the bottleneck
                # compression_opts=2,  # OLD
            )


def _async_uploader(upload_q: "queue.Queue[Optional[Path]]", cfg: ActivationConfig):
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
            except AttributeError:
                if item is None:
                    logger.debug("Drained a None sentinel from upload queue during no-URL shutdown.")
                    upload_q.task_done()
                else:
                    logger.error("Unexpected item type in upload queue during draining.")
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
                logger.info(
                    f"[Uploader Thread Attempt {attempt + 1}/{max_retries_per_chunk}] Uploading chunk: {p.name} to {url}"
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
    def __init__(self, dim: int, device: Optional[torch.device | str] = None):
        self.n = 0
        self.device = (
            torch.device(device) if isinstance(device, str) else device
        )  # Resolve device string to torch.device

        if self.device and self.device.type == "mps":
            self.stats_dtype = torch.float32
            logger.info("Using float32 for running stats on MPS device.")
        else:
            self.stats_dtype = torch.float64

        # Initialize to CPU if device is None, then move on first update, or initialize directly if device is known.
        initial_device_for_zeros = self.device if self.device else "cpu"
        self.mean = torch.zeros(dim, dtype=self.stats_dtype, device=initial_device_for_zeros)
        self.M2 = torch.zeros(dim, dtype=self.stats_dtype, device=initial_device_for_zeros)

    def update(self, x: torch.Tensor):
        """Update running mean & M2 using a mini-batch (Welford, parallel form).

        This corrects the previous implementation which **under-estimated** the
        variance by failing to include the between-batch mean shift term.
        """
        if self.device is None:
            self.device = x.device
            # Update self.stats_dtype if it was default and first tensor is MPS
            if self.device.type == "mps" and self.stats_dtype == torch.float64:
                self.stats_dtype = torch.float32
                logger.info("Switched running stats to float32 due to MPS device tensor.")
            self.mean = self.mean.to(device=self.device, dtype=self.stats_dtype)
            self.M2 = self.M2.to(device=self.device, dtype=self.stats_dtype)
        elif x.device != self.device:
            x = x.to(self.device)

        # Ensure x is on the correct device and has the stats_dtype for calculations
        x = x.to(device=self.device, dtype=self.stats_dtype)

        cnt = x.shape[0]
        if cnt == 0:
            return  # nothing to do

        # Batch statistics
        batch_mean = x.mean(0)
        batch_M2 = ((x - batch_mean).pow(2)).sum(0)  # Σ (x_i − μ_batch)^2

        # Combine with running statistics (parallel update formula)
        delta = batch_mean - self.mean
        total_n = self.n + cnt

        # Update running mean
        self.mean += delta * (cnt / total_n)

        # Update running M2 – include within-batch and between-batch terms
        self.M2 += batch_M2 + delta.pow(2) * self.n * cnt / total_n

        # Update count
        self.n = int(total_n)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        var = self.M2 / max(self.n - 1, 1)
        # Ensure tensors are moved to CPU before NumPy conversion and operations like np.sqrt
        mean_cpu = self.mean.cpu()
        var_cpu = var.cpu()
        return mean_cpu.numpy().astype("float32"), np.sqrt(var_cpu.numpy()).astype("float32")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ActivationGenerator:
    def __init__(self, cfg: ActivationConfig, device: torch.device | str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Profiler Init (Conditional) ---
        if cfg.enable_profiling:
            self.profiler: Optional[PerformanceProfiler] = PerformanceProfiler(
                chunk_tokens_threshold=cfg.chunk_token_threshold
            )
        else:
            self.profiler = None  # type: ignore
        # --- End Profiler Init ---

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
        self.out_dir = Path(cfg.activation_dir) / cfg.model_name / f"{ds_name}_{cfg.dataset_split}_{cfg.target_total_tokens}_{cfg.activation_dtype}"
        if cfg.dataset_skip is not None:
            start_idx = cfg.dataset_skip
            end_idx = cfg.dataset_skip + cfg.target_total_tokens
            self.out_dir = self.out_dir / f"{start_idx}_{end_idx}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_tmp = self.out_dir / "index.tmp"
        self.manifest_final = self.out_dir / "index.bin"
        # Background uploader
        self.upload_q: "queue.Queue[Optional[Path]]" = queue.Queue()
        if cfg.remote_server_url:
            self.uploader: Optional[threading.Thread] = threading.Thread(
                target=_async_uploader, args=(self.upload_q, cfg), daemon=True
            )
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
        if self.profiler:
            self.profiler.log_system_metrics("initial_system_state")
            self.profiler.total_tokens_processed_for_batch_profiling = 0
            self.profiler.batch_processing_total_calls = 0

        with self._conditional_measure("stream_activations_setup"):
            stream = self.extractor.stream_activations(
                dataset_path=cfg.dataset_path,
                dataset_split=cfg.dataset_split,
                dataset_text_column=cfg.dataset_text_column,
                dataset_skip=cfg.dataset_skip,
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

        for batch_idx, (batch_inp, batch_tgt) in enumerate(stream):
            with self._conditional_measure("batch_processing_total"):
                if tgt_tokens and g_row >= tgt_tokens:
                    break
                if not batch_inp:
                    continue

                with self._conditional_measure("batch_metadata_setup"):
                    if layer_ids is None:
                        layer_ids = sorted(batch_inp.keys())
                        d_model = batch_inp[layer_ids[0]].shape[-1]
                        dtype_str = str(batch_inp[layer_ids[0]].dtype)
                        if self.profiler:
                            self.profiler.set_layer_ids_ref(layer_ids)
                        for lid in layer_ids:
                            buf_inp[lid] = []
                            buf_tgt[lid] = []
                            if cfg.compute_norm_stats:
                                stats[lid] = {
                                    "inputs": _RunningStat(d_model, device=self.device),
                                    "targets": _RunningStat(d_model, device=self.device),
                                }
                        logger.info(
                            "Layers=%d d_model=%d dtype=%s", len(layer_ids) if layer_ids else 0, d_model, dtype_str
                        )

                n_tok_in_batch = 0
                if layer_ids and batch_inp.get(layer_ids[0]) is not None:
                    n_tok_in_batch = batch_inp[layer_ids[0]].shape[0]

                with self._conditional_measure("batch_gpu_tensor_accumulate"):
                    if layer_ids:
                        for lid in layer_ids:
                            if lid in batch_inp and lid in batch_tgt:
                                inp = batch_inp[lid].detach()
                                tgt = batch_tgt[lid].detach()
                                buf_inp[lid].append(inp)
                                buf_tgt[lid].append(tgt)
                                if cfg.compute_norm_stats and lid in stats:
                                    with self._conditional_measure(f"batch_norm_stats_update_layer_{lid}"):
                                        stats[lid]["inputs"].update(inp)
                                        stats[lid]["targets"].update(tgt)
                            else:
                                logger.warning(
                                    f"Layer {lid} expected but not found in current batch. Skipping accumulation for this layer."
                                )

                if n_tok_in_batch > 0:
                    g_row += n_tok_in_batch
                    pbar.update(n_tok_in_batch)
                    if self.profiler:
                        self.profiler.total_tokens_processed_for_batch_profiling += n_tok_in_batch
                if self.profiler:
                    self.profiler.batch_processing_total_calls += 1

                if layer_ids and buf_inp.get(layer_ids[0]):
                    cur_rows = sum(t.shape[0] for t in buf_inp[layer_ids[0]])
                    if cur_rows >= chunk_tokens:
                        with self._conditional_measure("chunk_write_dispatch"):
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
                        with self._conditional_measure("chunk_buffer_clear"):
                            if layer_ids:
                                for lid_clear in layer_ids:
                                    buf_inp[lid_clear].clear()
                                    buf_tgt[lid_clear].clear()
            if batch_idx > 0 and batch_idx % 50 == 0:
                if self.profiler:
                    self.profiler.log_system_metrics(f"batch_interval_{batch_idx}")

        # Flush final partial chunk
        if layer_ids and buf_inp.get(layer_ids[0]):
            with self._conditional_measure("final_chunk_write_dispatch"):
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

        if self.profiler:
            self.profiler.log_system_metrics("pre_manifest_write")
        with self._conditional_measure("manifest_concatenate_and_write"):
            if manifest_rows:
                manifest_arr = np.concatenate(manifest_rows, axis=0)
                manifest_arr.tofile(self.manifest_final)
            else:
                logger.warning("Manifest_rows is empty, skipping manifest write.")

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
        with self._conditional_measure("metadata_json_write"):
            with open(self.out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
        logger.info("metadata.json written")

        meta_path = self.out_dir / "metadata.json"
        if self.storage_type == "remote" and self.cfg.remote_server_url:
            with self._conditional_measure("metadata_json_upload"):
                try:
                    self._upload_json(meta_path, "metadata")
                    logger.info("metadata.json uploaded to server")
                except Exception as e:
                    logger.warning("Failed to upload metadata.json: %s", e)

        # Write norm_stats.json
        if cfg.compute_norm_stats and stats:
            norm: Dict[str, Any] = {}
            if layer_ids:
                for lid in layer_ids:
                    if lid in stats:
                        with self._conditional_measure(f"norm_stats_finalize_layer_{lid}"):
                            m_in, s_in = stats[lid]["inputs"].finalize()
                            m_tg, s_tg = stats[lid]["targets"].finalize()
                        norm[str(lid)] = {
                            "inputs": {"mean": m_in.tolist(), "std": s_in.tolist()},
                            "targets": {"mean": m_tg.tolist(), "std": s_tg.tolist()},
                        }
                    else:
                        logger.warning(f"Layer ID {lid} not found in stats dict during norm_stats finalization.")
            else:
                logger.warning("layer_ids is None, cannot write norm_stats.")

            if norm:
                with self._conditional_measure("norm_stats_json_write"):
                    with open(self.out_dir / "norm_stats.json", "w") as f:
                        json.dump(norm, f)
                logger.info("norm_stats.json written")

                norm_path = self.out_dir / "norm_stats.json"
                if self.storage_type == "remote" and self.cfg.remote_server_url:
                    with self._conditional_measure("norm_stats_json_upload"):
                        try:
                            self._upload_json(norm_path, "norm_stats")
                            logger.info("norm_stats.json uploaded to server")
                        except Exception as e:
                            logger.warning("Failed to upload norm_stats.json: %s", e)
            elif cfg.compute_norm_stats:
                logger.warning("Norm stats computation was enabled, but no norm stats were generated.")

        # Finish uploading (only if we are in remote mode)
        if self.storage_type == "remote" and self.uploader and self.upload_q:
            with self._conditional_measure("uploader_join"):
                self.upload_q.put(None)
                self.upload_q.join()
        logger.info("Finished: %d chunks, %s tokens", c_idx, f"{g_row:,}")
        if self.profiler:
            self.profiler.log_system_metrics("final_system_state")
            self.profiler.report()

    # ------------------------------------------------------------------
    @contextmanager
    def _conditional_measure(self, name: str):
        """Wrapper for profiler.measure that only runs if profiler is enabled."""
        if self.profiler:
            with self.profiler.measure(name):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    def _write_chunk(
        self,
        chunk_idx: int,
        buf_inp_gpu: Dict[int, List[torch.Tensor]],
        buf_tgt_gpu: Dict[int, List[torch.Tensor]],
        layer_ids: List[int],
        d_model: int,
        rows: int,
        manifest_rows: List[np.ndarray],
        offset: int,
    ):
        with self._conditional_measure(f"chunk_write_total_idx_{chunk_idx}"):
            p = self.out_dir / f"chunk_{chunk_idx}.{self.cfg.output_format}"

            if self.torch_dtype == torch.float32:
                h5py_dtype_str = "float32"
            elif self.torch_dtype == torch.float16:
                h5py_dtype_str = "float16"
            elif self.torch_dtype == torch.bfloat16:
                h5py_dtype_str = "uint16"
                logger.warning("Storing bfloat16 as uint16 in HDF5. Ensure client handles conversion.")
            else:
                raise ValueError(f"Unsupported torch_dtype for HDF5: {self.torch_dtype}")

            if self.cfg.output_format == "hdf5":
                num_write_workers = min(4, len(layer_ids) if layer_ids else 1)

                with self._conditional_measure(f"chunk_{chunk_idx}_hdf5_file_open_and_create_datasets"):
                    with h5py.File(p, "w", libver="latest") as hf:
                        for layer_id in layer_ids:
                            hf.create_dataset(
                                f"layer_{layer_id}/inputs",
                                shape=(rows, d_model),
                                dtype=h5py_dtype_str,
                                compression=self.cfg.compression if self.cfg.compression else None,
                                chunks=(min(rows, 16384), d_model),
                            )
                            hf.create_dataset(
                                f"layer_{layer_id}/targets",
                                shape=(rows, d_model),
                                dtype=h5py_dtype_str,
                                compression=self.cfg.compression if self.cfg.compression else None,
                                chunks=(min(rows, 16384), d_model),
                            )

                        # --- Use a SINGLE permutation shared across all layers --- #
                        if rows > 0:
                            shared_perm = torch.randperm(rows, device=next(iter(buf_inp_gpu.values()))[0].device)
                        else:
                            # Degenerate case – zero-row chunk (should not normally happen)
                            shared_perm = None

                        layer_data_to_write = []
                        for layer_id in layer_ids:
                            with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_data_prep"):
                                with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_concat"):
                                    layer_inp_gpu = torch.cat(buf_inp_gpu[layer_id], dim=0)
                                    layer_tgt_gpu = torch.cat(buf_tgt_gpu[layer_id], dim=0)

                                if shared_perm is not None:
                                    with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_permute"):
                                        layer_inp_gpu_perm = layer_inp_gpu[shared_perm]
                                        layer_tgt_gpu_perm = layer_tgt_gpu[shared_perm]
                                else:
                                    layer_inp_gpu_perm = layer_inp_gpu
                                    layer_tgt_gpu_perm = layer_tgt_gpu

                                with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_cpu_transfer"):
                                    layer_inp_cpu = layer_inp_gpu_perm.cpu()
                                    layer_tgt_cpu = layer_tgt_gpu_perm.cpu()

                                with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_convert_numpy"):
                                    inputs_np = (
                                        layer_inp_cpu.view(torch.int16).numpy()
                                        if self.torch_dtype == torch.bfloat16
                                        else layer_inp_cpu.numpy()
                                    )
                                    targets_np = (
                                        layer_tgt_cpu.view(torch.int16).numpy()
                                        if self.torch_dtype == torch.bfloat16
                                        else layer_tgt_cpu.numpy()
                                    )
                                layer_data_to_write.append((layer_id, inputs_np, targets_np))

                        def write_layer_data(layer_id_arg: int, inputs_data: np.ndarray, targets_data: np.ndarray):
                            try:
                                with h5py.File(p, "a", libver="latest") as hf_thread:
                                    hf_thread[f"layer_{layer_id_arg}/inputs"][:] = inputs_data
                                    hf_thread[f"layer_{layer_id_arg}/targets"][:] = targets_data
                                return layer_id_arg, None
                            except Exception as e:
                                logger.error(f"Error writing layer {layer_id_arg} to HDF5 chunk {chunk_idx}: {e}")
                                return layer_id_arg, e

                        futures = []
                        with self._conditional_measure(f"chunk_{chunk_idx}_parallel_hdf5_writes"):
                            with ThreadPoolExecutor(max_workers=num_write_workers) as executor:
                                for layer_id_val, inp_data, tgt_data in layer_data_to_write:
                                    futures.append(executor.submit(write_layer_data, layer_id_val, inp_data, tgt_data))

                            for future in as_completed(futures):
                                layer_written, error = future.result()
                                if error:
                                    logger.error(
                                        f"Failed HDF5 write for layer {layer_written} in chunk {chunk_idx}: {error}"
                                    )
                                # else:
                                # if self.profiler:
                                # self.profiler.log_event(f"chunk_{chunk_idx}_layer_{layer_written}_hdf5_write_success") # Commented out as log_event may not exist

            elif self.cfg.output_format == "npz":
                npz_save_dict = {}
                # --- Use a SINGLE permutation shared across all layers (same as HDF5 path) --- #
                if rows > 0:
                    shared_perm = torch.randperm(rows, device=next(iter(buf_inp_gpu.values()))[0].device)
                else:
                    shared_perm = None

                for layer_id in layer_ids:
                    with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_data_prep_npz"):
                        with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_concat_npz"):
                            layer_inp_gpu = torch.cat(buf_inp_gpu[layer_id], dim=0)
                            layer_tgt_gpu = torch.cat(buf_tgt_gpu[layer_id], dim=0)

                        if shared_perm is not None:
                            with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_permute_npz"):
                                layer_inp_gpu_perm = layer_inp_gpu[shared_perm]
                                layer_tgt_gpu_perm = layer_tgt_gpu[shared_perm]
                        else:
                            layer_inp_gpu_perm = layer_inp_gpu
                            layer_tgt_gpu_perm = layer_tgt_gpu

                        with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_cpu_transfer_npz"):
                            layer_inp_cpu = layer_inp_gpu_perm.cpu()
                            layer_tgt_cpu = layer_tgt_gpu_perm.cpu()

                        with self._conditional_measure(f"chunk_{chunk_idx}_layer_{layer_id}_convert_numpy_npz"):
                            inputs_np = (
                                layer_inp_cpu.view(torch.int16).numpy()
                                if self.torch_dtype == torch.bfloat16
                                else layer_inp_cpu.numpy()
                            )
                            targets_np = (
                                layer_tgt_cpu.view(torch.int16).numpy()
                                if self.torch_dtype == torch.bfloat16
                                else layer_tgt_cpu.numpy()
                            )
                        npz_save_dict[f"layer_{layer_id}_inputs"] = inputs_np
                        npz_save_dict[f"layer_{layer_id}_targets"] = targets_np

                with self._conditional_measure(f"chunk_{chunk_idx}_npz_file_save"):
                    if self.cfg.compression and self.cfg.compression.lower() != "none":
                        np.savez_compressed(p, **npz_save_dict)
                    else:
                        np.savez(p, **npz_save_dict)
            else:
                raise ValueError(f"Unsupported output_format: {self.cfg.output_format}")

            with self._conditional_measure(f"chunk_{chunk_idx}_manifest_append"):
                try:
                    # Attempt to use MANIFEST_DTYPE if defined on the instance
                    manifest_dtype = self.MANIFEST_DTYPE
                except AttributeError:
                    # Provide a default MANIFEST_DTYPE if not defined
                    manifest_dtype = np.dtype([("chunk_id", np.int32), ("num_tokens", np.int32), ("offset", np.int64)])

                # <<< ADDED DIAGNOSTIC LOGGING >>>
                logger.info(f"Manifest entry for chunk {chunk_idx}: num_tokens = {rows}, offset = {offset}")
                current_manifest_entry = np.array([(chunk_idx, rows, offset)], dtype=manifest_dtype)
                manifest_rows.append(current_manifest_entry)

            if self.storage_type == "remote" and self.cfg.remote_server_url:
                try:
                    self._schedule_upload(p, "chunk")  # Assuming _schedule_upload exists
                except AttributeError:
                    logger.warning("ActivationGenerator has no method '_schedule_upload'. Cannot upload chunk.")
            elif self.storage_type == "local" and self.cfg.delete_after_upload:
                pass

            logger.info(f"Chunk {chunk_idx} written ({rows} tokens) to {p}")

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

        if st == self.storage_type:
            return

        if st == "remote":
            if self.cfg.remote_server_url is None:
                raise ValueError("Cannot set storage_type to 'remote' because cfg.remote_server_url is not configured.")
            if self.uploader is None:
                self.uploader = threading.Thread(
                    target=_async_uploader,
                    args=(self.upload_q, self.cfg),
                    daemon=True,
                )
                self.uploader.start()
        else:  # switching to local
            if self.uploader is not None and self.upload_q:
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
            logger.warning(f"Attempted to upload {path.name} but remote_server_url is not configured.")
            return

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
                    return
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
            logger.warning(f"Attempted to upload {path.name} but remote_server_url is not configured.")
            return

        dataset_name = os.path.basename(self.cfg.dataset_path)
        dataset_id = quote(f"{self.cfg.model_name}/{dataset_name}_{self.cfg.dataset_split}", safe="")
        base = self.cfg.remote_server_url.rstrip("/") + "/"  # Point to root
        url = urljoin(base, f"datasets/{dataset_id}/{endpoint}")

        for attempt in range(max_retries):
            try:
                with open(path, "rb") as f:
                    file_key = f"{endpoint}_file"
                    if endpoint == "manifest":
                        file_key = "manifest_file"

                    files = {file_key: (path.name, f, "application/octet-stream")}
                    r = requests.post(url, files=files, timeout=300)
                    r.raise_for_status()
                    logger.info(f"Successfully uploaded {path.name} to {endpoint} endpoint on attempt {attempt + 1}")
                    return  # Success
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to upload {path.name} to {endpoint}: {e}")
                if attempt + 1 == max_retries:
                    logger.error(f"Final attempt failed to upload {path.name}. Giving up.")
                    return
                else:
                    delay = base_delay * (2**attempt)
                    logger.info(f"Retrying upload of {path.name} in {delay:.1f} seconds...")
                    time.sleep(delay)
            except Exception as e:
                logger.error(
                    f"Unexpected error during upload attempt {attempt + 1} for {path.name}: {e}", exc_info=True
                )
                return


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg_path = ap.parse_args().config
    with open(cfg_path) as f:
        loaded_config = yaml.safe_load(f)

    try:
        activation_config_instance = ActivationConfig(**loaded_config)
    except TypeError as e:
        logger.error(f"Error creating ActivationConfig from YAML. Ensure all keys are correct: {e}")
        import sys

        sys.exit(1)

    ActivationGenerator(activation_config_instance).generate_and_save()
