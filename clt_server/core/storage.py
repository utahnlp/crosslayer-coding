"""Stateless slice‑server for CLT activations.

Changes from previous version:

*   Replaces the ad‑hoc *random batch* endpoint with
    **`/slice?chunk=<int>&rows=...`** returning raw bf16 bytes.
*   Caches `h5py.File` handles in an LRU to avoid reopen cost.
*   Serves `metadata.json`, `norm_stats.json`, `index.bin` unchanged.
*   Upload endpoint (`POST /chunks/{idx}`) is kept for the generator.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

import h5py
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.status import HTTP_201_CREATED
from pydantic import BaseModel
import numpy as np

from .config import settings

ROOT = Path(settings.STORAGE_BASE_DIR)
ROOT.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
app = FastAPI(title="CLT Activation Slice Server", version="0.3")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ds_path(dataset_id: str) -> Path:
    return ROOT / dataset_id


@lru_cache(maxsize=128)
def _open_h5(path: Path) -> h5py.File:
    return h5py.File(path, "r")


@lru_cache(maxsize=256)  # Maxsize can be tuned based on typical dataset sizes and server memory
def _load_layer_data_cached(chunk_path: Path, layer_key: str, data_type: str) -> np.ndarray:
    """Loads an entire dataset (e.g., 'inputs' or 'targets' for a layer) from an HDF5 chunk and caches it."""
    logger.debug(f"Cache miss or first load: Reading {data_type} for layer {layer_key} from {chunk_path}")
    hf = _open_h5(chunk_path)  # This itself is cached
    try:
        # Ensure the dataset exists before trying to slice
        if layer_key not in hf:
            raise KeyError(f"Layer {layer_key} not found in chunk {chunk_path}")
        if data_type not in hf[layer_key]:
            raise KeyError(f"Data type {data_type} not found in layer {layer_key} of chunk {chunk_path}")

        data = hf[layer_key][data_type][:]  # Load entire dataset
        return data
    except KeyError as e:
        logger.error(f"Error accessing HDF5 dataset: {chunk_path} / {layer_key} / {data_type} - {e}")
        # Re-raise as HTTPException or a custom server error if this function is directly used by an endpoint handler
        # For now, re-raise KeyError for the caller (slice_chunk) to handle as HTTPException(500)
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading HDF5 dataset: {chunk_path} / {layer_key} / {data_type} - {e}")
        raise  # Let slice_chunk handle this as a 500 error


# ---------------------------------------------------------------------------
# Upload endpoint – unchanged except for file ext check
# ---------------------------------------------------------------------------
@app.post("/datasets/{dataset_id:path}/chunks/{chunk_idx}", status_code=HTTP_201_CREATED)
async def upload_chunk(dataset_id: str, chunk_idx: int, chunk_file: UploadFile = File(...)):
    ds_dir = _ds_path(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    fname = ds_dir / f"chunk_{chunk_idx}.h5"
    with open(fname, "wb") as f:
        while True:
            blk = await chunk_file.read(8 << 20)
            if not blk:
                break
            f.write(blk)
    logger.info("saved %s (%s bytes)", fname, fname.stat().st_size)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# New Slice Request Body Model
# ---------------------------------------------------------------------------
class SliceRequest(BaseModel):
    rows: List[int]


# ---------------------------------------------------------------------------
# New slice endpoint (changed to POST)
# ---------------------------------------------------------------------------
@app.post("/datasets/{dataset_id:path}/slice")
async def slice_chunk(
    dataset_id: str,
    chunk: int = Query(..., ge=0, description="Chunk index to slice"),
    request_body: SliceRequest = Body(...),
):
    """
    Return raw bf16 bytes for the requested rows of a given chunk.
    Payload layout per layer:  inputs  then  targets (contiguous).
    Reads row indices from the POST request body.
    """
    ds_dir = _ds_path(dataset_id)
    chunk_path = ds_dir / f"chunk_{chunk}.h5"
    if not chunk_path.exists():
        raise HTTPException(404, f"Chunk {chunk} not found")

    # ---- get rows from body and convert to numpy ----------------------------
    row_list = request_body.rows
    if not row_list:
        raise HTTPException(400, "rows list in request body cannot be empty")

    try:
        # Convert list of ints to numpy uint32 array
        row_idx = np.array(row_list, dtype=np.uint32)
    except (ValueError, TypeError):
        raise HTTPException(400, "rows must be a list of integers")

    row_idx.sort()  # <‑‑ key speed‑up (contiguous reads)

    # ---- open chunk (cached via LRU) -----------------------------------------
    try:
        hf = _open_h5(chunk_path)
    except Exception as e:
        logger.error("Failed to open %s: %s", chunk_path, e)
        raise HTTPException(500, "Corrupt chunk file")

    # Sort layer keys numerically to ensure correct byte order
    def _layer_sort_key(name: str) -> int:
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 1_000_000  # Push unparsable names to the end

    layer_keys = sorted([k for k in hf.keys() if k.startswith("layer_")], key=_layer_sort_key)
    if not layer_keys:
        raise HTTPException(500, "No layer groups in chunk")

    # ---- zero‑copy streaming --------------------------------------------------
    def iter_layers():
        """
        Yields bytes for each layer (inputs then targets) without
        accumulating them in a BytesIO buffer.
        """
        for lk in layer_keys:
            try:
                # Load full data (potentially from cache)
                inputs_data_full = _load_layer_data_cached(chunk_path, lk, "inputs")
                targets_data_full = _load_layer_data_cached(chunk_path, lk, "targets")

                # Slice the (potentially cached) full data
                yield inputs_data_full[row_idx, :].tobytes()
                yield targets_data_full[row_idx, :].tobytes()
            except KeyError as e:  # Catch KeyErrors from _load_layer_data_cached
                logger.error(f"Data integrity issue for chunk {chunk_path}, layer {lk}: {e}")
                raise HTTPException(status_code=500, detail=f"Missing data in chunk {chunk}, layer {lk}")
            except Exception as e:
                logger.error(f"Unexpected error processing layer {lk} in chunk {chunk_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {chunk}, layer {lk}")

    return StreamingResponse(
        iter_layers(),
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Static helpers – metadata, norm_stats, index.bin
# ---------------------------------------------------------------------------
@app.get("/datasets/{dataset_id:path}/info")
async def dataset_info(dataset_id: str):
    p = _ds_path(dataset_id) / "metadata.json"
    if not p.exists():
        raise HTTPException(404)
    return JSONResponse(json.load(open(p)))


@app.get("/datasets/{dataset_id:path}/norm_stats")
async def norm_stats(dataset_id: str):
    p = _ds_path(dataset_id) / "norm_stats.json"
    if not p.exists():
        raise HTTPException(404)
    return JSONResponse(json.load(open(p)))


@app.get("/datasets/{dataset_id:path}/manifest")
async def manifest(dataset_id: str):
    p = _ds_path(dataset_id) / "index.bin"
    if not p.exists():
        raise HTTPException(404)
    return StreamingResponse(open(p, "rb"), media_type="application/octet-stream")


@app.post("/datasets/{dataset_id:path}/manifest", status_code=HTTP_201_CREATED)
async def upload_manifest(dataset_id: str, manifest_file: UploadFile = File(...)):
    """Accept upload of the index.bin manifest file."""
    ds_dir = _ds_path(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    path = ds_dir / "index.bin"
    try:
        # Use async read/write for UploadFile
        content = await manifest_file.read()
        with open(path, "wb") as f:
            f.write(content)
        logger.info("Saved manifest %s (%s bytes)", path, len(content))
        return {"status": "ok"}
    except Exception as e:
        logger.error("Failed to save manifest for %s: %s", dataset_id, e)
        raise HTTPException(500, "Failed to save manifest")
    finally:
        await manifest_file.close()


# ---------------------------------------------------------------------------
# JSON Upload endpoints (metadata, norm_stats)
# ---------------------------------------------------------------------------
@app.post("/datasets/{dataset_id:path}/metadata", status_code=HTTP_201_CREATED)
async def upload_metadata(dataset_id: str, metadata: Dict[str, Any]):
    ds_dir = _ds_path(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    path = ds_dir / "metadata.json"
    try:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved %s", path)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Failed to save metadata for %s: %s", dataset_id, e)
        raise HTTPException(500, "Failed to save metadata")


@app.post("/datasets/{dataset_id:path}/norm_stats", status_code=HTTP_201_CREATED)
async def upload_norm_stats(dataset_id: str, norm_stats_data: Dict[str, Any]):
    ds_dir = _ds_path(dataset_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    path = ds_dir / "norm_stats.json"
    try:
        with open(path, "w") as f:
            json.dump(norm_stats_data, f, indent=2)
        logger.info("Saved %s", path)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Failed to save norm_stats for %s: %s", dataset_id, e)
        raise HTTPException(500, "Failed to save norm_stats")
