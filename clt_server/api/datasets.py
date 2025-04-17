from fastapi import (
    APIRouter,
    HTTPException,
    Body,
    UploadFile,
    File,
    Header,
    Query,
    Path as FastApiPath,
)
from fastapi.responses import JSONResponse, Response
from typing import List, Optional, Dict, Any
import logging
from urllib.parse import unquote  # To decode dataset_id

# Import the shared storage manager instance
from ..core.storage import storage_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# Helper to create dataset_id (consistent with generator)
def _create_dataset_id(model_name: str, dataset_name: str, split: str) -> str:
    return f"{model_name}/{dataset_name}_{split}"


# --- Dataset Listing and Info --- #


@router.get("/", summary="List available datasets")
async def list_datasets_endpoint():
    """Lists all datasets found in the storage directory."""

    try:
        datasets = await storage_manager.list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error listing datasets."
        )


@router.get("/{dataset_id:path}/info", summary="Get dataset metadata")
async def get_metadata_endpoint(
    dataset_id: str = FastApiPath(
        ..., description="Dataset ID (e.g., model_name/dataset_name_split)"
    )
):
    """Retrieves the metadata.json for a specific dataset."""
    # Decode dataset_id in case it contains URL-encoded characters like %2F for /
    decoded_dataset_id = unquote(dataset_id)
    metadata = await storage_manager.get_dataset_metadata(decoded_dataset_id)
    if metadata:
        return JSONResponse(content=metadata)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Metadata not found for dataset: {decoded_dataset_id}",
        )


@router.get("/{dataset_id:path}/norm_stats", summary="Get dataset normalization stats")
async def get_norm_stats_endpoint(
    dataset_id: str = FastApiPath(..., description="Dataset ID")
):
    """Retrieves the norm_stats.json for a specific dataset."""
    decoded_dataset_id = unquote(dataset_id)
    norm_stats = await storage_manager.get_norm_stats(decoded_dataset_id)
    if norm_stats:
        return JSONResponse(content=norm_stats)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Normalization stats not found for dataset: {decoded_dataset_id}",
        )


# --- Upload Endpoints (Generator -> Server) --- #
@router.post(
    "/{dataset_id:path}/chunks/{chunk_idx}",
    status_code=201,
    summary="Upload activation chunk",
)
async def upload_chunk_endpoint(
    dataset_id: str = FastApiPath(..., description="Dataset ID"),
    chunk_idx: int = FastApiPath(..., description="Index of the chunk being uploaded"),
    x_num_tokens: Optional[int] = Header(
        None, alias="X-Num-Tokens", description="Number of tokens in the chunk"
    ),
    # Get content type to check if it's HDF5
    content_type: Optional[str] = Header(
        None, alias="Content-Type", description="Content type of the chunk"
    ),
    # Add other headers if needed (X-Saved-Dtype)
    # Use UploadFile for efficient handling of large binary data
    chunk_file: UploadFile = File(
        ..., description="Serialized chunk data (HDF5 format)"
    ),
):
    """Receives and saves a single chunk of activation data (expected HDF5).
    The content type of the chunk should be 'application/x-hdf5'."""
    decoded_dataset_id = unquote(dataset_id)
    if x_num_tokens is None:
        raise HTTPException(status_code=400, detail="X-Num-Tokens header is required")
    # Validate content type (optional but good practice)
    # ---- Comment out this check as FastAPI handles multipart, and we check magic number later ----
    # if content_type is None or "hdf5" not in content_type.lower():
    #      logger.warning(f"Received chunk upload with unexpected Content-Type: {content_type}. Expected HDF5.")
    #      # Decide whether to reject or proceed cautiously
    #      raise HTTPException(status_code=415, detail="Unsupported Media Type. Expected application/x-hdf5")
    # ---------------------------------------------------------------------------------------------

    try:
        # Read the uploaded file content as bytes
        chunk_data = await chunk_file.read()
        if not chunk_data:
            raise HTTPException(status_code=400, detail="Received empty chunk data.")
        # Basic validation: Check if it *could* be HDF5 (magic number)
        # This is cheaper than fully loading/validating
        if not chunk_data.startswith(b"\x89HDF\r\n\x1a\n"):
            logger.error(
                f"Uploaded chunk {chunk_idx} for {decoded_dataset_id} does not have HDF5 magic number."
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid chunk data format. Expected HDF5 bytes.",
            )
        # Save using storage manager
        # StorageManager expects bytes and saves as .hdf5
        await storage_manager.save_chunk(
            decoded_dataset_id, chunk_idx, chunk_data, x_num_tokens
        )
        return {
            "message": f"Chunk {chunk_idx} for dataset {decoded_dataset_id} uploaded successfully."
        }
    except HTTPException as http_exc:
        raise http_exc  # Re-raise FastAPI validation errors
    except FileNotFoundError as fnf_err:
        logger.error(f"Storage error during chunk upload: {fnf_err}")
        raise HTTPException(
            status_code=500, detail="Server storage configuration error."
        )
    except Exception as e:
        logger.error(
            f"Error saving chunk {chunk_idx} for {decoded_dataset_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error saving chunk {chunk_idx}."
        )
    finally:
        # Ensure the temporary file is closed
        await chunk_file.close()


@router.post(
    "/{dataset_id:path}/metadata", status_code=201, summary="Upload dataset metadata"
)
async def upload_metadata_endpoint(
    dataset_id: str = FastApiPath(..., description="Dataset ID"),
    metadata: Dict[str, Any] = Body(..., description="Metadata JSON content"),
):
    """Receives and saves the metadata.json content."""
    decoded_dataset_id = unquote(dataset_id)
    try:
        await storage_manager.save_metadata(decoded_dataset_id, metadata)
        return {
            "message": f"Metadata for dataset {decoded_dataset_id} uploaded successfully."
        }
    except Exception as e:
        logger.error(
            f"Error saving metadata for {decoded_dataset_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error saving metadata."
        )


@router.post(
    "/{dataset_id:path}/norm_stats",
    status_code=201,
    summary="Upload normalization statistics",
)
async def upload_norm_stats_endpoint(
    dataset_id: str = FastApiPath(..., description="Dataset ID"),
    norm_stats: Dict[str, Any] = Body(
        ..., description="Normalization statistics JSON content"
    ),
):
    """Receives and saves the norm_stats.json content."""
    decoded_dataset_id = unquote(dataset_id)
    try:
        await storage_manager.save_norm_stats(decoded_dataset_id, norm_stats)
        return {
            "message": f"Normalization stats for dataset {decoded_dataset_id} uploaded successfully."
        }
    except Exception as e:
        logger.error(
            f"Error saving norm_stats for {decoded_dataset_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error saving normalization stats."
        )


# --- Batch Request Endpoint (Client -> Server) --- #


@router.get("/{dataset_id:path}/batch", summary="Request a training batch")
async def get_batch_endpoint(
    dataset_id: str = FastApiPath(..., description="Dataset ID"),
    num_tokens: int = Query(..., description="Target number of tokens for the batch"),
    layers: Optional[str] = Query(
        None, description="Comma-separated layer indices (e.g., '0,1,5'). Default: all."
    ),
    # format query param - TBD if needed
):
    """Retrieves a random batch of activations, serialized using torch.save."""
    decoded_dataset_id = unquote(dataset_id)
    layer_list: Optional[List[int]] = None
    if layers:
        try:
            layer_list = [int(l.strip()) for l in layers.split(",") if l.strip()]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid format for 'layers' query parameter. Must be comma-separated integers.",
            )
    if num_tokens <= 0:
        raise HTTPException(status_code=400, detail="'num_tokens' must be positive.")

    try:
        # Use the storage manager's get_batch method (which now reads HDF5)
        batch_bytes = await storage_manager.get_batch(
            decoded_dataset_id, num_tokens, layer_list
        )
        # Return the raw bytes with appropriate content type
        return Response(content=batch_bytes, media_type="application/octet-stream")

    except FileNotFoundError as e:
        logger.warning(f"Batch request failed for {decoded_dataset_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Dataset or required chunks not found for {decoded_dataset_id}. {e}",
        )
    except ValueError as e:
        logger.warning(f"Batch request value error for {decoded_dataset_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate batch for {decoded_dataset_id}: {e}",
        )
    except Exception as e:
        logger.error(
            f"Error getting batch for {decoded_dataset_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error retrieving batch."
        )
