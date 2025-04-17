import pytest
import pytest_asyncio
import torch
import h5py
import json
from pathlib import Path
import numpy as np
import sys
import os
from typing import AsyncGenerator, Dict, Any, Generator
from httpx import AsyncClient
from urllib.parse import quote  # For testing URL encoding
import io

# Ensure the project root is in the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastAPI app and storage manager/settings
try:
    from clt_server.main import app
    from clt_server.core.storage import StorageManager
    from clt_server.core.config import settings
except ImportError as e:
    print(f"Error importing server components: {e}")
    print(
        "Ensure the test is run from the project root or PYTHONPATH is set correctly."
    )
    raise

# --- Fixtures --- #


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provides an httpx AsyncClient configured for the test app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
def test_storage_manager(
    tmp_path: Path, monkeypatch
) -> Generator[StorageManager, None, None]:
    """Overrides the global storage_manager with a temporary one for the test scope."""
    test_storage_dir = tmp_path / "api_test_server_data"
    # Monkeypatch the setting itself BEFORE creating the StorageManager instance
    monkeypatch.setattr(settings, "STORAGE_BASE_DIR", test_storage_dir)

    # Create a new StorageManager instance for this test
    test_manager = StorageManager(base_dir=settings.STORAGE_BASE_DIR)

    # Monkeypatch the global instance used by the API endpoints
    # This is crucial because the API routes import the global instance directly
    monkeypatch.setattr("clt_server.api.datasets.storage_manager", test_manager)
    # Need to patch it in batch endpoint too if it uses the global one
    # Assuming it does, let's patch there as well for safety
    # (If datasets.py imports .core.storage.storage_manager directly, this works)

    yield test_manager  # Provide the test-specific manager if needed directly in tests

    # Teardown handled by monkeypatch


@pytest.fixture(scope="function")
def create_test_hdf5_chunk(tmp_path: Path) -> Dict[str, Any]:
    """Creates a dummy HDF5 chunk file and returns its path and info."""
    chunk_path = tmp_path / "test_chunk_for_api_0.hdf5"
    num_tokens = 50
    d_model = 8
    layer_indices = [0, 1]
    dtype = torch.float32
    np_dtype = np.float32

    with h5py.File(chunk_path, "w") as f:
        f.attrs["num_tokens"] = num_tokens
        f.attrs["saved_dtype"] = str(dtype)
        for layer_idx in layer_indices:
            group = f.create_group(f"layer_{layer_idx}")
            inputs_data = torch.randn(num_tokens, d_model, dtype=dtype)
            targets_data = torch.randn(num_tokens, d_model, dtype=dtype)
            group.create_dataset("inputs", data=inputs_data.numpy().astype(np_dtype))
            group.create_dataset("targets", data=targets_data.numpy().astype(np_dtype))

    # Read the raw bytes for upload testing
    with open(chunk_path, "rb") as f_bytes:
        chunk_bytes = f_bytes.read()

    return {
        "path": chunk_path,
        "bytes": chunk_bytes,
        "num_tokens": num_tokens,
        "d_model": d_model,
        "layer_indices": layer_indices,
        "dtype": dtype,
    }


# --- Test Cases --- #


# Tests for GET /api/v1/datasets/
@pytest.mark.asyncio
async def test_list_datasets_empty(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test listing datasets when none exist."""
    response = await async_client.get("/api/v1/datasets/")
    assert response.status_code == 200
    assert response.json() == {"datasets": []}


@pytest.mark.asyncio
async def test_list_datasets_one(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test listing datasets with one dataset present."""
    dataset_id = "model1/ds1_train"
    metadata = {"info": "test metadata 1"}
    await test_storage_manager.save_metadata(dataset_id, metadata)

    response = await async_client.get("/api/v1/datasets/")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["id"] == dataset_id
    assert datasets[0]["metadata"] == metadata


@pytest.mark.asyncio
async def test_list_datasets_multiple(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test listing datasets with multiple datasets present."""
    dataset_id1 = "model1/ds1_train"
    metadata1 = {"info": "test metadata 1"}
    dataset_id2 = "model2/ds2_test"
    metadata2 = {"config": "other format"}
    await test_storage_manager.save_metadata(dataset_id1, metadata1)
    await test_storage_manager.save_metadata(dataset_id2, metadata2)
    # Add one dir without metadata - should not be listed
    (test_storage_manager.get_dataset_dir("model1/no_meta_ds")).mkdir(parents=True)

    response = await async_client.get("/api/v1/datasets/")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert len(datasets) == 2
    dataset_ids_found = {d["id"] for d in datasets}
    assert dataset_ids_found == {dataset_id1, dataset_id2}


# TODO: Test GET /api/v1/datasets/ (error handling 500)


# --- Tests for GET /api/v1/datasets/{dataset_id}/info ---
@pytest.mark.asyncio
async def test_get_metadata_success(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test successful retrieval of metadata."""
    dataset_id = "model_meta/ds_meta_test"
    metadata = {"some_key": "some_value", "nested": {"num": 1}}
    await test_storage_manager.save_metadata(dataset_id, metadata)

    response = await async_client.get(f"/api/v1/datasets/{dataset_id}/info")
    assert response.status_code == 200
    assert response.json() == metadata


@pytest.mark.asyncio
async def test_get_metadata_not_found(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test retrieving metadata for a non-existent dataset."""
    dataset_id = "nonexistent/model_split"
    response = await async_client.get(f"/api/v1/datasets/{dataset_id}/info")
    assert response.status_code == 404
    assert dataset_id in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_metadata_url_encoded(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test retrieving metadata using a URL-encoded dataset ID."""
    dataset_id_plain = "model/with/slashes_split"
    dataset_id_encoded = quote(dataset_id_plain, safe="")  # Fully encode
    metadata = {"encoded_test": True}
    await test_storage_manager.save_metadata(dataset_id_plain, metadata)

    response = await async_client.get(f"/api/v1/datasets/{dataset_id_encoded}/info")
    assert response.status_code == 200
    assert response.json() == metadata


# TODO: Test GET /api/v1/datasets/{dataset_id}/info (error handling 500)


# --- Tests for GET /api/v1/datasets/{dataset_id}/norm_stats ---
@pytest.mark.asyncio
async def test_get_norm_stats_success(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test successful retrieval of normalization stats."""
    dataset_id = "model_norm/ds_norm_test"
    norm_stats = {"0": {"mean": [0.1, 0.2], "std": [1.0, 1.1]}}
    await test_storage_manager.save_norm_stats(dataset_id, norm_stats)

    response = await async_client.get(f"/api/v1/datasets/{dataset_id}/norm_stats")
    assert response.status_code == 200
    # JSON stores lists, storage manager loads them, API returns JSON
    assert response.json() == norm_stats


@pytest.mark.asyncio
async def test_get_norm_stats_dataset_not_found(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test retrieving norm stats for a non-existent dataset."""
    dataset_id = "nonexistent/model_split_norm"
    response = await async_client.get(f"/api/v1/datasets/{dataset_id}/norm_stats")
    assert response.status_code == 404  # Should be 404 as the dir won't exist
    assert dataset_id in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_norm_stats_file_not_found(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test retrieving norm stats when the dataset exists but stats file doesn't."""
    dataset_id = "model_norm/no_stats_file"
    # Create dataset dir but no norm_stats.json
    test_storage_manager.get_dataset_dir(dataset_id).mkdir(parents=True)

    response = await async_client.get(f"/api/v1/datasets/{dataset_id}/norm_stats")
    assert response.status_code == 404
    assert "Normalization stats not found" in response.json()["detail"]
    assert dataset_id in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_norm_stats_url_encoded(
    async_client: AsyncClient, test_storage_manager: StorageManager
):
    """Test retrieving norm stats using a URL-encoded dataset ID."""
    dataset_id_plain = "model/norm/slashes_split"
    dataset_id_encoded = quote(dataset_id_plain, safe="")
    norm_stats = {"1": {"mean": [0.5]}}
    await test_storage_manager.save_norm_stats(dataset_id_plain, norm_stats)

    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id_encoded}/norm_stats"
    )
    assert response.status_code == 200
    assert response.json() == norm_stats


# TODO: Test GET /api/v1/datasets/{dataset_id}/norm_stats (error handling 500)


# --- Tests for POST /api/v1/datasets/{dataset_id}/chunks/{chunk_idx} ---
@pytest.mark.asyncio
async def test_upload_chunk_success(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
):
    """Test successful upload of an HDF5 chunk."""
    dataset_id = "chunk_model/chunk_test_ds"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens = chunk_info["num_tokens"]
    chunk_bytes = chunk_info["bytes"]
    target_chunk_path = (
        test_storage_manager.get_dataset_dir(dataset_id) / f"chunk_{chunk_idx}.hdf5"
    )

    assert not target_chunk_path.exists()

    headers = {"X-Num-Tokens": str(num_tokens)}
    # httpx automatically sets Content-Type for files based on name
    files = {
        "chunk_file": (f"chunk_{chunk_idx}.hdf5", chunk_bytes, "application/x-hdf5")
    }

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/chunks/{chunk_idx}",
        headers=headers,
        files=files,
    )

    assert response.status_code == 201
    assert response.json() == {
        "message": f"Chunk {chunk_idx} for dataset {dataset_id} uploaded successfully."
    }

    # Verify file was saved correctly by the (mocked) storage manager
    assert target_chunk_path.exists()
    assert target_chunk_path.stat().st_size == len(chunk_bytes)
    # Quick check: verify HDF5 magic number
    with open(target_chunk_path, "rb") as f:
        assert f.read(8) == b"\x89HDF\r\n\x1a\n"


@pytest.mark.asyncio
async def test_upload_chunk_missing_header(
    async_client: AsyncClient, create_test_hdf5_chunk: Dict[str, Any]
):
    """Test uploading chunk without the X-Num-Tokens header."""
    dataset_id = "chunk_model/chunk_test_ds"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    chunk_bytes = chunk_info["bytes"]
    files = {
        "chunk_file": (f"chunk_{chunk_idx}.hdf5", chunk_bytes, "application/x-hdf5")
    }

    # No headers provided
    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/chunks/{chunk_idx}", files=files
    )

    assert response.status_code == 400
    assert "X-Num-Tokens header is required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_chunk_invalid_data(
    async_client: AsyncClient,
):
    """Test uploading data that is not a valid HDF5 file."""
    dataset_id = "chunk_model/invalid_data"
    chunk_idx = 0
    invalid_bytes = b"this is not hdf5 data"
    num_tokens = 10

    headers = {"X-Num-Tokens": str(num_tokens)}
    files = {
        "chunk_file": (
            f"chunk_{chunk_idx}.bin",
            invalid_bytes,
            "application/octet-stream",
        )
    }

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/chunks/{chunk_idx}",
        headers=headers,
        files=files,
    )

    assert response.status_code == 400
    assert "Invalid chunk data format. Expected HDF5 bytes" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_chunk_empty_data(
    async_client: AsyncClient,
):
    """Test uploading an empty file as a chunk."""
    dataset_id = "chunk_model/empty_data"
    chunk_idx = 0
    empty_bytes = b""
    num_tokens = 0  # Or some number, header is required anyway

    headers = {"X-Num-Tokens": str(num_tokens)}
    files = {
        "chunk_file": (f"chunk_{chunk_idx}.hdf5", empty_bytes, "application/x-hdf5")
    }

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/chunks/{chunk_idx}",
        headers=headers,
        files=files,
    )

    assert response.status_code == 400
    assert "Received empty chunk data" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_chunk_url_encoded_id(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
):
    """Test uploading a chunk using a URL-encoded dataset ID."""
    dataset_id_plain = "chunk/model/with_slash"
    dataset_id_encoded = quote(dataset_id_plain, safe="")
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens = chunk_info["num_tokens"]
    chunk_bytes = chunk_info["bytes"]
    target_chunk_path = (
        test_storage_manager.get_dataset_dir(dataset_id_plain)
        / f"chunk_{chunk_idx}.hdf5"
    )

    assert not target_chunk_path.exists()

    headers = {"X-Num-Tokens": str(num_tokens)}
    files = {
        "chunk_file": (f"chunk_{chunk_idx}.hdf5", chunk_bytes, "application/x-hdf5")
    }

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id_encoded}/chunks/{chunk_idx}",
        headers=headers,
        files=files,
    )

    assert response.status_code == 201
    assert dataset_id_plain in response.json()["message"]
    assert target_chunk_path.exists()


@pytest.mark.asyncio
async def test_upload_chunk_storage_error(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
    monkeypatch,
):
    """Test chunk upload when storage_manager.save_chunk fails."""
    dataset_id = "chunk_model/storage_fail"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens = chunk_info["num_tokens"]
    chunk_bytes = chunk_info["bytes"]

    # Mock the save_chunk method to raise an OSError
    async def mock_save_chunk_error(*args, **kwargs):
        raise OSError("Simulated disk full error")

    monkeypatch.setattr(test_storage_manager, "save_chunk", mock_save_chunk_error)

    headers = {"X-Num-Tokens": str(num_tokens)}
    files = {
        "chunk_file": (f"chunk_{chunk_idx}.hdf5", chunk_bytes, "application/x-hdf5")
    }

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/chunks/{chunk_idx}",
        headers=headers,
        files=files,
    )

    assert response.status_code == 500
    assert (
        f"Internal server error saving chunk {chunk_idx}" in response.json()["detail"]
    )


# --- Tests for POST /api/v1/datasets/{dataset_id}/metadata ---
@pytest.mark.asyncio
async def test_upload_metadata_success(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
):
    """Test successful upload of metadata JSON."""
    dataset_id = "meta_model/meta_ds_upload"
    metadata_content = {"config": "test", "layers": [0, 1, 2]}
    target_metadata_path = (
        test_storage_manager.get_dataset_dir(dataset_id) / "metadata.json"
    )

    assert not target_metadata_path.exists()

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/metadata",
        json=metadata_content,  # httpx handles JSON serialization
    )

    assert response.status_code == 201
    assert response.json() == {
        "message": f"Metadata for dataset {dataset_id} uploaded successfully."
    }

    # Verify file was saved correctly
    assert target_metadata_path.exists()
    with open(target_metadata_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == metadata_content


@pytest.mark.asyncio
async def test_upload_metadata_invalid_json(
    async_client: AsyncClient,
):
    """Test uploading invalid data as metadata JSON (should get 422)."""
    dataset_id = "meta_model/invalid_json"
    # Send raw string which is not valid top-level JSON for the Body(Dict)
    invalid_json_string = "this is not valid json"

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/metadata",
        content=invalid_json_string,  # Send as raw content
        headers={"Content-Type": "application/json"},
    )

    # FastAPI/Pydantic should handle validation and return 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_metadata_url_encoded_id(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
):
    """Test uploading metadata using a URL-encoded dataset ID."""
    dataset_id_plain = "meta/model/with_slash"
    dataset_id_encoded = quote(dataset_id_plain, safe="")
    metadata_content = {"encoded_id_test": True}
    target_metadata_path = (
        test_storage_manager.get_dataset_dir(dataset_id_plain) / "metadata.json"
    )

    assert not target_metadata_path.exists()

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id_encoded}/metadata", json=metadata_content
    )

    assert response.status_code == 201
    assert dataset_id_plain in response.json()["message"]
    assert target_metadata_path.exists()
    with open(target_metadata_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == metadata_content


@pytest.mark.asyncio
async def test_upload_metadata_storage_error(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    monkeypatch,
):
    """Test metadata upload when storage_manager.save_metadata fails."""
    dataset_id = "meta_model/storage_fail"
    metadata_content = {"data": "wont_be_saved"}

    # Mock the save_metadata method to raise an exception
    async def mock_save_meta_error(*args, **kwargs):
        raise IOError("Simulated permission error")

    monkeypatch.setattr(test_storage_manager, "save_metadata", mock_save_meta_error)

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/metadata", json=metadata_content
    )

    assert response.status_code == 500
    assert "Internal server error saving metadata" in response.json()["detail"]


# --- Tests for POST /api/v1/datasets/{dataset_id}/norm_stats ---
@pytest.mark.asyncio
async def test_upload_norm_stats_success(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
):
    """Test successful upload of norm_stats JSON."""
    dataset_id = "norm_model/norm_ds_upload"
    norm_stats_content = {"0": {"input_mean": [0.5], "input_std": [1.5]}}
    target_norm_stats_path = (
        test_storage_manager.get_dataset_dir(dataset_id) / "norm_stats.json"
    )

    assert not target_norm_stats_path.exists()

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/norm_stats", json=norm_stats_content
    )

    assert response.status_code == 201
    assert response.json() == {
        "message": f"Normalization stats for dataset {dataset_id} uploaded successfully."
    }

    # Verify file was saved correctly
    assert target_norm_stats_path.exists()
    with open(target_norm_stats_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == norm_stats_content


@pytest.mark.asyncio
async def test_upload_norm_stats_invalid_json(
    async_client: AsyncClient,
):
    """Test uploading invalid data as norm_stats JSON (should get 422)."""
    dataset_id = "norm_model/invalid_json_stats"
    invalid_json_string = "not: valid json"

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/norm_stats",
        content=invalid_json_string,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_norm_stats_url_encoded_id(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
):
    """Test uploading norm_stats using a URL-encoded dataset ID."""
    dataset_id_plain = "norm/model/with_slash_stats"
    dataset_id_encoded = quote(dataset_id_plain, safe="")
    norm_stats_content = {"1": {"output_mean": [0.1]}}
    target_norm_stats_path = (
        test_storage_manager.get_dataset_dir(dataset_id_plain) / "norm_stats.json"
    )

    assert not target_norm_stats_path.exists()

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id_encoded}/norm_stats", json=norm_stats_content
    )

    assert response.status_code == 201
    assert dataset_id_plain in response.json()["message"]
    assert target_norm_stats_path.exists()
    with open(target_norm_stats_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == norm_stats_content


@pytest.mark.asyncio
async def test_upload_norm_stats_storage_error(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    monkeypatch,
):
    """Test norm_stats upload when storage_manager.save_norm_stats fails."""
    dataset_id = "norm_model/storage_fail_stats"
    norm_stats_content = {"0": {"mean": [0.0]}}

    async def mock_save_stats_error(*args, **kwargs):
        raise PermissionError("Simulated write error")

    monkeypatch.setattr(test_storage_manager, "save_norm_stats", mock_save_stats_error)

    response = await async_client.post(
        f"/api/v1/datasets/{dataset_id}/norm_stats", json=norm_stats_content
    )

    assert response.status_code == 500
    assert (
        "Internal server error saving normalization stats" in response.json()["detail"]
    )


# --- Tests for GET /api/v1/datasets/{dataset_id}/batch ---
@pytest.mark.asyncio
async def test_get_batch_success(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
):
    """Test successful retrieval of a training batch."""
    dataset_id = "batch_model/batch_ds_success"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens_in_chunk = chunk_info["num_tokens"]
    d_model = chunk_info["d_model"]
    layer_indices = chunk_info["layer_indices"]
    num_tokens_to_request = 32

    # Setup: Save metadata and chunk
    metadata = {
        "dataset_stats": {
            "num_chunks": 1,
            "layer_indices": layer_indices,
            "d_model": d_model,
            "total_tokens": num_tokens_in_chunk,
            "computed_norm_stats": False,
        },
        # Add other required metadata fields if any
        "model_name": "batch_model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id, metadata)
    await test_storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_info["bytes"], num_tokens_in_chunk
    )

    # Make the request
    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch",
        params={"num_tokens": num_tokens_to_request},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) > 0

    # Deserialize and check structure
    buffer = io.BytesIO(response.content)
    batch_data = torch.load(buffer)
    assert isinstance(batch_data, dict)
    assert "inputs" in batch_data
    assert "targets" in batch_data
    assert set(batch_data["inputs"].keys()) == set(layer_indices)
    assert set(batch_data["targets"].keys()) == set(layer_indices)
    for layer_idx in layer_indices:
        assert batch_data["inputs"][layer_idx].shape == (num_tokens_to_request, d_model)
        assert batch_data["targets"][layer_idx].shape == (
            num_tokens_to_request,
            d_model,
        )
        # Check dtype if important (matches fixture)
        assert batch_data["inputs"][layer_idx].dtype == chunk_info["dtype"]


@pytest.mark.asyncio
async def test_get_batch_dataset_not_found(async_client: AsyncClient):
    """Test requesting batch from a non-existent dataset."""
    dataset_id = "nonexistent/batch_ds"
    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": 32}
    )
    assert response.status_code == 404
    # Check detail message if StorageManager raises FileNotFoundError
    assert "Dataset or required chunks not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_batch_chunk_file_missing(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
):
    """Test requesting batch when metadata exists but chunk file is missing."""
    dataset_id = "batch_model/no_chunk_file"
    # Setup: Save metadata only
    metadata = {
        "dataset_stats": {"num_chunks": 1, "layer_indices": [0], "total_tokens": 100},
        "model_name": "batch_model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id, metadata)

    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": 32}
    )
    assert response.status_code == 404
    assert "No valid HDF5 chunk files found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_batch_invalid_num_tokens(async_client: AsyncClient):
    """Test requesting batch with num_tokens <= 0."""
    dataset_id = "any/dataset"
    response_zero = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": 0}
    )
    assert response_zero.status_code == 400
    assert "'num_tokens' must be positive" in response_zero.json()["detail"]

    response_neg = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": -10}
    )
    assert response_neg.status_code == 400
    assert "'num_tokens' must be positive" in response_neg.json()["detail"]


@pytest.mark.asyncio
async def test_get_batch_invalid_layers_format(async_client: AsyncClient):
    """Test requesting batch with invalid 'layers' query parameter format."""
    dataset_id = "any/dataset"
    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch",
        params={"num_tokens": 32, "layers": "1,two,3"},
    )
    assert response.status_code == 400
    assert "Invalid format for 'layers'" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_batch_specific_layers(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
):
    """Test retrieving only specific layers in a batch."""
    dataset_id = "batch_model/specific_layers"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk  # Has layers 0, 1
    num_tokens_in_chunk = chunk_info["num_tokens"]
    d_model = chunk_info["d_model"]
    requested_layers = [1]  # Request only layer 1
    num_tokens_to_request = 16

    # Setup: Save metadata and chunk
    metadata = {
        "dataset_stats": {
            "num_chunks": 1,
            "layer_indices": chunk_info["layer_indices"],
            "d_model": d_model,
            "total_tokens": num_tokens_in_chunk,
            "computed_norm_stats": False,
        },
        "model_name": "batch_model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id, metadata)
    await test_storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_info["bytes"], num_tokens_in_chunk
    )

    # Make the request
    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch",
        params={
            "num_tokens": num_tokens_to_request,
            "layers": ",".join(map(str, requested_layers)),
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Deserialize and check structure
    buffer = io.BytesIO(response.content)
    batch_data = torch.load(buffer)
    assert "inputs" in batch_data
    assert "targets" in batch_data
    # Check ONLY requested layers are present
    assert set(batch_data["inputs"].keys()) == set(requested_layers)
    assert set(batch_data["targets"].keys()) == set(requested_layers)
    # Check shapes for the requested layer
    assert batch_data["inputs"][1].shape == (num_tokens_to_request, d_model)
    assert batch_data["targets"][1].shape == (num_tokens_to_request, d_model)


@pytest.mark.asyncio
async def test_get_batch_url_encoded_id(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
):
    """Test requesting a batch using a URL-encoded dataset ID."""
    dataset_id_plain = "batch/model/with_slash"
    dataset_id_encoded = quote(dataset_id_plain, safe="")
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens_in_chunk = chunk_info["num_tokens"]
    layer_indices = chunk_info["layer_indices"]
    num_tokens_to_request = 10

    # Setup: Save metadata and chunk using the plain ID
    metadata = {
        "dataset_stats": {
            "num_chunks": 1,
            "layer_indices": layer_indices,
            "total_tokens": num_tokens_in_chunk,
        },
        "model_name": "batch/model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id_plain, metadata)
    await test_storage_manager.save_chunk(
        dataset_id_plain, chunk_idx, chunk_info["bytes"], num_tokens_in_chunk
    )

    # Make the request using the encoded ID
    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id_encoded}/batch",
        params={"num_tokens": num_tokens_to_request},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert len(response.content) > 0

    # Quick check on deserialization
    buffer = io.BytesIO(response.content)
    batch_data = torch.load(buffer)
    assert isinstance(batch_data, dict)
    assert "inputs" in batch_data


@pytest.mark.asyncio
async def test_get_batch_storage_value_error(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
    monkeypatch,
):
    """Test batch retrieval when storage_manager.get_batch raises ValueError."""
    dataset_id = "batch_model/value_error"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens_in_chunk = chunk_info["num_tokens"]
    layer_indices = chunk_info["layer_indices"]

    # Setup: Save metadata and chunk (needed for initial checks)
    metadata = {
        "dataset_stats": {
            "num_chunks": 1,
            "layer_indices": layer_indices,
            "total_tokens": num_tokens_in_chunk,
        },
        "model_name": "batch_model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id, metadata)
    await test_storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_info["bytes"], num_tokens_in_chunk
    )

    # Mock get_batch to raise ValueError
    error_msg = "Simulated invalid data in chunk"

    async def mock_get_batch_value_error(*args, **kwargs):
        raise ValueError(error_msg)

    monkeypatch.setattr(test_storage_manager, "get_batch", mock_get_batch_value_error)

    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": 32}
    )

    assert response.status_code == 400
    assert error_msg in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_batch_storage_unexpected_error(
    async_client: AsyncClient,
    test_storage_manager: StorageManager,
    create_test_hdf5_chunk: Dict[str, Any],
    monkeypatch,
):
    """Test batch retrieval when storage_manager.get_batch raises unexpected Exception."""
    dataset_id = "batch_model/unexpected_error"
    chunk_idx = 0
    chunk_info = create_test_hdf5_chunk
    num_tokens_in_chunk = chunk_info["num_tokens"]
    layer_indices = chunk_info["layer_indices"]

    # Setup
    metadata = {
        "dataset_stats": {
            "num_chunks": 1,
            "layer_indices": layer_indices,
            "total_tokens": num_tokens_in_chunk,
        },
        "model_name": "batch_model",
        "storage_params": {"output_format": "hdf5"},
    }
    await test_storage_manager.save_metadata(dataset_id, metadata)
    await test_storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_info["bytes"], num_tokens_in_chunk
    )

    # Mock get_batch to raise generic Exception
    async def mock_get_batch_generic_error(*args, **kwargs):
        raise Exception("Something went very wrong")

    monkeypatch.setattr(test_storage_manager, "get_batch", mock_get_batch_generic_error)

    response = await async_client.get(
        f"/api/v1/datasets/{dataset_id}/batch", params={"num_tokens": 32}
    )

    assert response.status_code == 500
    assert "Internal server error retrieving batch" in response.json()["detail"]


# Note: Add tests for 500 errors by mocking the storage_manager methods later.
