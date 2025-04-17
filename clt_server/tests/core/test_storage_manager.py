import pytest
import torch
import h5py
import json
import io
from pathlib import Path
import numpy as np
import sys  # Add sys for path manipulation
import os  # Add os for path manipulation

# Ensure the project root is in the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Adjust import path based on running tests from the root project directory
# This assumes you run pytest from the `crosslayer-coding` directory
from clt_server.core.storage import StorageManager
from clt_server.core.config import settings

# --- Fixtures --- #


@pytest.fixture
def storage_manager(tmp_path: Path) -> StorageManager:
    """Provides a StorageManager instance using a temporary directory."""
    # Override the settings temporarily for the test
    original_storage_dir = settings.STORAGE_BASE_DIR
    settings.STORAGE_BASE_DIR = tmp_path / "test_server_data"
    manager = StorageManager(base_dir=settings.STORAGE_BASE_DIR)
    yield manager
    # Restore original settings after test
    settings.STORAGE_BASE_DIR = original_storage_dir


@pytest.fixture
def create_test_hdf5_chunk(tmp_path: Path) -> Path:
    """Creates a dummy HDF5 chunk file for testing get_batch."""
    chunk_path = tmp_path / "chunk_0.hdf5"
    num_tokens = 100
    d_model = 16
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
    return chunk_path


# --- Test Cases --- #


def test_storage_manager_init(tmp_path: Path):
    """Test StorageManager initialization creates the base directory."""
    base_dir = tmp_path / "init_test_dir"
    assert not base_dir.exists()
    manager = StorageManager(base_dir=base_dir)
    assert manager.base_dir == base_dir
    assert base_dir.exists()
    assert base_dir.is_dir()


def test_get_dataset_dir(storage_manager: StorageManager):
    """Test dataset directory path construction."""
    dataset_id = "test_model/test_dataset_train"
    expected_path = storage_manager.base_dir / dataset_id
    assert storage_manager.get_dataset_dir(dataset_id) == expected_path
    # Ensure it doesn't create the dir yet
    assert not expected_path.exists()


@pytest.mark.asyncio
async def test_save_and_get_metadata(storage_manager: StorageManager):
    """Test saving and retrieving metadata."""
    dataset_id = "meta_model/meta_dataset_test"
    metadata = {"config": {"model": "meta_model"}, "stats": {"num_chunks": 5}}
    dataset_dir = storage_manager.get_dataset_dir(dataset_id)
    metadata_path = dataset_dir / "metadata.json"

    # Ensure it doesn't exist first
    assert not metadata_path.exists()
    assert await storage_manager.get_dataset_metadata(dataset_id) is None

    # Save metadata
    await storage_manager.save_metadata(dataset_id, metadata)

    # Check file exists and content is correct
    assert metadata_path.exists()
    retrieved_metadata = await storage_manager.get_dataset_metadata(dataset_id)
    assert retrieved_metadata == metadata


@pytest.mark.asyncio
async def test_save_and_get_norm_stats(storage_manager: StorageManager):
    """Test saving and retrieving normalization stats."""
    dataset_id = "norm_model/norm_dataset_test"
    norm_stats = {"0": {"inputs": {"mean": [0.1], "std": [1.1]}}}
    dataset_dir = storage_manager.get_dataset_dir(dataset_id)
    norm_stats_path = dataset_dir / "norm_stats.json"

    # Ensure it doesn't exist first
    assert not norm_stats_path.exists()
    assert await storage_manager.get_norm_stats(dataset_id) is None

    # Save stats
    await storage_manager.save_norm_stats(dataset_id, norm_stats)

    # Check file exists and content is correct
    assert norm_stats_path.exists()
    retrieved_stats = await storage_manager.get_norm_stats(dataset_id)
    assert retrieved_stats == norm_stats


@pytest.mark.asyncio
async def test_save_chunk(storage_manager: StorageManager, create_test_hdf5_chunk):
    """Test saving chunk bytes (assuming they are valid HDF5)."""
    dataset_id = "chunk_model/chunk_dataset_test"
    chunk_idx = 0
    num_tokens = 100  # Should match the fixture
    chunk_path_to_read = create_test_hdf5_chunk  # Use fixture path for reading
    dataset_dir = storage_manager.get_dataset_dir(dataset_id)
    target_chunk_path = dataset_dir / f"chunk_{chunk_idx}.hdf5"

    # Read bytes from the test chunk file
    with open(chunk_path_to_read, "rb") as f:
        chunk_data = f.read()

    assert not target_chunk_path.exists()
    # Save chunk bytes
    await storage_manager.save_chunk(dataset_id, chunk_idx, chunk_data, num_tokens)

    # Check file exists and has content
    assert target_chunk_path.exists()
    assert target_chunk_path.stat().st_size > 0
    # Basic validation: try opening it
    try:
        with h5py.File(target_chunk_path, "r") as f:
            assert f.attrs["num_tokens"] == num_tokens
    except Exception as e:
        pytest.fail(f"Failed to open saved HDF5 chunk: {e}")


@pytest.mark.asyncio
async def test_list_datasets(storage_manager: StorageManager):
    """Test listing datasets based on metadata files."""
    # Create some dummy datasets
    metadata1 = {"stats": {"num_chunks": 1}}
    metadata2 = {"stats": {"num_chunks": 2}}
    dataset_id1 = "model1/ds1_train"
    dataset_id2 = "model2/ds2_test"
    dataset_id3_no_meta = "model1/ds3_val"  # Should not be listed

    await storage_manager.save_metadata(dataset_id1, metadata1)
    await storage_manager.save_metadata(dataset_id2, metadata2)
    # Create directory for dataset 3 but no metadata
    (storage_manager.get_dataset_dir(dataset_id3_no_meta)).mkdir(
        parents=True, exist_ok=True
    )

    datasets = await storage_manager.list_datasets()

    assert len(datasets) == 2
    # Order might vary, so check contents
    dataset_ids_found = {d["id"] for d in datasets}
    assert dataset_ids_found == {dataset_id1, dataset_id2}
    for d in datasets:
        if d["id"] == dataset_id1:
            assert d["metadata"] == metadata1
        elif d["id"] == dataset_id2:
            assert d["metadata"] == metadata2


@pytest.mark.asyncio
async def test_get_batch_success(
    storage_manager: StorageManager, create_test_hdf5_chunk
):
    """Test successful batch retrieval."""
    dataset_id = "batch_model/batch_dataset_train"
    num_tokens_to_sample = 32
    chunk_idx = 0
    num_tokens_in_chunk = 100  # Must match fixture
    d_model = 16  # Must match fixture

    # Create metadata for the dataset
    metadata = {
        "dataset_stats": {"num_chunks": 1, "layer_indices": [0, 1], "d_model": d_model}
    }
    await storage_manager.save_metadata(dataset_id, metadata)

    # Save the chunk using the storage manager
    chunk_path_to_read = create_test_hdf5_chunk
    with open(chunk_path_to_read, "rb") as f:
        chunk_data = f.read()
    await storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_data, num_tokens_in_chunk
    )

    # Request a batch
    batch_bytes = await storage_manager.get_batch(dataset_id, num_tokens_to_sample)

    assert isinstance(batch_bytes, bytes)
    assert len(batch_bytes) > 0

    # Deserialize and check structure
    buffer = io.BytesIO(batch_bytes)
    batch_data = torch.load(buffer)
    assert isinstance(batch_data, dict)
    assert "inputs" in batch_data
    assert "targets" in batch_data
    assert isinstance(batch_data["inputs"], dict)
    assert isinstance(batch_data["targets"], dict)

    # Check layers and shapes
    assert set(batch_data["inputs"].keys()) == {0, 1}
    assert set(batch_data["targets"].keys()) == {0, 1}
    for layer_idx in [0, 1]:
        assert batch_data["inputs"][layer_idx].shape == (num_tokens_to_sample, d_model)
        assert batch_data["targets"][layer_idx].shape == (num_tokens_to_sample, d_model)
        assert batch_data["inputs"][layer_idx].dtype == torch.float32  # Match fixture
        assert batch_data["targets"][layer_idx].dtype == torch.float32


@pytest.mark.asyncio
async def test_get_batch_specific_layers(
    storage_manager: StorageManager, create_test_hdf5_chunk
):
    """Test retrieving only specific layers in a batch."""
    dataset_id = "batch_model/batch_dataset_layers"
    num_tokens_to_sample = 10
    requested_layers = [1]  # Only request layer 1
    chunk_idx = 0
    num_tokens_in_chunk = 100
    d_model = 16

    # Metadata includes layers 0 and 1
    metadata = {
        "dataset_stats": {"num_chunks": 1, "layer_indices": [0, 1], "d_model": d_model}
    }
    await storage_manager.save_metadata(dataset_id, metadata)

    # Save the chunk
    chunk_path_to_read = create_test_hdf5_chunk
    with open(chunk_path_to_read, "rb") as f:
        chunk_data = f.read()
    await storage_manager.save_chunk(
        dataset_id, chunk_idx, chunk_data, num_tokens_in_chunk
    )

    # Request batch with specific layers
    batch_bytes = await storage_manager.get_batch(
        dataset_id, num_tokens_to_sample, layers=requested_layers
    )
    buffer = io.BytesIO(batch_bytes)
    batch_data = torch.load(buffer)

    # Check only requested layer is present
    assert set(batch_data["inputs"].keys()) == {1}
    assert set(batch_data["targets"].keys()) == {1}
    assert batch_data["inputs"][1].shape == (num_tokens_to_sample, d_model)
    assert batch_data["targets"][1].shape == (num_tokens_to_sample, d_model)


@pytest.mark.asyncio
async def test_get_batch_dataset_not_found(storage_manager: StorageManager):
    """Test requesting a batch from a non-existent dataset."""
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        await storage_manager.get_batch("nonexistent/dataset", 32)


@pytest.mark.asyncio
async def test_get_batch_metadata_not_found(storage_manager: StorageManager):
    """Test requesting batch when metadata is missing."""
    dataset_id = "no_meta/dataset"
    # Create directory but no metadata
    storage_manager.get_dataset_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Metadata not found"):
        await storage_manager.get_batch(dataset_id, 32)


@pytest.mark.asyncio
async def test_get_batch_chunk_not_found(storage_manager: StorageManager):
    """Test requesting batch when the required chunk file is missing."""
    dataset_id = "no_chunk/dataset"
    metadata = {"dataset_stats": {"num_chunks": 1, "layer_indices": [0]}}
    await storage_manager.save_metadata(dataset_id, metadata)
    # Don't save any chunk file
    with pytest.raises(FileNotFoundError, match="No valid HDF5 chunk files found"):
        await storage_manager.get_batch(dataset_id, 32)


@pytest.mark.asyncio
async def test_get_batch_empty_chunk(storage_manager: StorageManager, tmp_path: Path):
    """Test requesting batch from a chunk with 0 tokens."""
    dataset_id = "empty_chunk/dataset"
    chunk_idx = 0
    metadata = {"dataset_stats": {"num_chunks": 1, "layer_indices": [0]}}
    await storage_manager.save_metadata(dataset_id, metadata)

    # Create an HDF5 file with 0 tokens
    dataset_dir = storage_manager.get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = dataset_dir / f"chunk_{chunk_idx}.hdf5"
    with h5py.File(chunk_path, "w") as f:
        f.attrs["num_tokens"] = 0
        f.create_group("layer_0")  # Create group but no data

    with pytest.raises(ValueError, match="invalid or missing 'num_tokens' attribute"):
        await storage_manager.get_batch(dataset_id, 32)


@pytest.mark.asyncio
async def test_get_batch_invalid_num_tokens_attr(
    storage_manager: StorageManager, tmp_path: Path
):
    """Test requesting batch from chunk missing num_tokens attribute."""
    dataset_id = "invalid_chunk/dataset"
    chunk_idx = 0
    metadata = {"dataset_stats": {"num_chunks": 1, "layer_indices": [0]}}
    await storage_manager.save_metadata(dataset_id, metadata)

    # Create an HDF5 file without num_tokens attr
    dataset_dir = storage_manager.get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = dataset_dir / f"chunk_{chunk_idx}.hdf5"
    with h5py.File(chunk_path, "w") as f:
        f.create_group("layer_0")

    with pytest.raises(ValueError, match="invalid or missing 'num_tokens' attribute"):
        await storage_manager.get_batch(dataset_id, 32)
