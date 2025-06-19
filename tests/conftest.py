import pytest
import torch
import json
import numpy as np
from pathlib import Path

from tests.helpers.tiny_configs import create_tiny_clt_config
from tests.helpers.fake_hdf5 import make_tiny_chunk_files


def get_available_devices():
    """Returns available devices, including cpu, mps, and cuda if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = get_available_devices()


@pytest.fixture(params=DEVICES)
def device(request):
    """Fixture to iterate over all available devices."""
    return torch.device(request.param)


@pytest.fixture
def tmp_local_dataset(tmp_path: Path) -> Path:
    """
    Creates a temporary local dataset directory with metadata, a manifest,
    and a dummy HDF5 chunk file for testing activation stores.
    """
    dataset_path = tmp_path / "tiny_dataset"
    dataset_path.mkdir()

    # --- Configs ---
    clt_config = create_tiny_clt_config(num_layers=2, d_model=8)
    # The tokens here must match the chunk file generation
    n_tokens_per_chunk = 32
    num_chunks = 2

    # --- Create Fake Data ---
    make_tiny_chunk_files(
        dataset_path,
        num_chunks=num_chunks,
        n_layers=clt_config.num_layers,
        n_tokens=n_tokens_per_chunk,
        d_model=clt_config.d_model,
        dtype=np.float16,
    )

    # --- Create Metadata ---
    metadata = {
        "num_layers": clt_config.num_layers,
        "d_model": clt_config.d_model,
        "total_tokens": n_tokens_per_chunk * num_chunks,
        "chunk_tokens": n_tokens_per_chunk,
        "dtype": "float16",
    }
    with open(dataset_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # --- Create Manifest (legacy 2-field format) ---
    manifest_rows = []
    for chunk_id in range(num_chunks):
        for row_id in range(n_tokens_per_chunk):
            manifest_rows.append([chunk_id, row_id])

    manifest_arr = np.array(manifest_rows, dtype=np.uint32)
    manifest_arr.tofile(dataset_path / "index.bin")

    return dataset_path
