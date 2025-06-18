import pytest
import torch
import numpy as np
import h5py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Make sure clt is discoverable
import sys
import os

# This assumes the test is run from the project root
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from clt.activation_generation.generator import ActivationGenerator, _RunningStat
from clt.config.data_config import ActivationConfig

# --- Fixtures ---


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for activation generation output."""
    return tmp_path / "test_activations"


@pytest.fixture(params=[("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def mock_extractor(request):
    """
    Fixture to mock the ActivationExtractorCLT.
    Parametrized to test different dtypes.
    """
    dtype_str, torch_dtype = request.param

    # --- Create Mock Data ---
    d_model = 8
    num_layers = 2
    tokens_per_batch = 100
    num_batches = 3

    mock_batches = []
    for i in range(num_batches):
        base_data = torch.arange(i * tokens_per_batch, (i + 1) * tokens_per_batch, dtype=torch.float32).unsqueeze(1)
        batch_tensor = base_data.expand(-1, d_model)

        inputs = {layer: batch_tensor.clone().to(torch_dtype) for layer in range(num_layers)}
        targets = {layer: (batch_tensor.clone() + 1.0).to(torch_dtype) for layer in range(num_layers)}
        mock_batches.append((inputs, targets))

    # --- Create Mock Extractor ---
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.stream_activations.return_value = iter(mock_batches)
    mock_extractor_instance.model_name = "mock_model"
    mock_extractor_instance.d_model = d_model
    mock_extractor_instance.layer_ids = list(range(num_layers))

    with patch(
        "clt.activation_generation.generator.ActivationExtractorCLT", return_value=mock_extractor_instance
    ) as mock_class:
        yield mock_class, {
            "d_model": d_model,
            "num_layers": num_layers,
            "tokens_per_batch": tokens_per_batch,
            "num_batches": num_batches,
            "total_tokens": tokens_per_batch * num_batches,
            "dtype_str": dtype_str,
            "torch_dtype": torch_dtype,
            "mock_batches": mock_batches,  # For manual stat calculation
        }


@pytest.fixture(params=["hdf5", "npz"])
def activation_config(request, temp_output_dir, mock_extractor):
    """
    Create a default ActivationConfig for tests.
    Parametrized for different output formats.
    """
    _, mock_data_info = mock_extractor
    return ActivationConfig(
        model_name="mock_model",
        mlp_input_module_path_template="mock.path.in.{}",
        mlp_output_module_path_template="mock.path.out.{}",
        dataset_path="mock_dataset",
        activation_dir=str(temp_output_dir),
        chunk_token_threshold=150,
        activation_dtype=mock_data_info["dtype_str"],
        output_format=request.param,
        compute_norm_stats=True,
        compression=None,
    )


# --- Helper Functions ---


def load_chunk_data(path, num_layers, torch_dtype):
    """Helper to load data from either HDF5 or NPZ chunk."""
    if path.suffix == ".hdf5":
        with h5py.File(path, "r") as hf:
            data = {}
            for i in range(num_layers):
                data[f"inputs_{i}"] = torch.from_numpy(hf[f"layer_{i}/inputs"][:])
                data[f"targets_{i}"] = torch.from_numpy(hf[f"layer_{i}/targets"][:])
    elif path.suffix == ".npz":
        with np.load(path) as npz:
            data = {}
            for i in range(num_layers):
                data[f"inputs_{i}"] = torch.from_numpy(npz[f"layer_{i}_inputs"])
                data[f"targets_{i}"] = torch.from_numpy(npz[f"layer_{i}_targets"])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Handle bfloat16 conversion if necessary
    if torch_dtype == torch.bfloat16:
        for key in data:
            data[key] = data[key].view(torch.uint16).view(torch.bfloat16)
    return data


# --- Test Cases ---


@pytest.mark.parametrize("enable_profiling", [True, False])
def test_shared_permutation_across_layers_and_chunks(activation_config, mock_extractor, enable_profiling):
    """
    Tests that:
    1. A single permutation is applied to all layers within a chunk.
    2. Different chunks use different permutations.
    3. The `enable_profiling` flag does not change the output.
    """
    # Arrange
    activation_config.enable_profiling = enable_profiling
    mock_extractor_class, mock_data_info = mock_extractor
    generator = ActivationGenerator(cfg=activation_config, device="cpu")

    # Act
    generator.generate_and_save()

    # Assert
    output_path = Path(activation_config.activation_dir) / "mock_model" / "mock_dataset_train"
    chunk_paths = sorted(output_path.glob(f"chunk_*.{activation_config.output_format}"))
    assert len(chunk_paths) == 2  # Based on fixture config

    # --- Verify alignment within each chunk and uniqueness between chunks ---
    previous_chunk_first_row = None
    for chunk_path in chunk_paths:
        chunk_data = load_chunk_data(chunk_path, mock_data_info["num_layers"], mock_data_info["torch_dtype"])

        # 1. Verify alignment within the chunk
        for i in range(1, mock_data_info["num_layers"]):
            torch.testing.assert_close(
                chunk_data["inputs_0"], chunk_data[f"inputs_{i}"], msg=f"Input alignment failed in {chunk_path.name}"
            )
            torch.testing.assert_close(
                chunk_data["targets_0"], chunk_data[f"targets_{i}"], msg=f"Target alignment failed in {chunk_path.name}"
            )

        # 2. Verify uniqueness of permutation between chunks
        current_chunk_first_row = chunk_data["inputs_0"][0]
        if previous_chunk_first_row is not None:
            # The mock data is sequential, so if permutations are different, first rows should be different.
            assert not torch.allclose(
                previous_chunk_first_row, current_chunk_first_row
            ), "Permutation seems to be reused across chunks."
        previous_chunk_first_row = current_chunk_first_row


def test_metadata_and_manifest_generation(activation_config, mock_extractor):
    """
    Tests that metadata.json is correct and index.bin has consistent offsets and token counts.
    """
    # Arrange
    mock_extractor_class, mock_data_info = mock_extractor
    generator = ActivationGenerator(cfg=activation_config, device="cpu")

    # Act
    generator.generate_and_save()

    # Assert
    output_path = Path(activation_config.activation_dir) / "mock_model" / "mock_dataset_train"
    metadata_path = output_path / "metadata.json"
    manifest_path = output_path / "index.bin"
    assert metadata_path.exists() and manifest_path.exists()

    # --- Test metadata.json ---
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    assert meta["total_tokens"] == mock_data_info["total_tokens"]
    assert meta["chunk_tokens"] == activation_config.chunk_token_threshold
    assert meta["dtype"] == mock_data_info["dtype_str"]

    # --- Test index.bin (manifest) ---
    manifest_dtype = np.dtype([("chunk_id", np.int32), ("num_tokens", np.int32), ("offset", np.int64)])
    manifest = np.fromfile(manifest_path, dtype=manifest_dtype)

    assert len(manifest) == 2
    assert manifest["num_tokens"].sum() == mock_data_info["total_tokens"]
    # Check for monotonic increase in offsets and consistency
    for i in range(len(manifest) - 1):
        assert manifest["offset"][i + 1] > manifest["offset"][i], "Offsets are not monotonically increasing."
        assert (
            manifest["offset"][i] + manifest["num_tokens"][i] == manifest["offset"][i + 1]
        ), "Offset inconsistency between chunks."


def test_norm_stats_correctness(activation_config, mock_extractor):
    """
    Tests that the normalization statistics are computed correctly.
    """
    # Arrange
    # Disable shuffling to make manual calculation straightforward
    activation_config.chunk_token_threshold = 1_000_000  # Ensure one big chunk
    mock_extractor_class, mock_data_info = mock_extractor
    generator = ActivationGenerator(cfg=activation_config, device="cpu")

    # Act
    generator.generate_and_save()

    # Assert
    output_path = Path(activation_config.activation_dir) / "mock_model" / "mock_dataset_train"
    norm_stats_path = output_path / "norm_stats.json"
    assert norm_stats_path.exists()

    # --- Manually compute stats for comparison ---
    all_inputs = torch.cat([b[0][0] for b in mock_data_info["mock_batches"]], dim=0).to(torch.float32)
    manual_mean = all_inputs.mean(dim=0)
    manual_std = all_inputs.std(dim=0)

    # --- Load generated stats and compare ---
    with open(norm_stats_path, "r") as f:
        gen_stats = json.load(f)

    gen_mean = torch.tensor(gen_stats["0"]["inputs"]["mean"])
    gen_std = torch.tensor(gen_stats["0"]["inputs"]["std"])

    torch.testing.assert_close(gen_mean, manual_mean, rtol=1e-5, atol=1e-5)
    # The generator uses Welford's algorithm, which can have minor precision differences.
    torch.testing.assert_close(gen_std, manual_std, rtol=1e-5, atol=1e-5)
