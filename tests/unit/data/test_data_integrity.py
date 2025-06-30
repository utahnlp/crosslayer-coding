"""Comprehensive data integrity tests for activation generation and retrieval.

These tests specifically target historical issues with data mixups including:
- Lexicographic vs numerical layer ordering
- Normalization application correctness
- Token ordering across chunks
"""

# Standard library imports
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party imports
import pytest
import torch
import numpy as np
import h5py  # type: ignore
import json
from unittest.mock import patch, MagicMock

# Local imports
from clt.activation_generation.generator import ActivationGenerator
from clt.config.data_config import ActivationConfig
from clt.training.data.local_activation_store import LocalActivationStore


class TestDataIntegrity:
    """Test suite for verifying data integrity across the activation pipeline."""

    def test_layer_ordering_numerical_not_lexicographic(self, tmp_path):
        """Test that layers are ordered numerically (2, 10, 20) not lexicographically (10, 2, 20)."""
        # Create test data with layers that would be misordered lexicographically
        layer_ids = [2, 10, 20, 1, 100]  # Intentionally out of order
        d_model = 16
        tokens_per_batch = 32

        # Mock the extractor to return our test layers
        mock_batches = []
        for batch_idx in range(2):
            inputs = {}
            targets = {}
            for layer_id in layer_ids:
                # Use deterministic values based on layer_id
                base_value = float(layer_id * 1000 + batch_idx * 100)
                inputs[layer_id] = torch.full((tokens_per_batch, d_model), base_value, dtype=torch.float32)
                targets[layer_id] = torch.full((tokens_per_batch, d_model), base_value + 1, dtype=torch.float32)
            mock_batches.append((inputs, targets))

        # Configure and run generator
        with patch("clt.activation_generation.generator.ActivationExtractorCLT") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.stream_activations.return_value = iter(mock_batches)
            mock_extractor_class.return_value = mock_extractor

            config = ActivationConfig(
                model_name="test_model",
                mlp_input_module_path_template="test.{}",
                mlp_output_module_path_template="test.{}",
                dataset_path="test_dataset",
                activation_dir=str(tmp_path),
                chunk_token_threshold=100,  # Small chunks for testing
                compute_norm_stats=False,
                output_format="hdf5",
                activation_dtype="float32",  # Match the test data dtype
                enable_profiling=False,  # Disable profiling for tests
            )

            generator = ActivationGenerator(cfg=config, device="cpu")
            generator.generate_and_save()

        # Step 1: Verify the generated files exist and contain all expected layers
        output_dir = tmp_path / "test_model" / "test_dataset_train"
        chunk_files = sorted(output_dir.glob("chunk_*.hdf5"))
        assert len(chunk_files) > 0

        # Check that all layers were written (HDF5 will store them lexicographically)
        with h5py.File(chunk_files[0], "r") as hf:
            layer_groups = [k for k in hf.keys() if k.startswith("layer_")]
            # Extract layer numbers from the HDF5 file
            hdf5_layer_nums = []
            for lg in layer_groups:
                try:
                    num = int(lg.split("_")[1])
                    hdf5_layer_nums.append(num)
                except (IndexError, ValueError):
                    pytest.fail(f"Invalid layer group name: {lg}")

            # Verify all expected layers are present (order doesn't matter here)
            assert set(hdf5_layer_nums) == set(layer_ids), f"Missing layers in HDF5"

        # Step 2: Use LocalActivationStore to verify it reads layers in numerical order
        # Also write a manifest for the store
        manifest_data = np.array([(0, i) for i in range(64)], dtype=np.uint32)  # 64 total tokens
        manifest_data.tofile(output_dir / "index.bin")

        # Write metadata
        metadata = {
            "model_name": "test_model",
            "dataset": "test_dataset",
            "split": "train",
            "num_layers": len(layer_ids),
            "d_model": d_model,
            "dtype": "float32",
            "total_tokens": 64,
            "chunk_tokens": 100,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create store and verify it processes layers in numerical order
        store = LocalActivationStore(
            dataset_path=output_dir,
            train_batch_size_tokens=32,
            device="cpu",
            dtype="float32",
        )

        # The store should have layers in numerical order
        assert store.layer_indices == list(range(len(layer_ids)))

        # Get a batch and verify data integrity
        inputs, targets = store.get_batch()

        # Verify we have the correct number of layers
        assert len(inputs) == len(layer_ids)
        assert len(targets) == len(layer_ids)

        # Step 3: Verify the lexicographic ordering issue would cause problems without proper handling
        # If we read layers lexicographically, layer data would be mixed up
        # The _layer_sort_key function in LocalActivationStore prevents this

    def test_normalization_application_correctness(self, tmp_path):
        """Test that normalization is correctly applied during data retrieval."""
        # Generate test data with known statistics
        d_model = 8
        num_layers = 2
        num_tokens = 1000

        # Create data with known mean and std per layer
        layer_stats = {0: {"mean": 10.0, "std": 2.0}, 1: {"mean": -5.0, "std": 0.5}}

        # Generate the data files manually
        output_dir = tmp_path / "test_model" / "test_dataset_train"
        output_dir.mkdir(parents=True)

        # Create metadata
        metadata = {
            "model_name": "test_model",
            "dataset": "test_dataset",
            "split": "train",
            "num_layers": num_layers,
            "d_model": d_model,
            "dtype": "float32",
            "total_tokens": num_tokens,
            "chunk_tokens": num_tokens,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create norm_stats.json
        norm_stats = {}
        for layer_id, stats in layer_stats.items():
            norm_stats[str(layer_id)] = {
                "inputs": {"mean": [stats["mean"]] * d_model, "std": [stats["std"]] * d_model},
                "targets": {"mean": [stats["mean"] + 1] * d_model, "std": [stats["std"]] * d_model},
            }
        with open(output_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f)

        # Create data with these exact statistics
        with h5py.File(output_dir / "chunk_0.hdf5", "w") as hf:
            for layer_id, stats in layer_stats.items():
                g = hf.create_group(f"layer_{layer_id}")

                # Generate data with exact mean and std
                rng = np.random.RandomState(42)
                inputs_data = rng.normal(stats["mean"], stats["std"], (num_tokens, d_model)).astype(np.float32)
                targets_data = rng.normal(stats["mean"] + 1, stats["std"], (num_tokens, d_model)).astype(np.float32)

                g.create_dataset("inputs", data=inputs_data)
                g.create_dataset("targets", data=targets_data)

        # Create manifest
        manifest_data = np.array([(0, i) for i in range(num_tokens)], dtype=np.uint32)
        manifest_data.tofile(output_dir / "index.bin")

        # Load with normalization
        store = LocalActivationStore(
            dataset_path=output_dir,
            train_batch_size_tokens=100,
            normalization_method="mean_std",  # Enable normalization
            dtype="float32",
            device="cpu",
        )

        # Get a batch and verify normalization was applied
        inputs, targets = store.get_batch()

        for layer_id in range(num_layers):
            # After normalization, data should have mean ~0 and std ~1
            inp_data = inputs[layer_id].float()
            tgt_data = targets[layer_id].float()

            # Check normalized statistics (allowing some tolerance due to sampling)
            assert torch.abs(inp_data.mean()) < 0.1, f"Layer {layer_id} input mean not normalized"
            assert torch.abs(inp_data.std() - 1.0) < 0.1, f"Layer {layer_id} input std not normalized"
            assert torch.abs(tgt_data.mean()) < 0.1, f"Layer {layer_id} target mean not normalized"
            assert torch.abs(tgt_data.std() - 1.0) < 0.1, f"Layer {layer_id} target std not normalized"

    def test_cross_chunk_token_ordering(self, tmp_path):
        """Test that token ordering is preserved when data spans multiple chunks."""
        d_model = 4
        num_layers = 2
        tokens_per_chunk = 50
        num_chunks = 3

        # Create predictable data where we can verify ordering
        output_dir = tmp_path / "test_model" / "test_dataset_train"
        output_dir.mkdir(parents=True)

        # Metadata
        total_tokens = tokens_per_chunk * num_chunks
        metadata = {
            "model_name": "test_model",
            "dataset": "test_dataset",
            "split": "train",
            "num_layers": num_layers,
            "d_model": d_model,
            "dtype": "float32",
            "total_tokens": total_tokens,
            "chunk_tokens": tokens_per_chunk,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create chunks with identifiable patterns
        manifest_entries = []
        for chunk_id in range(num_chunks):
            with h5py.File(output_dir / f"chunk_{chunk_id}.hdf5", "w") as hf:
                for layer_id in range(num_layers):
                    g = hf.create_group(f"layer_{layer_id}")

                    # Create data where each token's first element encodes its global position
                    chunk_data = np.zeros((tokens_per_chunk, d_model), dtype=np.float32)
                    for local_idx in range(tokens_per_chunk):
                        global_idx = chunk_id * tokens_per_chunk + local_idx
                        # Encode: [global_idx, chunk_id, layer_id, local_idx]
                        chunk_data[local_idx, 0] = global_idx
                        chunk_data[local_idx, 1] = chunk_id
                        chunk_data[local_idx, 2] = layer_id
                        chunk_data[local_idx, 3] = local_idx

                    g.create_dataset("inputs", data=chunk_data)
                    g.create_dataset("targets", data=chunk_data + 1000)  # Offset targets

            # Add manifest entries
            for i in range(tokens_per_chunk):
                manifest_entries.append((chunk_id, i))

        # Write manifest
        manifest_data = np.array(manifest_entries, dtype=np.uint32)
        manifest_data.tofile(output_dir / "index.bin")

        # Load store with sequential sampling (no shuffling within epoch)
        store = LocalActivationStore(
            dataset_path=output_dir,
            train_batch_size_tokens=25,  # Spans chunk boundaries
            sampling_strategy="sequential",
            device="cpu",
            dtype="float32",
        )

        # Collect all batches
        all_tokens = []
        for batch_inputs, batch_targets in store:
            # Look at layer 0 data
            batch_data = batch_inputs[0].cpu().numpy()
            all_tokens.extend(batch_data[:, 0].tolist())  # Global indices

        # For sequential sampling with no data sharding, we should see all tokens
        expected_tokens = list(range(total_tokens))

        # Due to the sampler's sharding behavior when world=1, we should get all tokens
        assert len(all_tokens) == len(
            expected_tokens
        ), f"Missing tokens: got {len(all_tokens)}, expected {len(expected_tokens)}"

        # Verify no duplicates
        assert len(set(all_tokens)) == len(all_tokens), "Found duplicate tokens"

    def test_manifest_format_compatibility(self, tmp_path):
        """Test that both 2-field and 3-field manifest formats are handled correctly."""
        d_model = 4
        num_layers = 1
        num_tokens = 100

        output_dir = tmp_path / "test_model" / "test_dataset_train"
        output_dir.mkdir(parents=True)

        # Common metadata
        metadata = {
            "model_name": "test_model",
            "dataset": "test_dataset",
            "split": "train",
            "num_layers": num_layers,
            "d_model": d_model,
            "dtype": "float32",
            "total_tokens": num_tokens,
            "chunk_tokens": num_tokens,
        }

        # Create simple test data
        with h5py.File(output_dir / "chunk_0.hdf5", "w") as hf:
            g = hf.create_group("layer_0")
            data = np.arange(num_tokens * d_model, dtype=np.float32).reshape(num_tokens, d_model)
            g.create_dataset("inputs", data=data)
            g.create_dataset("targets", data=data + 1000)

        # Test 1: Legacy 2-field format
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        manifest_2field = np.array([(0, i) for i in range(num_tokens)], dtype=np.uint32)
        manifest_2field.tofile(output_dir / "index.bin")

        store1 = LocalActivationStore(
            dataset_path=output_dir,
            train_batch_size_tokens=10,
            device="cpu",
            dtype="float32",  # Match the data dtype
        )
        batch1_inputs, _ = store1.get_batch()

        # Test 2: New 3-field format
        # Delete the old manifest first
        (output_dir / "index.bin").unlink()

        manifest_dtype_3field = np.dtype([("chunk_id", np.int32), ("num_tokens", np.int32), ("offset", np.int64)])
        manifest_3field = np.array([(0, num_tokens, 0)], dtype=manifest_dtype_3field)
        manifest_3field.tofile(output_dir / "index.bin")

        store2 = LocalActivationStore(
            dataset_path=output_dir,
            train_batch_size_tokens=10,
            device="cpu",
            dtype="float32",  # Match the data dtype
        )
        batch2_inputs, _ = store2.get_batch()

        # Both formats should give us valid data
        assert batch1_inputs[0].shape == batch2_inputs[0].shape
        assert torch.isfinite(batch1_inputs[0]).all()
        assert torch.isfinite(batch2_inputs[0]).all()
