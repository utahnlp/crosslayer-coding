"""Helper to create synthetic HDF5 activation data for testing."""

import h5py
import numpy as np
from pathlib import Path
from typing import Union, Any


def make_tiny_chunk_files(
    path: Union[str, Path],
    num_chunks: int = 1,
    n_layers: int = 2,
    n_tokens: int = 32,
    d_model: int = 8,
    dtype: Any = np.float16,
):
    """
    Creates one or more HDF5 chunk files with synthetic data.

    Args:
        path: Directory to save the chunk files in.
        num_chunks: Number of chunk files to create.
        n_layers: Number of layers.
        n_tokens: Number of tokens (rows) per chunk.
        d_model: Feature dimension.
        dtype: Numpy dtype for the data.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    for i in range(num_chunks):
        chunk_path = path / f"chunk_{i}.h5"
        with h5py.File(chunk_path, "w") as f:
            for layer_idx in range(n_layers):
                layer_group = f.create_group(f"layer_{layer_idx}")
                # "inputs" are from the source model's MLP outputs
                # "targets" are from the source model's MLP inputs
                # Both have the same shape
                shape = (n_tokens, d_model)
                inputs_data = (rng.random(size=shape) * 10).astype(dtype)
                targets_data = (rng.random(size=shape) * 5).astype(dtype)
                layer_group.create_dataset("inputs", data=inputs_data)
                layer_group.create_dataset("targets", data=targets_data)
