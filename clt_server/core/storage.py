import os
import json
import torch
import io
import logging
import random  # Import random for chunk selection
from pathlib import Path
from typing import Dict, List, Optional, Any
import h5py  # Add HDF5 import
import numpy as np  # Add numpy import

from .config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages the storage and retrieval of activation datasets."""

    def __init__(self, base_dir: Path = settings.STORAGE_BASE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageManager initialized with base directory: {self.base_dir}")

    def get_dataset_dir(self, dataset_id: str) -> Path:
        """Get the directory path for a given dataset ID."""
        # Dataset ID is expected to be like model_name/dataset_name_split
        # Path() handles joining paths correctly
        return self.base_dir / dataset_id

    async def list_datasets(self) -> List[Dict[str, Any]]:
        """Lists available datasets by finding metadata.json files."""
        datasets = []
        try:
            # Iterate through model directories
            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    # Iterate through dataset_split directories
                    for dataset_dir in model_dir.iterdir():
                        if dataset_dir.is_dir():
                            metadata_path = dataset_dir / "metadata.json"
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, "r") as f:
                                        metadata = json.load(f)
                                    # Construct dataset_id from path parts
                                    dataset_id = f"{model_dir.name}/{dataset_dir.name}"
                                    datasets.append(
                                        {"id": dataset_id, "metadata": metadata}
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error reading metadata {metadata_path}: {e}"
                                    )
        except Exception as e:
            logger.error(f"Error listing datasets in {self.base_dir}: {e}")
        return datasets

    async def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the metadata for a specific dataset."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        metadata_path = dataset_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata {metadata_path}: {e}")
                return None
        return None

    async def get_norm_stats(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the normalization statistics for a specific dataset."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        norm_stats_path = dataset_dir / "norm_stats.json"
        if norm_stats_path.exists():
            try:
                with open(norm_stats_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading norm_stats {norm_stats_path}: {e}")
                return None
        return None

    async def save_chunk(
        self, dataset_id: str, chunk_idx: int, chunk_data: bytes, num_tokens: int
    ):
        """Saves a chunk of activation data (received as HDF5 bytes)."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save directly as HDF5, as sent by the generator
        chunk_filename = f"chunk_{chunk_idx}.hdf5"  # Save with .hdf5 extension
        chunk_path = dataset_dir / chunk_filename

        try:
            # Use aiofiles for async write if possible, or fallback to sync write
            try:
                import aiofiles

                async with aiofiles.open(chunk_path, "wb") as f:
                    await f.write(chunk_data)
            except ImportError:
                logger.warning(
                    "aiofiles not installed, using sync file write for chunks."
                )
                with open(chunk_path, "wb") as f:
                    f.write(chunk_data)

            logger.info(
                f"Saved chunk {chunk_idx} (HDF5) for dataset {dataset_id} to {chunk_path}"
            )
            # TODO: Potentially validate HDF5 structure here?
            # TODO: Add metadata update (e.g., record num_tokens per chunk)

        except Exception as e:
            logger.error(f"Error saving chunk {chunk_idx} to {chunk_path}: {e}")
            # Attempt cleanup
            if chunk_path.exists():
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass
            raise  # Re-raise the exception so the endpoint knows saving failed

    async def save_metadata(self, dataset_id: str, metadata: Dict[str, Any]):
        """Saves the metadata JSON file."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = dataset_dir / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata for dataset {dataset_id} to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {metadata_path}: {e}")
            raise

    async def save_norm_stats(self, dataset_id: str, norm_stats: Dict[str, Any]):
        """Saves the normalization statistics JSON file."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        norm_stats_path = dataset_dir / "norm_stats.json"
        try:
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2, default=str)
            logger.info(
                f"Saved norm_stats for dataset {dataset_id} to {norm_stats_path}"
            )
        except Exception as e:
            logger.error(f"Error saving norm_stats to {norm_stats_path}: {e}")
            raise

    async def get_batch(
        self, dataset_id: str, num_tokens: int, layers: Optional[List[int]] = None
    ) -> bytes:
        """Retrieves a random batch of activations from HDF5 chunks, serialized using torch.save."""
        dataset_dir = self.get_dataset_dir(dataset_id)
        if not dataset_dir.is_dir():  # Use is_dir() for clarity
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # 1. Load metadata
        metadata = await self.get_dataset_metadata(dataset_id)
        if not metadata:
            raise FileNotFoundError(f"Metadata not found for dataset: {dataset_id}")

        try:
            dataset_stats = metadata["dataset_stats"]
            num_chunks = dataset_stats.get("num_chunks", 0)
            available_layer_indices = dataset_stats.get("layer_indices", [])
            # Consider dtype if needed for loading?
            # dtype_str = dataset_stats.get("dtype", "torch.float32")
        except KeyError as e:
            raise ValueError(f"Metadata for {dataset_id} is missing required key: {e}")

        if num_chunks == 0:
            raise ValueError(f"Dataset {dataset_id} contains no chunks.")
        if not available_layer_indices:
            raise ValueError(
                f"Dataset {dataset_id} metadata contains no layer indices."
            )

        # 2. Select a random chunk
        selected_chunk_idx = random.randint(0, num_chunks - 1)
        chunk_path = dataset_dir / f"chunk_{selected_chunk_idx}.hdf5"
        found_chunk = False

        if chunk_path.exists():
            found_chunk = True
        else:
            logger.warning(
                f"Initial random chunk {selected_chunk_idx} not found at {chunk_path}. Trying alternatives..."
            )
            # --- Try N more random chunks ---
            tried_indices = {selected_chunk_idx}
            # Use the configured number of attempts
            for attempt in range(settings.CHUNK_RETRY_ATTEMPTS):
                if len(tried_indices) >= num_chunks:
                    # Avoid infinite loop if few chunks exist
                    break
                # Select a different random chunk
                potential_idx = random.randint(0, num_chunks - 1)
                while potential_idx in tried_indices:
                    potential_idx = random.randint(0, num_chunks - 1)
                tried_indices.add(potential_idx)

                potential_path = dataset_dir / f"chunk_{potential_idx}.hdf5"
                if potential_path.exists():
                    chunk_path = potential_path
                    selected_chunk_idx = potential_idx  # Update index for logging
                    logger.info(f"Found alternative random chunk {selected_chunk_idx}.")
                    found_chunk = True
                    break
                else:
                    logger.debug(
                        f"Alternative random chunk {potential_idx} also not found."
                    )

            # --- Fallback to sequential scan if random attempts failed ---
            if not found_chunk:
                logger.warning(
                    "Random attempts failed. Falling back to sequential scan for existing chunk..."
                )
                for i in range(num_chunks):
                    # Look for .hdf5 files
                    potential_path = dataset_dir / f"chunk_{i}.hdf5"
                    if potential_path.exists():
                        chunk_path = potential_path
                        selected_chunk_idx = i  # Update index for logging
                        logger.warning(f"Using chunk {i} found via sequential scan.")
                        found_chunk = True
                        break

        # If no chunk found after all attempts, raise error
        if not found_chunk:
            raise FileNotFoundError(
                f"No valid HDF5 chunk files found for dataset {dataset_id} after multiple attempts."
            )

        # 3. Load the selected chunk using h5py
        logger.debug(
            f"Opening chunk {chunk_path} (index {selected_chunk_idx}) for batch request..."
        )
        try:
            with h5py.File(chunk_path, "r") as hf:
                # 4. Get chunk info and Sample indices
                num_tokens_in_chunk = hf.attrs.get("num_tokens")
                if num_tokens_in_chunk is None or num_tokens_in_chunk <= 0:
                    raise ValueError(
                        f"Chunk {chunk_path.name} has invalid or missing 'num_tokens' attribute."
                    )

                actual_sample_size = min(num_tokens, num_tokens_in_chunk)
                if actual_sample_size <= 0:
                    raise ValueError(f"Cannot sample {actual_sample_size} tokens.")

                # Generate random permutation and SORT them for efficient HDF5 reading
                indices_perm = torch.randperm(num_tokens_in_chunk)
                indices = torch.sort(indices_perm[:actual_sample_size])[
                    0
                ].numpy()  # Get sorted numpy indices

                # 5. Extract data for requested layers
                batch_inputs = {}
                batch_targets = {}

                # Determine layers to extract
                layers_in_chunk_groups = [
                    name for name in hf.keys() if name.startswith("layer_")
                ]
                if not layers_in_chunk_groups:
                    raise ValueError(
                        f"Chunk {chunk_path.name} contains no layer groups."
                    )

                layers_available_in_chunk = sorted(
                    [int(g.split("_")[1]) for g in layers_in_chunk_groups]
                )
                requested_layers = (
                    layers if layers is not None else available_layer_indices
                )
                # Filter requested_layers to those actually present in this chunk
                layers_to_extract = [
                    l for l in requested_layers if l in layers_available_in_chunk
                ]

                if not layers_to_extract:
                    logger.warning(
                        f"None of the requested layers ({requested_layers}) found in chunk {chunk_path.name} layers ({layers_available_in_chunk})."
                    )
                    # Return empty dicts
                    pass

                # Read data slices using sorted indices
                for layer_idx in layers_to_extract:
                    layer_group_name = f"layer_{layer_idx}"
                    if layer_group_name in hf:
                        # Read inputs
                        if "inputs" in hf[layer_group_name]:
                            # Read slice directly into torch tensor (avoids intermediate numpy copy if possible)
                            # h5py slicing with numpy array indices returns a numpy array
                            input_slice_np = hf[layer_group_name]["inputs"][indices, :]
                            batch_inputs[layer_idx] = torch.from_numpy(input_slice_np)
                        else:
                            logger.warning(
                                f"'inputs' dataset missing for layer {layer_idx} in chunk {chunk_path.name}"
                            )

                        # Read targets
                        if "targets" in hf[layer_group_name]:
                            target_slice_np = hf[layer_group_name]["targets"][
                                indices, :
                            ]
                            batch_targets[layer_idx] = torch.from_numpy(target_slice_np)
                        else:
                            logger.warning(
                                f"'targets' dataset missing for layer {layer_idx} in chunk {chunk_path.name}"
                            )
                    else:
                        logger.warning(
                            f"Layer group '{layer_group_name}' not found in chunk {chunk_path.name}"
                        )

            # 6. Construct the batch dictionary
            batch_dict = {"inputs": batch_inputs, "targets": batch_targets}

            # 7. Serialize the batch dictionary using torch.save
            output_buffer = io.BytesIO()
            torch.save(batch_dict, output_buffer)
            output_buffer.seek(0)
            batch_bytes = output_buffer.read()
            logger.debug(f"Serialized batch of size {len(batch_bytes)} bytes.")

            # 8. Return the serialized bytes
            return batch_bytes

        except FileNotFoundError:  # Should be caught earlier, but for safety
            raise
        except ValueError as e:
            logger.error(
                f"Value error processing chunk {chunk_path.name}: {e}", exc_info=True
            )
            raise  # Re-raise specific value errors
        except KeyError as e:
            logger.error(
                f"Data structure error in HDF5 chunk {chunk_path.name}: Missing key {e}",
                exc_info=True,
            )
            raise ValueError(f"Corrupted data format in chunk: {chunk_path.name}")
        except IndexError as e:  # Should be less likely with h5py slicing, but possible
            logger.error(
                f"Indexing error during sampling from chunk {chunk_path.name}: {e}",
                exc_info=True,
            )
            raise ValueError(f"Error sampling data from chunk: {chunk_path.name}")
        except Exception as e:
            logger.error(
                f"Unexpected error processing HDF5 chunk {chunk_path.name} for batch request: {e}",
                exc_info=True,
            )
            raise ValueError(f"Failed to process data from chunk: {chunk_path.name}")


# Create a single instance of the storage manager
storage_manager = StorageManager()
