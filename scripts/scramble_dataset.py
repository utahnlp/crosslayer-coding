#!/usr/bin/env python3
"""
Script to scramble an existing locally stored activation dataset.

Reads an existing dataset (HDF5 chunks + manifest), shuffles all rows globally,
and writes a new dataset with the scrambled data and a corrected manifest.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import h5py
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- HDF5 Helper (adapted from generator) ---
def _create_datasets(
    hf: h5py.File,
    layer_ids: List[int],
    rows: int,
    d: int,
    h5py_dtype: str = "float16",
):
    """Create datasets within an HDF5 file for a chunk."""
    for lid in layer_ids:
        g = hf.create_group(f"layer_{lid}")
        for name in ("inputs", "targets"):
            try:
                g.create_dataset(
                    name,
                    shape=(rows, d),
                    dtype=h5py_dtype,
                    chunks=(1, d),  # Row-chunking is good for sequential reads later
                    compression="gzip",
                    compression_opts=2,
                )
            except Exception as e:
                logger.error(f"Failed to create dataset {name} for layer {lid}: {e}")
                raise


def _get_h5py_dtype(torch_dtype_str: str) -> str:
    """Maps torch dtype string to h5py compatible dtype string."""
    if torch_dtype_str == "float32":
        return "float32"
    elif torch_dtype_str == "float16":
        return "float16"
    elif torch_dtype_str == "bfloat16":
        # Store as uint16, requires conversion on read by client
        return "uint16"
    else:
        raise ValueError(f"Unsupported activation dtype for HDF5 storage: {torch_dtype_str}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scramble an existing local CLT activation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input dataset directory (containing index.bin, metadata.json, chunk_*.h5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory where the scrambled dataset will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for the permutation. If None, uses system time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        # Basic check if output dir exists and is not empty
        logger.error(
            f"Output directory {output_dir} exists and is not empty. Please remove or choose a different path."
        )
        return

    # --- Load Metadata ---
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"metadata.json not found in {input_dir}")
        return
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        # Extract necessary info
        num_layers = int(metadata["num_layers"])
        d_model = int(metadata["d_model"])
        # Use original 'chunk_tokens' as the target size for new chunks
        chunk_size = int(metadata["chunk_tokens"])
        activation_dtype_str = metadata.get("dtype", "bfloat16")  # Default if missing
        h5_dtype = _get_h5py_dtype(activation_dtype_str)
        layer_ids = list(range(num_layers))  # Assume simple 0..N-1 layer indexing
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error reading metadata.json: {e}")
        return

    # --- Load Original Manifest ---
    manifest_path = input_dir / "index.bin"
    if not manifest_path.exists():
        logger.error(f"index.bin not found in {input_dir}")
        return
    try:
        original_manifest = np.fromfile(manifest_path, dtype=np.uint32).reshape(-1, 2)
        total_rows = len(original_manifest)
        if total_rows == 0:
            logger.error("Original manifest is empty.")
            return
        logger.info(f"Loaded original manifest ({total_rows} rows) from {manifest_path}")
    except Exception as e:
        logger.error(f"Error reading or reshaping index.bin: {e}")
        return

    # --- Prepare Output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Copy metadata and norm_stats (if exists)
    try:
        with open(output_dir / "metadata.json", "w") as f:
            # Update creation time in copied metadata
            metadata["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["scrambled_from"] = str(input_dir)  # Add provenance
            metadata["scramble_seed"] = args.seed
            json.dump(metadata, f, indent=2)

        norm_stats_path = input_dir / "norm_stats.json"
        if norm_stats_path.exists():
            with open(norm_stats_path, "r") as f_in, open(output_dir / "norm_stats.json", "w") as f_out:
                norm_data = json.load(f_in)
                json.dump(norm_data, f_out)
            logger.info("Copied norm_stats.json")
        else:
            logger.info("norm_stats.json not found in input, skipping copy.")
    except Exception as e:
        logger.error(f"Error copying metadata/norm_stats: {e}")
        return

    # --- Generate Permutation and Shuffle Original Manifest ---
    seed = args.seed if args.seed is not None else int(time.time())
    logger.info(f"Using random seed: {seed}")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_rows)

    # Shuffle the original manifest rows according to the permutation
    # shuffled_manifest[i] gives the (original_chunk, original_row) for the i-th new global row
    shuffled_manifest = original_manifest[permutation]
    logger.info("Generated global permutation and shuffled original manifest.")

    # --- Process Rows and Write New Chunks/Manifest ---
    new_manifest_data = np.zeros((total_rows, 2), dtype=np.uint32)
    output_h5_files: Dict[int, h5py.File] = {}  # Cache open output files

    logger.info(f"Starting row processing and writing {total_rows} rows...")
    try:
        # Use LRU cache for input files implicitly via _open_h5 logic if adapted/imported
        # Simple dictionary cache for input files for this script
        input_h5_cache: Dict[int, h5py.File] = {}

        def get_input_h5(chunk_id: int) -> h5py.File:
            if chunk_id not in input_h5_cache:
                path = input_dir / f"chunk_{chunk_id}.h5"
                if not path.exists():
                    raise FileNotFoundError(f"Input chunk file not found: {path}")
                try:
                    input_h5_cache[chunk_id] = h5py.File(path, "r")
                except OSError as e:
                    raise OSError(f"Failed to open input chunk {path}: {e}") from e
            return input_h5_cache[chunk_id]

        for i in tqdm(range(total_rows), desc="Scrambling rows"):
            # Determine original location from shuffled manifest
            original_chunk_id, original_row_id = shuffled_manifest[i]

            # Determine new location
            new_chunk_id = i // chunk_size
            new_row_id = i % chunk_size

            # Store in the new manifest
            new_manifest_data[i, 0] = new_chunk_id
            new_manifest_data[i, 1] = new_row_id

            # Ensure output HDF5 file is open and datasets exist
            if new_chunk_id not in output_h5_files:
                output_chunk_path = output_dir / f"chunk_{new_chunk_id}.h5"
                try:
                    # Calculate rows for this *specific* new chunk
                    start_idx = new_chunk_id * chunk_size
                    end_idx = min(start_idx + chunk_size, total_rows)
                    rows_in_new_chunk = end_idx - start_idx

                    hf_out = h5py.File(output_chunk_path, "w", libver="latest")
                    _create_datasets(hf_out, layer_ids, rows_in_new_chunk, d_model, h5py_dtype=h5_dtype)
                    output_h5_files[new_chunk_id] = hf_out
                    logger.debug(f"Created output chunk {new_chunk_id} with {rows_in_new_chunk} rows.")
                except Exception as e:
                    logger.error(f"Failed to create or open output chunk file {output_chunk_path}: {e}")
                    raise  # Propagate error

            hf_out = output_h5_files[new_chunk_id]

            # Read data from original chunk
            try:
                hf_in = get_input_h5(original_chunk_id)
                # Read data for all layers for this row
                input_row_data: Dict[int, np.ndarray] = {}
                target_row_data: Dict[int, np.ndarray] = {}
                for lid in layer_ids:
                    layer_key = f"layer_{lid}"
                    # Read the single row (returns shape [1, d_model])
                    inp_data = hf_in[layer_key]["inputs"][original_row_id, :]
                    tgt_data = hf_in[layer_key]["targets"][original_row_id, :]
                    input_row_data[lid] = inp_data
                    target_row_data[lid] = tgt_data

            except (FileNotFoundError, OSError, KeyError) as e:
                logger.error(f"Error reading row {original_row_id} from original chunk {original_chunk_id}: {e}")
                raise  # Propagate error

            # Write data to new chunk
            try:
                for lid in layer_ids:
                    layer_key = f"layer_{lid}"
                    # Write the single row data to the new position
                    hf_out[layer_key]["inputs"][new_row_id, :] = input_row_data[lid]
                    hf_out[layer_key]["targets"][new_row_id, :] = target_row_data[lid]
            except Exception as e:
                logger.error(
                    f"Error writing row {new_row_id} to new chunk {new_chunk_id} (original: {original_chunk_id},{original_row_id}): {e}"
                )
                raise  # Propagate error

        # --- Finalize ---
        logger.info("Saving new manifest file...")
        new_manifest_path = output_dir / "index.bin"
        new_manifest_data.tofile(new_manifest_path)

    except Exception as e:
        logger.exception(f"An error occurred during scrambling: {e}")
        logger.error("Scrambling failed. Output directory may be incomplete or corrupted.")
    finally:
        # Ensure all HDF5 files are closed
        logger.info("Closing HDF5 files...")
        for hf in output_h5_files.values():
            try:
                hf.close()
            except Exception as e:
                logger.warning(f"Error closing output HDF5 file: {e}")
        for hf in input_h5_cache.values():
            try:
                hf.close()
            except Exception as e:
                logger.warning(f"Error closing input HDF5 file: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Scrambling finished in {elapsed_time:.2f} seconds.")
    logger.info(f"Scrambled dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
