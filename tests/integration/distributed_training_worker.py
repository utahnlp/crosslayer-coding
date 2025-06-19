"""
This script is designed to be launched by torchrun for distributed training tests.
It initializes a trainer, runs a few steps, and saves its final model state.
The calling test can then inspect the saved states for consistency.
"""

import sys
from pathlib import Path

# Ensure the project root is in the python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import os
import argparse
import json
import numpy as np

from clt.training.trainer import CLTTrainer
from tests.helpers.tiny_configs import create_tiny_clt_config, create_tiny_training_config
from tests.helpers.fake_hdf5 import make_tiny_chunk_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save final model states.")
    args = parser.parse_args()

    # --- Get Distributed Info ---
    rank = int(os.environ.get("RANK", "0"))

    # --- Create a temporary dataset for this run (shared across ranks) ---
    dataset_path = Path(args.output_dir) / "test_dataset"

    # Parameters for tiny dataset
    num_chunks = 2
    num_layers = 2
    d_model = 8
    n_tokens_per_chunk = 64

    # Let every rank attempt to create the dataset; implementation is idempotent
    make_tiny_chunk_files(
        path=dataset_path,
        num_chunks=num_chunks,
        n_layers=num_layers,
        n_tokens=n_tokens_per_chunk,
        d_model=d_model,
    )

    # Rank 0 writes metadata and manifest (lightweight JSON/bin files)
    if rank == 0:
        metadata = {
            "num_layers": num_layers,
            "d_model": d_model,
            "total_tokens": n_tokens_per_chunk * num_chunks,
            "chunk_tokens": n_tokens_per_chunk,
            "dtype": "float16",
        }
        (dataset_path / "metadata.json").write_text(json.dumps(metadata))

        # Create simple legacy 2-field manifest
        manifest_rows = []
        for cid in range(num_chunks):
            for rid in range(n_tokens_per_chunk):
                manifest_rows.append([cid, rid])
        manifest_arr = np.asarray(manifest_rows, dtype=np.uint32)
        manifest_arr.tofile(dataset_path / "index.bin")

    # --- Configuration ---
    clt_config = create_tiny_clt_config(num_layers=2, d_model=8, num_features=16)

    # When distributed=True, the trainer correctly sets shard_data=False for the activation store,
    # which is required for tensor parallelism.
    training_config = create_tiny_training_config(
        training_steps=5,
        train_batch_size_tokens=16,
        activation_source="local_manifest",
        activation_path=str(dataset_path),
        activation_dtype="float32",
        precision="fp32",
    )

    # --- Initialize and run trainer ---
    # The trainer will automatically initialize the process group based on env variables.
    trainer = CLTTrainer(
        clt_config=clt_config,
        training_config=training_config,
        log_dir=str(Path(args.output_dir) / f"rank_{rank}_logs"),
        distributed=True,
    )

    trainer.train()

    # --- Save final model state for verification ---
    output_path = Path(args.output_dir) / f"rank_{rank}_final_model.pt"
    torch.save(trainer.model.state_dict(), output_path)

    print(f"Rank {rank} finished and saved model to {output_path}")


if __name__ == "__main__":
    main()
