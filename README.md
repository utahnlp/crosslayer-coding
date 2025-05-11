# Cross-Layer Coding

This library is intended for the training and analysis of cross-layer sparse coding models, including the Cross-Layer Transcoder as described in "[Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)"

Currently, that is the only type of architecture supported, but in the future this will support skip-transcoders and other variants.

## Overview

A Cross-Layer Transcoder (CLT) is a multi-layer dictionary learning model designed to extract sparse, interpretable features from transformers, using an encoder for each layer and a decoder for each (source layer, destination layer) pair (e.g., 12 encoders and 78 decoders for `gpt2-small`). This implementation focuses on the core functionality needed to train and use CLTs, leveraging `nnsight` for model introspection and `datasets` for data handling.

## Installation

```bash
# Ensure you have Python 3.8+ and pip installed
git clone https://github.com/curt-tigges/crosslayer-coding.git

cd crosslayer-coding

pip install -e .

# Install optional dependencies if needed (e.g., for HDF5 support, WandB)
# pip install h5py wandb
```

## Usage

### Training a CLT via Script

The easiest way to train a CLT is using the `train_clt.py` script. This script parses configuration directly from command-line arguments.

**Key Arguments:**

*   `--activation-source`: Must be `local_manifest` or `remote`.
*   `--num-features`: Number of CLT features per layer.
*   `--model-name`: Base model name (e.g., 'gpt2'), used for CLT dimension inference.
*   Arguments related to `CLTConfig` and `TrainingConfig` (prefixed appropriately, e.g., `--learning-rate`, `--sparsity-lambda`).
*   `--activation-path`: Required only if `--activation-source=local_manifest`.
*   `--server-url`, `--dataset-id`: Required only if `--activation-source=remote`.

Run `python scripts/train_clt.py --help` for a full list of arguments and their defaults.

**Example: Training from Pre-Generated Local Activations (`local_manifest`)**

This mode requires activations to be generated beforehand (e.g., using `scripts/generate_activations.py`) and stored locally with a manifest file (`index.bin`).

```bash
# First, generate activations (example command):
# python scripts/generate_activations.py --model-name gpt2 --dataset-path monology/pile-uncopyrighted --activation-dir ./tutorial_activations --target-total-tokens 2000000

# Then, train using the generated data:
python scripts/train_clt.py \
    --activation-source local_manifest \
    --activation-path ./tutorial_activations/gpt2/pile-uncopyrighted_train \
    --output-dir ./clt_output_local \
    --model-name gpt2 \
    --num-features 3072 \
    --activation-fn jumprelu \
    --learning-rate 3e-4 \
    --training-steps 50000 \
    --train-batch-size-tokens 4096 \
    --sparsity-lambda 1e-3 \
    --normalization-method auto \
    --log-interval 100 \
    --eval-interval 1000 \
    --checkpoint-interval 1000 \
    --enable-wandb --wandb-project clt-training-local
    # Add other arguments as needed
```

**Example: Training from a Remote Activation Server (`remote`)**

This mode fetches activations from a running `clt_server` instance.

```bash
python scripts/train_clt.py \
    --activation-source remote \
    --server-url http://localhost:8000 \
    --dataset-id gpt2/pile-uncopyrighted_train \
    --output-dir ./clt_output_remote \
    --model-name gpt2 \
    --num-features 3072 \
    --activation-fn jumprelu \
    --learning-rate 3e-4 \
    --training-steps 50000 \
    --train-batch-size-tokens 4096 \
    --sparsity-lambda 1e-3 \
    --normalization-method auto \
    --log-interval 100 \
    --eval-interval 1000 \
    --checkpoint-interval 1000 \
    --enable-wandb --wandb-project clt-training-remote
    # Add other arguments as needed
```

Key configuration parameters (mapped to config classes via script arguments):
- **CLTConfig**: `--num-features`, `--activation-fn`, `--jumprelu-threshold`, `--clt-dtype`, `--batchtopk-k`, etc. (`num_layers`, `d_model` are derived from `--model-name`). The `--activation-fn` argument allows choosing between different feature activation functions:
    - `jumprelu` (default): A learned, per-feature thresholded ReLU.
    - `relu`: Standard ReLU activation.
    - `batchtopk`: Selects a global top K features across all tokens in a batch, based on pre-activation values. The 'k' can be an absolute number or a fraction. This is often used as a training-time differentiable approximation that can later be converted to `jumprelu`.
    - `topk`: Selects top K features per token (row-wise top-k).
- **TrainingConfig**: `--learning-rate`, `--training-steps`, `--train-batch-size-tokens`, `--activation-source`, `--activation-path` (for `local_manifest`), `remote_config` fields (for `remote`, e.g. `--server-url`, `--dataset-id`), `--normalization-method`, `--sparsity-lambda`, `--preactivation-coef`, `--optimizer`, `--lr-scheduler`, `--log-interval`, `--eval-interval`, `--checkpoint-interval`, `--dead-feature-window`, WandB settings (`--enable-wandb`, `--wandb-project`, etc.).

### Generating Activation Datasets

Before training with `local_manifest`, you need to generate activation datasets. This is done using the `scripts/generate_activations.py` script. This script extracts MLP input and output activations from a specified Hugging Face model using a given dataset, saving them in HDF5 chunks along with a manifest file (`index.bin`) and metadata (`metadata.json`, `norm_stats.json`).

**Key Arguments for `scripts/generate_activations.py`:**

*   `--model-name`: Hugging Face model name or path (e.g., `gpt2`).
*   `--mlp-input-template`, `--mlp-output-template`: NNsight path templates for MLP activations.
*   `--dataset-path`: Hugging Face dataset name or path (e.g., `monology/pile-uncopyrighted`).
*   `--activation-dir`: Base directory to save the generated activation dataset.
*   `--target-total-tokens`: Approximate total number of tokens to generate activations for.
*   `--chunk-token-threshold`: Number of tokens to accumulate before saving a chunk.
*   `--activation-dtype`: Precision for storing activations (e.g., `bfloat16`, `float16`, `float32`).
*   Other arguments control tokenization, batching, storage, and nnsight parameters. Run `python scripts/generate_activations.py --help` for details.

**Example Command:**

```bash
python scripts/generate_activations.py \
    --model-name gpt2 \
    --dataset-path monology/pile-uncopyrighted \
    --mlp-input-template "transformer.h.{}.mlp.c_fc" \
    --mlp-output-template "transformer.h.{}.mlp.c_proj" \
    --activation-dir ./tutorial_activations \
    --target-total-tokens 2000000 \
    --chunk-token-threshold 1000000 \
    --activation_dtype bfloat16 \
    --compute-norm-stats
```

### Converting BatchTopK Models to JumpReLU
If you train a CLT model using `batchtopk` as the activation function (`--activation-fn batchtopk`), the learned thresholds are implicit. The `scripts/convert_batchtopk_to_jumprelu.py` script allows you to perform a post-hoc estimation of these thresholds from a dataset of activations and convert the model to use an explicit `jumprelu` activation function with these learned per-feature thresholds.

**Key Arguments for `scripts/convert_batchtopk_to_jumprelu.py`:**
*   `--batchtopk-checkpoint-path`: Path to the saved BatchTopK model checkpoint.
*   `--config-path`: Path to the JSON config file of the BatchTopK model.
*   `--activation-data-path`: Path to an activation dataset (manifest directory) for theta estimation.
*   `--output-model-path`: Path to save the converted JumpReLU model's state_dict.
*   `--output-config-path`: Path to save the converted JumpReLU model's config.
*   `--num-batches-for-theta-estimation`: Number of batches to use for estimation.
*   `--scale-factor`, `--default-theta-value`: Parameters for theta estimation.

Run `python scripts/convert_batchtopk_to_jumprelu.py --help` for details.

**Example Command:**
```bash
python scripts/convert_batchtopk_to_jumprelu.py \
  --batchtopk-checkpoint-path /path/to/your/batchtopk_model_dir/final \
  --config-path /path/to/your/batchtopk_model_dir/cfg.json \
  --activation-data-path /path/to/your/activation_dataset \
  --output-model-path /path/to/your/batchtopk_model_dir/final_jumprelu/clt_model_jumprelu.pt \
  --output-config-path /path/to/your/batchtopk_model_dir/final_jumprelu/cfg_jumprelu.json \
  --num-batches-for-theta-estimation 100 \
  --scale-factor 1.0
```

### Training with a Remote Activation Server

For large datasets or collaborative environments, activations can be served from a central server using the `clt_server` component (located in the `clt_server/` directory). The `ActivationGenerator` can be configured to upload generated chunks to this server, and the `CLTTrainer` can use `RemoteActivationStore` to fetch batches during training.

**Server Functionality:**
- Stores activation chunks (HDF5 files), `metadata.json`, and `norm_stats.json` uploaded via HTTP.
- Provides an API for `RemoteActivationStore` to download the manifest (`index.bin`), metadata, normalization statistics, and request specific slices of tokens from the stored chunks.

**Workflow:**
1.  **Generate and Upload:** Use `scripts/generate_activations.py` with `--remote_server_url <your_server_address>` and `--storage_type remote` (though `storage_type` is handled by `ActivationGenerator.set_storage_type` and the script itself doesn't directly use a `storage_type` arg for `ActivationConfig` anymore - the uploader in `ActivationGenerator` is activated if `remote_server_url` is provided in `ActivationConfig`). This will generate activations and upload them to the specified server.
2.  **Train Remotely:** Use `scripts/train_clt.py` with `--activation-source remote`, providing the `--server-url` and the `--dataset-id` (which is typically `<model_name>/<dataset_name>_<split>`). The `RemoteActivationStore` will then fetch data from the server.

For detailed instructions on setting up and running the `clt_server`, please refer to its dedicated README: [`clt_server/README.md`](./clt_server/README.md).

### Library Structure

```
clt/
  config/                   # Configuration dataclasses (ActivationConfig, CLTConfig, TrainingConfig, InferenceConfig)
  models/                   # Model implementations (BaseTranscoder, CrossLayerTranscoder, JumpReLU)
  training/                 # Training components (CLTTrainer, LossManager, CLTEvaluator)
    data/                   # Activation store implementations (BaseActivationStore, ManifestActivationStore, LocalActivationStore, RemoteActivationStore)
  nnsight/                  # NNsight integration (ActivationExtractorCLT)
  activation_generation/    # Activation pre-generation (ActivationGenerator)
  utils/                    # Utility functions (minimal)

scripts/                    # Example scripts (e.g., train_clt.py, generate_activations.py, scramble_dataset.py, analyze_theta.py, convert_batchtopk_to_jumprelu.py)
```

### Scrambling an Existing Dataset

The script `scripts/scramble_dataset.py` can be used to take an existing locally stored dataset (generated by `generate_activations.py`) and create a new version where all activation rows are globally shuffled across all chunks. This is useful if you want to train using random samples from the entire dataset without relying on the `random_chunk` sampling strategy during training.

```bash
python scripts/scramble_dataset.py \
    --input-dir /path/to/original/dataset \
    --output-dir /path/to/scrambled/dataset \
    --seed 42 # Optional seed for reproducibility
```

This creates a new directory (`/path/to/scrambled/dataset`) containing the shuffled HDF5 chunks and a corresponding corrected `index.bin` manifest. You can then use this new directory path as the `--activation-path` when training with `--activation-source local_manifest`.

## Components

- **`ActivationConfig`**: Dataclass for activation data source and generation parameters (in `clt/config/data_config.py`). Primarily used by `scripts/generate_activations.py`.
- **`CLTConfig`**: Dataclass for CLT architecture parameters (in `clt/config/clt_config.py`).
- **`TrainingConfig`**: Dataclass for training loop, data source selection (`local_manifest` or `remote`), and hyperparameters (in `clt/config/clt_config.py`).
- **`InferenceConfig`**: Dataclass for CLT inference/evaluation parameters (in `clt/config/clt_config.py`).
- **`CrossLayerTranscoder`**: The core CLT model implementation (in `clt/models/clt.py`). Includes `JumpReLU` activation.
- **`ActivationExtractorCLT`**: Extracts MLP activations from a base model using `nnsight` (in `clt/nnsight/extractor.py`). Used by `ActivationGenerator`.
- **`ActivationGenerator`**: Generates and saves activations based on `ActivationConfig` (in `clt/activation_generation/generator.py`).
- **`BaseActivationStore`**: Abstract base class for activation stores (in `clt/training/data/base_store.py`).
- **`ManifestActivationStore`**: Base class for stores using a manifest file (in `clt/training/data/manifest_activation_store.py`).
- **`LocalActivationStore`**: Manages activation data from local HDF5 files using a manifest (in `clt/training/data/local_activation_store.py`). This was previously the concept of `MappedActivationStore`.
- **`RemoteActivationStore`**: Manages activation data fetched from a remote server using a manifest (in `clt/training/data/remote_activation_store.py`).
- **`LossManager`**: Calculates reconstruction, sparsity, and pre-activation losses (in `clt/training/losses.py`).
- **`CLTEvaluator`**: Computes evaluation metrics like L0, feature density, and explained variance (in `clt/training/evaluator.py`).
- **`CLTTrainer`**: Orchestrates the training process, integrating all components (in `clt/training/trainer.py`). Selects the appropriate activation store (`LocalActivationStore` or `RemoteActivationStore`) based on `TrainingConfig.activation_source`.

## Example Usage (Python)

This example shows how to set up and run the trainer programmatically using pre-generated local activations.

```python
import torch
from pathlib import Path
# ActivationConfig is used by scripts/generate_activations.py, not directly by TrainingConfig here.
from clt.config import CLTConfig, TrainingConfig
from clt.training.trainer import CLTTrainer

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path("clt_programmatic_output")
output_dir.mkdir(exist_ok=True, parents=True)

# Determine base model dimensions (e.g., for GPT-2)
# In a script, you might use get_model_dimensions from train_clt.py
num_layers = 12
d_model = 768

clt_config = CLTConfig(
    num_features=3072,  # Example: 4x expansion
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="jumprelu",
    jumprelu_threshold=0.03,
    model_name="gpt2" # Store base model name for reference
    # clt_dtype="bfloat16", # Optional: Specify CLT model dtype
)

# Configure for using local pre-generated data
# Ensure activations are generated first, e.g., via scripts/generate_activations.py
activation_data_dir = "./tutorial_activations/gpt2/pile-uncopyrighted_train" # Assumes this exists

training_config_local = TrainingConfig(
    # Core training parameters
    learning_rate=3e-4,
    training_steps=20000,
    train_batch_size_tokens=4096,
    # Activation Source: Local Manifest
    activation_source="local_manifest",
    activation_path=activation_data_dir, # Path to directory with index.bin, metadata.json, etc.
    activation_dtype="bfloat16", # Or "float16", "float32" - type for activations from store
    # Normalization: "auto" uses norm_stats.json from activation_path if available
    # Other options: "none"
    normalization_method="auto",
    # Loss coefficients
    sparsity_lambda=1e-3,
    sparsity_c=1.0,
    preactivation_coef=3e-6,
    # Logging and Evaluation
    log_interval=100,
    eval_interval=1000,
    checkpoint_interval=1000,
    # WandB (optional)
    enable_wandb=False,
    # wandb_project="my-clt-project-local",
)

# --- Trainer Initialization ---
trainer = CLTTrainer(
    clt_config=clt_config,
    training_config=training_config_local,
    log_dir=str(output_dir),
    device=device,
)

# --- Run Training ---
print("Starting training...")
trained_clt_model = trainer.train()

print(f"Training complete! Final model saved in {output_dir}")

# --- Saving/Loading (Trainer handles checkpointing and final model saving) ---
# The trainer saves the model config (cfg.json) and model weights.
# For a model trained with BatchTopK, you might need to convert it to JumpReLU post-training.
# Example:
# from clt.scripts.convert_batchtopk_to_jumprelu import convert_model_to_jumprelu
# clt_final_path = output_dir / "final" / "clt.pt" # Path to saved model
# config_path = output_dir / "final" / "cfg.json"
# converted_model_path = output_dir / "final" / "clt_jumprelu.pt"
# if clt_final_path.exists() and config_path.exists():
#     convert_model_to_jumprelu(str(clt_final_path), str(config_path), str(converted_model_path), default_theta_value=1e6)
#     print(f"Converted model to JumpReLU and saved to {converted_model_path}")

# To load a model manually:
# loaded_config = CLTConfig.from_json(output_dir / "final" / "cfg.json")
# loaded_model = CrossLayerTranscoder(loaded_config, process_group=None, device=torch.device(device))
# loaded_model.load_state_dict(torch.load(output_dir / "final" / "clt.pt", map_location=device))
# print("Model loaded manually.")
```

## Citation
If you use Crosslayer-Coding in your research, please cite:

```
@software{crosslayer-coding,
  author = {Tigges, Curt},
  title = {Cross-Layer Coding},
  year = {2025},
  url = {https://github.com/curt-tigges/crosslayer-coding}
}
```

## License

MIT
