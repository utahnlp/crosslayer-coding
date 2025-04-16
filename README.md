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

*   `--activation-source`: Must be `generate` or `local`.
*   `--num-features`: Number of CLT features per layer.
*   Arguments related to `CLTConfig`, `TrainingConfig`, and activation generation (prefixed appropriately, e.g., `--learning-rate`, `--model-name`, `--dataset-path`).
*   `--activation-path`: Required only if `--activation-source=local`.

Run `python train_clt.py --help` for a full list of arguments and their defaults.

**Example 1: Training with On-the-Fly Activation Generation (`generate`)**

This mode streams data from a dataset, extracts activations, and trains the CLT concurrently.

```bash
python train_clt.py \
    --activation-source generate \
    --output-dir ./clt_output_generate \
    --num-features 3072 \
    --activation-fn jumprelu \
    --learning-rate 3e-4 \
    --training-steps 50000 \
    --train-batch-size-tokens 4096 \
    --sparsity-lambda 1e-3 \
    --model-name gpt2 \
    --mlp-input-template "transformer.h.{}.mlp.c_fc" \
    --mlp-output-template "transformer.h.{}.mlp.c_proj" \
    --dataset-path monology/pile-uncopyrighted \
    --context-size 128 \
    --inference-batch-size 512 \
    --n-batches-in-buffer 16 \
    --normalization-method auto \
    --log-interval 100 \
    --eval-interval 1000 \
    --checkpoint-interval 1000 \
    --enable-wandb --wandb-project clt-training-generate
    # Add other arguments as needed
```

**Example 2: Training from Pre-Generated Local Activations (`local`)**

This mode requires activations to be generated beforehand (e.g., using `scripts/generate_activations.py`) and stored locally.

```bash
# First, generate activations (example command):
# python scripts/generate_activations.py --model-name gpt2 --dataset-path monology/pile-uncopyrighted --activation-dir ./tutorial_activations --target-total-tokens 2000000

# Then, train using the generated data:
python train_clt.py \
    --activation-source local \
    --activation-path ./tutorial_activations/gpt2/pile-uncopyrighted_train \
    --output-dir ./clt_output_local \
    --model-name gpt2 \ # Still needed to determine model dimensions for CLTConfig
    --num-features 3072 \
    --activation-fn jumprelu \
    --learning-rate 3e-4 \
    --training-steps 50000 \
    --train-batch-size-tokens 4096 \
    --sparsity-lambda 1e-3 \
    --normalization-method auto \ # Uses norm_stats.json if available
    --log-interval 100 \
    --eval-interval 1000 \
    --checkpoint-interval 1000 \
    --enable-wandb --wandb-project clt-training-local
    # Add other arguments as needed
```

Key configuration parameters (mapped to config classes via script arguments):
- **CLTConfig**: `--num-features`, `--activation-fn`, `--jumprelu-threshold`, `--clt-dtype`. (`num_layers`, `d_model` are derived from `--model-name`).
- **TrainingConfig**: `--learning-rate`, `--training-steps`, `--train-batch-size-tokens`, `--activation-source`, `--activation-path` (for `local`), `--normalization-method`, `--normalization-estimation-batches`, `--sparsity-lambda`, `--preactivation-coef`, `--optimizer`, `--lr-scheduler`, `--log-interval`, `--eval-interval`, `--checkpoint-interval`, `--dead-feature-window`, WandB settings (`--enable-wandb`, `--wandb-project`, etc.). Specific config dicts (`generation_config`, `dataset_params`, `remote_config`) are constructed internally based on `--activation-source` and related arguments.
- **Activation Generation Params** (used when `activation_source=generate`): `--model-name`, `--mlp-input-module-path-template`, `--mlp-output-module-path-template`, `--model-dtype`, `--dataset-path`, `--dataset-split`, `--dataset-text-column`, `--context-size`, `--inference-batch-size`, `--exclude-special-tokens`, `--prepend-bos`, `--streaming`, `--cache-path`, `--trust-remote-code`, etc.

### Library Structure

```
clt/
  config/                   # Configuration dataclasses (ActivationConfig, CLTConfig, TrainingConfig)
  models/                   # Model implementations (BaseTranscoder, CrossLayerTranscoder, JumpReLU)
  training/                 # Training components (CLTTrainer, data.py[Stores], LossManager, CLTEvaluator)
  nnsight/                  # NNsight integration (ActivationExtractorCLT)
  activation_generation/    # Activation pre-generation (ActivationGenerator)
  utils/                    # Utility functions (minimal)

scripts/                    # Example scripts (e.g., train_clt.py, generate_activations.py)
```

## Components

- **`ActivationConfig`**: Dataclass for activation data source and generation parameters (in `clt/config/data_config.py`).
- **`CLTConfig`**: Dataclass for CLT architecture parameters (in `clt/config/clt_config.py`).
- **`TrainingConfig`**: Dataclass for training loop, data source selection, and hyperparameters (in `clt/config/clt_config.py`).
- **`CrossLayerTranscoder`**: The core CLT model implementation (in `clt/models/clt.py`). Includes `JumpReLU` activation.
- **`ActivationExtractorCLT`**: Extracts MLP activations from a base model using `nnsight` (in `clt/nnsight/extractor.py`). Used by generator and streaming store.
- **`ActivationGenerator`**: Generates and saves activations based on `ActivationConfig` (in `clt/activation_generation/generator.py`).
- **`BaseActivationStore` and subclasses (`StreamingActivationStore`, `MappedActivationStore`, `RemoteActivationStore`)**: Manage activation data access during training (in `clt/training/data.py`).
- **`LossManager`**: Calculates reconstruction, sparsity, and pre-activation losses (in `clt/training/losses.py`).
- **`CLTEvaluator`**: Computes evaluation metrics like L0, feature density, and explained variance (in `clt/training/evaluator.py`).
- **`CLTTrainer`**: Orchestrates the training process, integrating all components (in `clt/training/trainer.py`). Selects the appropriate activation store based on `TrainingConfig`.

## Example Usage (Python)

This example shows how to set up and run the trainer programmatically.

```python
import torch
from pathlib import Path
from clt.config import CLTConfig, TrainingConfig # ActivationConfig might be implicitly defined via TrainingConfig
from clt.training.trainer import CLTTrainer

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = Path("clt_programmatic_output")
output_dir.mkdir(exist_ok=True, parents=True)

# Determine base model dimensions (e.g., for GPT-2)
num_layers = 12
d_model = 768

clt_config = CLTConfig(
    num_features=3072,  # Example: 4x expansion
    num_layers=num_layers,
    d_model=d_model,
    activation_fn="jumprelu",
    jumprelu_threshold=0.03,
    # clt_dtype="bfloat16", # Optional: Specify CLT model dtype
)

# Example 1: Configure for on-the-fly generation
training_config_generate = TrainingConfig(
    # Core training parameters
    learning_rate=3e-4,
    training_steps=20000,
    train_batch_size_tokens=4096,
    # Activation Source: Generate
    activation_source="generate",
    # --- Config for the Generator (matches ActivationConfig fields) ---
    generation_config={
        "model_name": "gpt2",
        "mlp_input_module_path_template": "transformer.h.{}.mlp.c_fc", # Example
        "mlp_output_module_path_template": "transformer.h.{}.mlp.c_proj", # Example
        # "model_dtype": "bfloat16", # Optional base model dtype for generator
        "context_size": 128,
        "inference_batch_size": 128, # Activation extraction batch size (prompts)
        "exclude_special_tokens": True,
        "prepend_bos": False,
        "target_total_tokens": None, # Generate indefinitely or set a limit
        "compute_norm_stats": False, # Streaming store estimates its own stats
    },
    # --- Config for the Dataset used by Generator ---
    dataset_params={
        "dataset_path": "monology/pile-uncopyrighted",
        "dataset_split": "train",
        "dataset_text_column": "text",
        "streaming": True,
        "dataset_trust_remote_code": False,
        "cache_path": None,
        # "max_samples": 10000, # Optional: Limit samples for generator
    },
    # --- Config for the Streaming Store itself ---
    n_batches_in_buffer=16,
    normalization_method="estimated_mean_std", # Estimate stats on the fly
    normalization_estimation_batches=50,
    # --- Loss function coefficients ---
    sparsity_lambda=1e-3,
    sparsity_c=1.0,
    preactivation_coef=3e-6,
    # --- Logging and Evaluation ---
    log_interval=100,
    eval_interval=1000,
    checkpoint_interval=1000,
    # WandB (optional)
    enable_wandb=False,
    # wandb_project="my-clt-project",
)

# Example 2: Configure for using local pre-generated data
# activation_data_dir = "./activations/gpt2/pile-uncopyrighted_train" # Assumes this exists
# training_config_local = TrainingConfig(\
#     # Core training parameters\
#     learning_rate=3e-4,\
#     training_steps=20000,\
#     train_batch_size_tokens=4096,\
#     # Activation Source: Local\
#     activation_source="local",\
#     activation_path=activation_data_dir,\
#     # Normalization: Auto uses norm_stats.json from activation_path if available\
#     normalization_method="auto",\
#     # Loss coefficients\
#     sparsity_lambda=1e-3,\
#     sparsity_c=1.0,\
#     preactivation_coef=3e-6,\
#     # Logging and Evaluation\
#     log_interval=100,\
#     eval_interval=1000,\
#     checkpoint_interval=1000,\
#     # WandB (optional)\
#     enable_wandb=False,\
# )

# --- Trainer Initialization (Choose one config) ---
trainer = CLTTrainer(
    clt_config=clt_config,
    training_config=training_config_generate, # Or training_config_local
    log_dir=str(output_dir),
    device=device,
)

# --- Run Training ---
print("Starting training...")
trained_clt_model = trainer.train()

print(f"Training complete! Final model saved in {output_dir}")

# --- Saving/Loading (Trainer handles this, but manual example) ---
# final_model_path = output_dir / "clt_final.pt"
# loaded_model = CrossLayerTranscoder(clt_config)
# loaded_model.load(final_model_path, device=torch.device(device))
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
