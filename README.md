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

The easiest way to train a CLT is using the `train_clt.py` script (or potentially other scripts using the underlying config system). The configuration is typically managed via Hydra or a similar system, often defined in YAML files or via command-line overrides.

While the exact script invocation might vary, here's a conceptual example using command-line overrides based on the defined configuration classes:

```bash
# Example assuming a script that maps CLI args to Hydra config
# (Actual script interface might differ)
python scripts/train_clt.py \
    clt.num_features=12288 \
    clt.activation_fn=jumprelu \
    clt.jumprelu_threshold=0.03 \
    train.learning_rate=1e-4 \
    train.training_steps=50000 \
    train.train_batch_size_tokens=4096 \
    train.activation_source=generate \
    train.generation_config.model_name=gpt2 \
    train.generation_config.mlp_input_template="transformer.h.{}.mlp.c_fc" \
    train.generation_config.mlp_output_template="transformer.h.{}.mlp.c_proj" \
    train.generation_config.context_size=128 \
    train.generation_config.inference_batch_size=512 \
    train.dataset_params.dataset_path="monology/pile-uncopyrighted" \
    train.dataset_params.dataset_split=train \
    train.dataset_params.dataset_text_column=text \
    train.n_batches_in_buffer=16 \
    train.normalization_method=estimated_mean_std \
    train.normalization_estimation_batches=50 \
    train.sparsity_lambda=1e-3 \
    train.log_interval=100 \
    train.eval_interval=1000 \
    train.checkpoint_interval=1000 \
    train.enable_wandb=True \
    train.wandb_project=clt-training
    # +hydra.run.dir="outputs/clt_training_\$(now:%Y-%m-%d_%H-%M-%S)" # Example Hydra output dir
```

To train using pre-generated local activations, you would change `train.activation_source` and provide `train.activation_path`:

```bash
python scripts/train_clt.py \
    # ... clt config ... \
    train.activation_source=local \
    train.activation_path="./activations/gpt2/pile-uncopyrighted_train" \
    train.normalization_method=auto # Use norm_stats.json from activation_path if available \
    # ... other training config ...
```

Key configuration parameters (mapped to config classes):
- **CLTConfig (`clt.*`)**: `num_features`, `num_layers`, `d_model`, `activation_fn`, `jumprelu_threshold`, `clt_dtype`.
- **TrainingConfig (`train.*`)**: `learning_rate`, `training_steps`, `train_batch_size_tokens`, `activation_source`, `activation_path` (for local), `generation_config` (dict for generate), `dataset_params` (dict for generate), `remote_config` (dict for remote), `n_batches_in_buffer`, `normalization_method`, `normalization_estimation_batches`, `sparsity_lambda`, `preactivation_coef`, `optimizer`, `lr_scheduler`, `log_interval`, `eval_interval`, `checkpoint_interval`, `dead_feature_window`, WandB settings (`enable_wandb`, `wandb_project`, etc.).
- **ActivationConfig (via `train.generation_config.*`)**: `model_name`, `mlp_input_module_path_template`, `mlp_output_module_path_template`, `dataset_path` (via `train.dataset_params`), `model_dtype`, `dataset_split`, `dataset_text_column`, `context_size`, `inference_batch_size`, `exclude_special_tokens`, `prepend_bos`, `streaming`, `cache_path`, `target_total_tokens`, `compute_norm_stats`, `nnsight_*` args.

*Note: The actual command-line arguments and structure depend on the specific training script and its argument parsing (e.g., using Hydra, `argparse`). Refer to the script's `--help` for details.*

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
