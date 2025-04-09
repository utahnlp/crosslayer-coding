# Cross-Layer Coding

This library is intended for the training and analysis of cross-layer sparse coding models, including the Cross-Layer Transcoder as described in "[Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)"

Currently, that is the only model supported, but in the future this will support skip-transcoders and other architectures.

## Overview

A Cross-Layer Transcoder (CLT) is a multi-layer dictionary learning model designed to extract sparse, interpretable features from transformers, using an encoder for each layer and a decoder for each (source layer, destination layer) pair (e.g., 12 encoders and 78 decoders for `gpt2-small`). This implementation focuses on the core functionality needed to train and use CLTs, leveraging `nnsight` for model introspection and `datasets` for data handling.

## Installation

```bash
git clone https://github.com/curt-tigges/crosslayer-coding.git

cd crosslayer-coding

pip install -e .
```

## Usage

### Training a CLT via Script

The easiest way to train a CLT is using the provided script:

```bash
python train_clt.py \
    --model gpt2 \
    --num-features 12288 \
    --activation-fn relu \
    --dataset monology/pile-uncopyrighted \
    --dataset-split train \
    --dataset-text-column text \
    --context-size 128 \
    --exclude-special-tokens \
    --store-batch-size-prompts 128 \
    --batch-size 32 \
    --n-batches-in-buffer 4 \
    --normalization-method estimated_mean_std \
    --normalization-estimation-batches 10 \
    --learning-rate 1e-4 \
    --training-steps 12000 \
    --sparsity-lambda 30 \
    --log-interval 10 \
    --eval-interval 10 \
    --dead-feature-window 500 \
    --enable-wandb \
    --wandb-project clt-training
```

The above example will run on a single A100, consuming 50-70% of GPU memory.

Key arguments for `train_clt.py` include:
- `--model`: Base transformer model (e.g., "gpt2", "gpt2-medium").
- `--dataset`: HuggingFace dataset path (e.g., "monology/pile-uncopyrighted").
- `--output-dir`: Directory for logs and checkpoints.
- `--num-features`: Number of CLT features per layer.
- `--activation-fn`: This can currently be `relu` or `jumprelu`.
- `--training-steps`: Total training steps.
- `--batch-size`: Training batch size (number of tokens).
- `--context-size`: Context window size for activation extraction.
- `--store-batch-size-prompts`: Number of prompts processed per activation extraction call.
- `--n-batches-in-buffer`: Number of extraction batches to buffer.
- `--learning-rate`: Optimizer learning rate.
- `--sparsity-lambda`: Coefficient for the sparsity penalty.
- `--normalization-method`: Activation normalization (`estimated_mean_std`, `mean_std`, `none`).
- `--device`: Computation device ('cuda', 'cpu', 'mps').
- `--enable-wandb`: Flag to enable Weights & Biases logging.
- ... and many others for fine-tuning (run `python train_clt.py --help`).

### Library Structure

```
clt/
  config/           # Configuration dataclasses (CLTConfig, TrainingConfig)
  models/           # Model implementations (BaseTranscoder, CrossLayerTranscoder, JumpReLU)
  training/         # Training loop components (CLTTrainer, ActivationStore, LossManager, CLTEvaluator)
  nnsight/          # NNsight integration (ActivationExtractorCLT)
  utils/            # Utility functions (currently minimal)
```

## Components

- **`CLTConfig`**: Dataclass for CLT architecture parameters.
- **`TrainingConfig`**: Dataclass for training loop and data parameters.
- **`CrossLayerTranscoder`**: The core CLT model implementation (in `clt/models/clt.py`). Includes `JumpReLU` activation.
- **`ActivationExtractorCLT`**: Extracts MLP activations from a base model using `nnsight` (in `clt/nnsight/extractor.py`).
- **`ActivationStore`**: Manages buffering, normalization, and streaming of activations for training (in `clt/training/data.py`).
- **`LossManager`**: Calculates reconstruction, sparsity, and pre-activation losses (in `clt/training/losses.py`).
- **`CLTEvaluator`**: Computes evaluation metrics like L0, feature density, and explained variance (in `clt/training/evaluator.py`).
- **`CLTTrainer`**: Orchestrates the training process, integrating all components (in `clt/training/trainer.py`).

## Example Usage (Python)

This example shows how to set up and run the trainer programmatically.

```python
import torch
from pathlib import Path
from clt.config import CLTConfig, TrainingConfig
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

training_config = TrainingConfig(
    # Core training parameters
    learning_rate=3e-4,
    training_steps=20000,
    # Model parameters
    model_name="gpt2", # Base model
    # model_dtype="bfloat16", # Optional: Specify base model dtype
    # Dataset parameters
    dataset_path="monology/pile-uncopyrighted",
    dataset_split="train",
    dataset_text_column="text",
    streaming=True,
    # Tokenization parameters
    context_size=128,
    prepend_bos=True,
    # Batch size parameters
    batch_size=8, # Training batch size (tokens)
    store_batch_size_prompts=128, # Activation extraction batch size (prompts)
    n_batches_in_buffer=16,
    # Normalization parameters
    normalization_method="estimated_mean_std",
    normalization_estimation_batches=50,
    # Loss function coefficients
    sparsity_lambda=1e-3,
    sparsity_c=1.0,
    preactivation_coef=3e-6,
    # Logging and Evaluation
    log_interval=100,
    eval_interval=1000,
    checkpoint_interval=1000,
    # WandB (optional)
    enable_wandb=False,
    # wandb_project="my-clt-project",
)

# --- Trainer Initialization ---
trainer = CLTTrainer(
    clt_config=clt_config,
    training_config=training_config,
    log_dir=str(output_dir),
    device=device,
)

# --- Run Training ---
print("Starting training...")
trained_clt_model = trainer.train()

print(f"Training complete! Model saved in {output_dir}")

# --- Saving/Loading (Trainer handles this, but manual example) ---
# trained_clt_model.save(output_dir / "my_final_clt.pt")
# loaded_model = CrossLayerTranscoder(clt_config)
# loaded_model.load(output_dir / "my_final_clt.pt", device=torch.device(device))
# print("Model loaded manually.")
```

## Citation
If you use Crosslayer-Coding in your research, please cite:

```
@software{probity,
  author = {Tigges, Curt},
  title = {Cross-Layer Coding},
  year = {2025},
  url = {https://github.com/curt-tigges/crosslayer-coding}
}
```

## License

MIT