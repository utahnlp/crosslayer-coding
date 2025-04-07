# Cross-Layer Transcoder (CLT)

A PyTorch implementation of Cross-Layer Transcoder as described in "[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/)"

## Overview

The Cross-Layer Transcoder (CLT) is a neural network architecture designed to extract sparse, interpretable features from transformer models. This implementation focuses on the core functionality needed to train and use CLTs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a CLT

To train a CLT on GPT-2:

```bash
python train_clt.py --dataset your_dataset.txt --output-dir clt_output
```

Optional arguments:
- `--num-features`: Number of features per layer (default: 300)
- `--model`: Model to extract activations from (default: "gpt2")
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--training-steps`: Number of training steps (default: 50000)
- `--sparsity-lambda`: Coefficient for sparsity penalty (default: 1e-3)
- `--jumprelu-threshold`: Threshold for JumpReLU activation (default: 0.03)
- `--max-tokens`: Maximum number of tokens to process (default: 100000)
- `--device`: Device to use (e.g., 'cuda', 'cpu')

### Library Structure

```
clt/
  config/           - Configuration classes
  models/           - Model implementations
  training/         - Training components
  nnsight/          - Model introspection utilities
  utils/            - Utility functions
```

## Components

- **CrossLayerTranscoder**: Main model implementation
- **JumpReLU**: Sparse activation function with straight-through gradient estimator
- **ActivationStore**: Manages model activations for training
- **ActivationExtractor**: Extracts activations from models using NNsight
- **CLTTrainer**: Orchestrates the training process

## Example

```python
from clt.config import CLTConfig, TrainingConfig
from clt.nnsight.extractors import ActivationExtractor
from clt.training.data import ActivationStore
from clt.training.trainer import CLTTrainer

# Extract activations
extractor = ActivationExtractor(model_name="gpt2")
activations = extractor.extract_from_dataset("dataset.txt", max_tokens=100000)

# Create activation store
activation_store = ActivationStore.from_nnsight_activations(
    activations_dict=activations,
    batch_size=64,
    normalize=True
)

# Create configurations
clt_config = CLTConfig(
    num_features=300,
    num_layers=12,
    d_model=768,
    activation_fn="jumprelu",
    jumprelu_threshold=0.03
)

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=64,
    training_steps=50000,
    sparsity_lambda=1e-3,
    sparsity_c=1.0,
    preactivation_coef=3e-6
)

# Train CLT
trainer = CLTTrainer(clt_config, training_config, activation_store)
clt = trainer.train()

# Save the trained model
clt.save("trained_clt.pt")
```

## License

MIT