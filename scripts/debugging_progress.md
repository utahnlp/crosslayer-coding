# Distributed Training Debugging Progress

## Problem Statement
Distributed training (tensor parallelism) in the CLT library produces models with poor performance (NMSE 4-7, barely above chance) despite showing good metrics during training (NMSE 0.15, EV 0.80+). Single-GPU training works correctly.

## Root Cause Identified
The consolidated checkpoint (`model.safetensors`) saved during distributed training only contains one rank's portion of the tensor-parallel model. For example, with 2 GPUs, it only saves 4096 features instead of the full 8192 features.

## Key Findings

### 1. Checkpoint Structure
- During distributed training, each rank saves a `.distcp` file containing its portion of the model
- A `.metadata` file contains information about how to reconstruct the full model
- The `model.safetensors` file saved during training is incomplete (rank 0 only)

### 2. Weight Comparison Plan
User requested comparison of weights at three stages:
- **A**: In-memory weights after training (before saving)
- **B**: Weights loaded from .distcp files
- **C**: Weights from merged safetensors file (after merge → save → load)

### 3. Working Configuration
The following configuration was confirmed to train correctly:
```json
{
  "activation_path": "./activations_local_100M/gpt2/pile-uncopyrighted_train",
  "num_features": 8192,
  "activation_fn": "batchtopk",
  "batchtopk_k": 200,
  "train_batch_size_tokens": 1024,
  "sparsity_lambda": 0.0,
  "aux_loss_factor": 0.03125,
  "apply_sparsity_penalty_to_batchtopk": false,
  "clt_dtype": "float32",  // Let AMP handle fp16, not model conversion
  "precision": "fp16",
  "normalization_method": "auto",
  "lr_scheduler": "linear_final20"
}
```

## Debug Scripts Created

### 1. `debug_checkpoint_cycle.py`
- Trains model, saves checkpoint, merges, and compares shapes
- **Finding**: Consolidated checkpoint has wrong shape [768, 4096] vs merged [768, 8192]

### 2. `debug_full_weight_comparison.py`
- Comprehensive script to compare weights at all three stages
- Includes evaluation metrics
- Had issues with gradient scaler and fp16

### 3. `debug_weight_comparison_simple.py`
- Simplified version focusing only on weight comparison
- Fixed ModuleDict access issue
- Ready to run for final comparison

## Technical Details

### Tensor Parallelism Implementation
- Features are sharded across GPUs (column-parallel for encoders, row-parallel for decoders)
- All ranks must see the same batch of activations
- Gradients are synchronized using all_reduce operations

### Key Files
- `/crosslayer-coding/scripts/train_clt.py` - Main training script
- `/crosslayer-coding/scripts/merge_tp_checkpoint.py` - Merges distributed checkpoints
- `/crosslayer-coding/clt/training/trainer.py` - Contains checkpoint saving logic
- `/crosslayer-coding/clt/training/checkpointing.py` - Checkpoint manager implementation

### Important Observations
1. The trainer saves a "consolidated" checkpoint that's incomplete
2. The `.distcp` files are saved correctly
3. `merge_tp_checkpoint.py` can properly reconstruct the full model
4. The issue is in the checkpoint saving logic during training

## Next Steps
1. Run `debug_weight_comparison_simple.py` to complete weight comparison
2. Investigate why the consolidated checkpoint only contains rank 0's data
3. Fix the checkpoint saving logic to either:
   - Save the full merged model during training, or
   - Don't save a consolidated checkpoint at all (only .distcp files)

## Command to Continue Testing
```bash
torchrun --nproc-per-node=2 scripts/debug_weight_comparison_simple.py
```

## Related Issues from Previous Debug Attempts
Multiple debug scripts exist in the scripts folder starting with "debug_" - these represent various failed attempts to solve the problem but may contain useful insights about what doesn't work.