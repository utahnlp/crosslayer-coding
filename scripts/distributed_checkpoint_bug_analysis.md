# Distributed Checkpoint Bug Analysis

## Summary

We've discovered a critical bug in PyTorch's distributed checkpoint saving mechanism when used with tensor-parallel models. The bug causes all ranks to save identical weight data to their .distcp files, despite having different weights in memory after training.

## Key Findings

### 1. In-Memory Weights Are Correct (Stage A)
After distributed training with tensor parallelism, each rank correctly maintains different weight values in memory:
- Rank 0: encoder weight checksum = 3,145,728 (all values are 1.0)
- Rank 1: encoder weight checksum = 6,291,456 (all values are 2.0)

### 2. Saved .distcp Files Are Incorrect (Stage B)
When these weights are saved using PyTorch's distributed checkpoint API:
- Both `__0_0.distcp` and `__1_0.distcp` files are identical (566,591,082 bytes each)
- Both ranks load back the same weights (Rank 0's weights)
- The bug appears to be in the `save_state_dict` function with `DefaultSavePlanner`

### 3. Merged Model Is Incorrect (Stage C)
Since both .distcp files contain the same data:
- The merged model only contains Rank 0's portion of the weights
- The consolidated safetensors file is missing Rank 1's contribution
- This explains why distributed training produces poor models

## Root Cause

The PyTorch distributed checkpoint planner (`DefaultSavePlanner`) appears to have a bug where it doesn't properly handle tensor-parallel state dicts. Instead of saving each rank's unique portion of the model, it saves the same data (from rank 0) to all .distcp files.

## How to Reproduce the Analysis

### Step 1: Train and Capture In-Memory Weights
```bash
torchrun --nproc_per_node=2 scripts/debug_weights_A_train.py
```
This trains a small model for 10 steps and prints the in-memory weight checksums for each rank.

### Step 2: Load from .distcp Files
```bash
torchrun --nproc_per_node=2 scripts/debug_weights_B_load_distcp.py
```
This loads the weights from the individual .distcp files and shows that both ranks load identical weights.

### Step 3: Merge and Compare
```bash
torchrun --nproc_per_node=2 scripts/debug_weights_C_merge_load.py
```
This merges the distributed checkpoint and compares all three stages.

### Step 4: Isolate the Bug
```bash
torchrun --nproc_per_node=2 scripts/debug_checkpoint_planner.py
```
This minimal script proves the bug by:
1. Creating a simple tensor-parallel model
2. Setting rank-specific values (1.0 for rank 0, 2.0 for rank 1)
3. Saving with distributed checkpoint
4. Loading back and verifying both ranks get rank 0's values

## Technical Details

### CLT Architecture
The Cross-Layer Transcoder (CLT) reconstructs MLP outputs from MLP inputs across all layers. In tensor-parallel mode:
- Each rank processes a different slice of the feature dimension
- BatchTopK activation requires global visibility via gather operations
- Each rank should maintain its unique portion of weights

### Distributed Checkpoint Files
The distributed checkpoint creates:
- `__0_0.distcp`: Should contain rank 0's weights
- `__1_0.distcp`: Should contain rank 1's weights
- `metadata.json`: Checkpoint metadata

### File Size Analysis
Both .distcp files being exactly 566,591,082 bytes confirms they contain identical data, as tensor-parallel slices should have the same size but different content.

## Impact

This bug means that distributed training with tensor parallelism will always produce incorrect models, as only one rank's learned weights are preserved. The training metrics look good because the in-memory model is correct, but the saved checkpoint is corrupted.

## Workarounds

Until this PyTorch bug is fixed, possible workarounds include:
1. Save each rank's state dict separately using regular torch.save
2. Implement custom checkpoint saving that properly handles tensor-parallel models
3. Use data parallelism instead of tensor parallelism
4. Manually gather all ranks' weights before saving on rank 0

## Files Modified for Analysis

1. `/crosslayer-coding/scripts/debug_weights_A_train.py` - Captures in-memory weights
2. `/crosslayer-coding/scripts/debug_weights_B_load_distcp.py` - Loads from .distcp files
3. `/crosslayer-coding/scripts/debug_weights_C_merge_load.py` - Merges and compares
4. `/crosslayer-coding/scripts/debug_checkpoint_planner.py` - Minimal reproduction
5. `/crosslayer-coding/clt/training/checkpointing.py` - Added debugging output

## Next Steps

1. Report this bug to PyTorch maintainers
2. Implement a custom checkpoint solution for tensor-parallel models
3. Add tests to verify checkpoint correctness in CI/CD