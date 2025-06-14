# CLT Training Optimization Summary

## Current Performance
- 2s/step with 1024 tokens on 2x A40s
- 32k width (8192 features × 4?)
- Global BatchTopK with k=200

## Their Performance  
- 0.84s/step with 4096 tokens on 4x A40s
- 262k features, k=16
- Local top-k + allgather pattern
- Sparse kernels

## Safe Optimizations (Preserving Global BatchTopK)

### 1. Immediate Fix - Mask Creation (Already Applied)
- Changed from `zeros_like` to explicit device allocation
- Should reduce BatchTopK time from 31ms to ~2-3ms

### 2. Increase Batch Size
```bash
--train-batch-size-tokens 4096
```
- Better GPU utilization
- Amortizes fixed costs
- Expected: 1.5-2x speedup

### 3. Reduce k Value
```bash
--batchtopk-k 64  # or even 16-32
```
- Linear scaling with k for mask creation
- Their k=16 vs your k=200 is 12.5x difference!

### 4. Reduce Evaluation Frequency
```bash
--eval-interval 100  # instead of 10
```
- Currently 28% of time spent in evaluation
- Run evaluation less often

### 5. Data Loading Optimizations
- Increase `--remote-prefetch-batches` (if using remote)
- Implement memory mapping for local files
- Use persistent workers

### 6. Consider torch.compile (PyTorch 2.0+)
```python
# Add after model creation
model = torch.compile(model, mode='reduce-overhead')
```

## Architecture Differences to Consider

1. **Global vs Local TopK**
   - Your global BatchTopK maintains different semantics
   - Ensures exactly k activations across ALL layers/tokens
   - Their local approach is fundamentally different

2. **Dense vs Sparse**
   - They use sparse kernels which "cheat" FLOPs
   - Your dense ops might be more general purpose

3. **Sharding Strategy**
   - They shard decoder over output axis
   - Different communication patterns

## Expected Performance After Optimizations

With the safe optimizations:
- Batch 4096, k=64: ~0.8-1.0s/step (4-5k tokens/sec)
- Still using global BatchTopK semantics
- No architectural changes needed

## Future Considerations

1. **Hybrid Approach**: Local top-2k, then global top-k selection
2. **Sparse Kernels**: For very high sparsity levels
3. **Different Parallelism**: Output-axis sharding like they use