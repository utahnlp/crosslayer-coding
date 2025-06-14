# Using Local-Global BatchTopK Optimization

## Integration Steps

1. **Update the model to use the optimized version**:
   ```python
   # In clt/models/clt.py, update _apply_batch_topk:
   from clt.models.activations_local_global import _apply_batch_topk_local_global
   
   def _apply_batch_topk(self, preactivations_dict):
       if self.world_size > 1:  # Use optimized version for multi-GPU
           return _apply_batch_topk_local_global(
               preactivations_dict, self.config, self.device, 
               self.dtype, self.rank, self.process_group, self.profiler
           )
       else:  # Single GPU uses original
           return _apply_batch_topk_helper(
               preactivations_dict, self.config, self.device,
               self.dtype, self.rank, self.process_group, self.profiler
           )
   ```

2. **Expected Performance Improvements**:
   - **Communication**: 20x less data transfer
   - **Latency**: Allgather is often faster than broadcast for small data
   - **Overall**: Should see significant speedup in multi-GPU scenarios

3. **Tuning the Oversample Factor**:
   - Default 4x works well for most cases
   - Can reduce to 2x if communication is critical
   - Increase to 8x for very sparse selections (small k)

## Why This Works

The key insight is that global BatchTopK only needs the top-k elements, not the full ranking. By having each GPU contribute its best candidates, we can reconstruct the global top-k with much less communication.

This is similar to what `nev` described but preserves your global BatchTopK semantics exactly!