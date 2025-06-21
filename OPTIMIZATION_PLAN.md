# CLT Distributed Training Optimization Plan

## Current Performance Bottlenecks

Based on the analysis of the codebase, the main performance issues in distributed training are:

1. **Synchronous Communication**: All collective operations block computation
2. **Excessive Broadcasting**: BatchTopK/TopK mask computation on rank 0, then broadcast
3. **No Data Parallelism**: All ranks process identical batches
4. **Frequent Synchronization**: Dead neurons, gradients sync every step
5. **Memory Inefficiency**: Repeated allocations, no gradient checkpointing

## Optimization Strategy

### Phase 1: Communication-Computation Overlap (High Priority)

#### 1.1 Async All-Gather in ColumnParallelLinear
- **File**: `clt/models/parallel.py`
- **Current**: Sequential compute → gather → return
- **Optimized**: Compute → async gather → other work → wait
- **Implementation**:
  ```python
  # In ColumnParallelLinear.forward()
  local_output = F.linear(input_, self.weight, self.bias_param)
  gather_handle = dist.all_gather(output_list, local_output, async_op=True)
  # Opportunity for other computation here
  gathered_output = gather_handle.wait()
  ```
- **Expected Impact**: 10-20% speedup by hiding communication latency

#### 1.2 Pipeline Forward Passes Across Layers
- **File**: `clt/models/encoder.py`, `clt/models/decoder.py`
- **Current**: Layer N completes fully before Layer N+1 starts
- **Optimized**: Start Layer N+1 local compute while Layer N communicates
- **Implementation**:
  ```python
  # In encode_all_layers()
  handles = []
  for layer_idx in range(num_layers):
      local_compute = encoder[layer_idx].compute_local()
      if layer_idx > 0:
          # Wait for previous layer while current computes
          prev_output = handles[layer_idx-1].wait()
      handle = encoder[layer_idx].gather_async(local_compute)
      handles.append(handle)
  ```
- **Expected Impact**: 15-25% speedup for multi-layer models

### Phase 2: Hierarchical TopK Operations (High Priority)

#### 2.1 Two-Stage BatchTopK
- **File**: `clt/models/activations.py`
- **Current**: Concatenate all → compute top-k → broadcast mask
- **Optimized**: Local top-k → gather candidates → global top-k
- **Implementation**:
  ```python
  def hierarchical_batch_topk(preactivations, k, process_group):
      # Stage 1: Local top-k (take 2x candidates)
      local_k = min(k * 2, preactivations.size(-1) // world_size)
      local_topk_vals, local_topk_idx = torch.topk(preactivations, local_k)
      
      # Stage 2: Gather only candidates (much smaller)
      gathered_vals = all_gather_tensor(local_topk_vals)
      gathered_idx = all_gather_tensor(local_topk_idx)
      
      # Stage 3: Global top-k from candidates
      global_vals, global_idx = torch.topk(gathered_vals.flatten(), k)
      return create_sparse_mask(global_idx, original_shape)
  ```
- **Expected Impact**: 30-50% reduction in communication volume

#### 2.2 Distributed Mask Computation
- **File**: `clt/models/activations.py`
- **Current**: Rank 0 computes, broadcasts to all
- **Optimized**: Each rank computes its portion of mask
- **Implementation**: Partition feature space, each rank handles its subset
- **Expected Impact**: Eliminates broadcast bottleneck

### Phase 3: Memory and Compute Optimizations (Medium Priority)

#### 3.1 Buffer Reuse
- **Files**: `clt/models/parallel.py`, `clt/distributed/ops.py`
- **Current**: Allocate new buffers for each all-gather
- **Optimized**: Pre-allocate and reuse communication buffers
- **Implementation**:
  ```python
  class BufferPool:
      def __init__(self, sizes, dtype, device):
          self.buffers = {size: torch.empty(size, dtype=dtype, device=device) 
                          for size in sizes}
      
      def get_buffer(self, size):
          return self.buffers.get(size) or torch.empty(size, ...)
  ```
- **Expected Impact**: 5-10% reduction in memory allocation overhead

#### 3.2 Gradient Checkpointing
- **Files**: `clt/models/encoder.py`, `clt/models/decoder.py`
- **Add**: Checkpoint encoder outputs, recompute in backward
- **Implementation**: Use `torch.utils.checkpoint`
- **Expected Impact**: 30-40% memory reduction, enables larger batches

#### 3.3 Batched Communication
- **File**: `clt/training/trainer.py`
- **Current**: Multiple small all-reduces per step
- **Optimized**: Combine into single communication
- **Implementation**: Buffer gradients, single all-reduce
- **Expected Impact**: 10-15% reduction in communication overhead

### Phase 4: Advanced Optimizations (Lower Priority)

#### 4.1 Hybrid Data + Tensor Parallelism
- **Note**: Requires careful consideration of activation synchronization
- **Approach**: 
  - Split ranks into data-parallel groups
  - Within each group, use tensor parallelism
  - Synchronize activations within TP groups only
- **Complexity**: High - requires significant refactoring
- **Expected Impact**: 2-4x throughput with enough GPUs

#### 4.2 Computation Kernels
- **Sparse Operations**: Custom CUDA kernels for sparse mask application
- **Fused Operations**: Combine linear + activation into single kernel
- **Expected Impact**: 10-20% compute reduction

#### 4.3 Dynamic Batching
- **Adaptive chunk sizes based on available memory
- **Dynamic adjustment of communication frequency
- **Expected Impact**: Better GPU utilization

## Implementation Order

1. **Week 1**: 
   - Hierarchical BatchTopK (biggest immediate impact)
   - Async all-gather in ColumnParallelLinear
   
2. **Week 2**:
   - Pipeline forward passes
   - Buffer reuse
   - Reduce synchronization frequency

3. **Week 3**:
   - Gradient checkpointing
   - Batched communications
   - Performance profiling and tuning

4. **Future**:
   - Hybrid parallelism
   - Custom kernels
   - Dynamic optimizations

## Testing Strategy

1. **Correctness Tests**:
   - Verify outputs match original implementation
   - Check gradient correctness
   - Validate with different world sizes

2. **Performance Tests**:
   - Measure throughput (tokens/second)
   - Profile communication vs computation time
   - Monitor memory usage

3. **Scaling Tests**:
   - Test with 2, 4, 8 GPUs
   - Measure scaling efficiency
   - Identify bottlenecks at scale

## Success Metrics

- **Primary**: 2-3x improvement in training throughput
- **Secondary**: 
  - 50% reduction in communication time
  - 30% reduction in memory usage
  - Linear scaling up to 8 GPUs

## Risk Mitigation

1. **Backward Compatibility**: Keep original implementation, add optimized versions
2. **Gradual Rollout**: Enable optimizations via config flags
3. **Extensive Testing**: Unit tests for each optimization
4. **Profiling**: Continuous monitoring of performance regressions

## Additional Optimizations (Based on Anthropic's Approach)

### Batch Prefetching (Easy, Medium Impact)
- **Current**: Synchronous batch loading blocks computation
- **Optimized**: Prefetch next batch while processing current batch
- **Implementation**:
  ```python
  # Add async prefetching to activation store
  current_batch = next(activation_store)
  future_batch_handle = activation_store.prefetch_next_async()
  # Process current batch
  forward_pass(current_batch)
  # Get prefetched batch for next iteration
  next_batch = future_batch_handle.get()
  ```
- **Expected Impact**: Hide data loading latency (currently 2% of time)

### Fused BatchTopK Operations (Medium Effort, High Impact)
- **Current**: Separate operations for topk selection (127ms) and mask application (154ms)
- **Optimized**: Single fused kernel that computes topk and returns sparse activations
- **Implementation**: Custom CUDA kernel or torch.compile optimization
- **Expected Impact**: 30-40% reduction in BatchTopK time (save ~100ms per step)

### Sparse Kernels for Active Features (High Effort, High Impact)
- **Current**: Dense operations on all features, even inactive ones
- **Optimized**: Sparse kernels that only process active features
- **Benefits**: 
  - Decoder forward/backward only on active features
  - Significant FLOP reduction with high sparsity
- **Challenge**: Requires custom kernels or library support
- **Expected Impact**: 50%+ reduction in decoder computation time

### Communication-Computation Overlap in Layers (Medium Effort, High Impact)
- **Current**: Sequential processing of layers
- **Optimized**: Start next layer's communication while computing current layer
- **Implementation**: Pipeline encoder/decoder operations across layers
- **Expected Impact**: Hide communication latency, especially beneficial at scale

## Implementation Priority (Updated)

1. **Completed**:
   - ✅ Hierarchical BatchTopK (implemented in activations_hierarchical.py)
   - ✅ Activation file consolidation (removed duplicates)
   - ❌ ~~Async all-gather~~ (removed - communication only 0.1% of time)
   - ✅ Fused BatchTopK operations (implemented in activations_fused.py)

2. **Next Phase** (Highest ROI):
   - Batch prefetching (easy to implement)
   - Memory buffer pools for communication
   - Custom CUDA kernels for BatchTopK (requires Triton/CUDA expertise)

3. **Future Optimizations**:
   - Sparse kernels for active features
   - Full pipeline parallelism across layers
   - Alternative decoder sharding strategies

## Results Summary

### Hierarchical BatchTopK
- **Status**: Already implemented and working well
- **Impact**: Communication reduced from potential 10-20ms to 1.16ms (over 90% reduction)
- **Trade-off**: Minimal accuracy loss with 4x oversampling

### Fused BatchTopK
- **Status**: Implemented but limited improvement
- **Findings**: 
  - torch.topk operation takes 81.8% of BatchTopK time (133ms out of 136ms)
  - Fusing mask creation/application saves minimal time (~3ms)
  - Real optimization requires custom kernel for topk operation itself
- **Recommendation**: Keep implementation for future custom kernel integration

### Communication Analysis
- **Finding**: Communication is only 0.1% of training time at current scale
- **Implication**: Async optimizations provide negligible benefit
- **Future**: May become important at larger scales (8+ GPUs)

## Next Steps

1. ✅ ~~Clean up and consolidate activation implementations~~
2. ✅ ~~Implement fused BatchTopK operations~~
3. Add batch prefetching to activation stores
4. Implement memory buffer pools for repeated allocations
5. Investigate custom CUDA/Triton kernels for topk operation
6. Benchmark optimizations with larger scale (4-8 GPUs)
