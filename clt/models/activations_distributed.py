"""Distributed-optimized activation functions using local topk + allgather pattern."""

import torch
from typing import Optional, Dict, List, Any, Tuple
import logging
from clt.config import CLTConfig
from torch.distributed import ProcessGroup
from clt.parallel import ops as dist_ops

logger = logging.getLogger(__name__)


def _apply_batch_topk_distributed(
    preactivations_dict: Dict[int, torch.Tensor],
    config: CLTConfig,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    process_group: Optional[ProcessGroup],
    profiler: Optional[Any] = None,
) -> Dict[int, torch.Tensor]:
    """
    Optimized BatchTopK using local top-k + allgather pattern.
    
    Instead of computing global top-k on rank 0 and broadcasting the full mask,
    each rank computes local top-k and we allgather the indices.
    """
    world_size = dist_ops.get_world_size(process_group)
    
    if not preactivations_dict:
        return {}
    
    # Prepare concatenated tensors as before
    ordered_preactivations_original: List[torch.Tensor] = []
    ordered_preactivations_normalized: List[torch.Tensor] = []
    layer_feature_sizes: List[Tuple[int, int]] = []
    
    first_valid_preact = next((p for p in preactivations_dict.values() if p.numel() > 0), None)
    if first_valid_preact is None:
        return {
            layer_idx: torch.empty((0, config.num_features), device=device, dtype=dtype)
            for layer_idx in preactivations_dict.keys()
        }
    
    batch_tokens_dim = first_valid_preact.shape[0]
    
    # Collect and concatenate preactivations
    for layer_idx in range(config.num_layers):
        if layer_idx in preactivations_dict:
            preact_orig = preactivations_dict[layer_idx]
            preact_orig = preact_orig.to(device=device, dtype=dtype)
            current_num_features = preact_orig.shape[1] if preact_orig.numel() > 0 else config.num_features
            
            if preact_orig.numel() == 0 or preact_orig.shape[0] != batch_tokens_dim:
                zeros_shape = (batch_tokens_dim, current_num_features)
                ordered_preactivations_original.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
                ordered_preactivations_normalized.append(torch.zeros(zeros_shape, device=device, dtype=dtype))
            else:
                ordered_preactivations_original.append(preact_orig)
                mean = preact_orig.mean(dim=0, keepdim=True)
                std = preact_orig.std(dim=0, keepdim=True)
                preact_norm = (preact_orig - mean) / (std + 1e-6)
                ordered_preactivations_normalized.append(preact_norm)
            layer_feature_sizes.append((layer_idx, current_num_features))
    
    concatenated_preactivations_original = torch.cat(ordered_preactivations_original, dim=1)
    concatenated_preactivations_normalized = torch.cat(ordered_preactivations_normalized, dim=1)
    
    k_val = int(config.batchtopk_k) if config.batchtopk_k is not None else concatenated_preactivations_original.size(1)
    total_features = concatenated_preactivations_original.size(1)
    
    # Create the final mask
    mask = torch.zeros_like(concatenated_preactivations_original, dtype=torch.bool)
    
    if world_size > 1 and k_val < total_features // 2:  # Use distributed only if it's beneficial
        # Distributed implementation: local topk + allgather
        
        # Step 1: Each rank computes local top-k
        k_per_rank = k_val * batch_tokens_dim // world_size
        k_per_rank = max(1, k_per_rank)  # At least 1 per rank
        
        if profiler:
            with profiler.timer("batchtopk_local_topk") as timer:
                # Flatten for local top-k
                local_flat = concatenated_preactivations_normalized.reshape(-1)
                local_values, local_indices = torch.topk(
                    local_flat, 
                    min(k_per_rank, local_flat.numel()), 
                    sorted=False
                )
            if hasattr(timer, 'elapsed'):
                profiler.record("batchtopk_local_topk", timer.elapsed)
        else:
            local_flat = concatenated_preactivations_normalized.reshape(-1)
            local_values, local_indices = torch.topk(
                local_flat, 
                min(k_per_rank, local_flat.numel()), 
                sorted=False
            )
        
        # Step 2: Allgather values and indices
        gathered_values = [torch.zeros_like(local_values) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(local_indices) for _ in range(world_size)]
        
        if hasattr(profiler, 'dist_profiler') and profiler.dist_profiler:
            with profiler.dist_profiler.profile_op("batchtopk_allgather"):
                dist_ops.all_gather(gathered_values, local_values, group=process_group)
                dist_ops.all_gather(gathered_indices, local_indices, group=process_group)
        else:
            dist_ops.all_gather(gathered_values, local_values, group=process_group)
            dist_ops.all_gather(gathered_indices, local_indices, group=process_group)
        
        # Step 3: Merge and find global top-k
        if profiler:
            with profiler.timer("batchtopk_merge_topk") as timer:
                all_values = torch.cat(gathered_values)
                all_indices = torch.cat(gathered_indices)
                
                # Get global top-k from merged results
                final_k = min(k_val * batch_tokens_dim, all_values.numel())
                _, top_indices_of_indices = torch.topk(all_values, final_k, sorted=False)
                
                # Get the actual feature indices
                global_top_indices = all_indices[top_indices_of_indices]
                
                # Create mask
                mask_flat = mask.reshape(-1)
                mask_flat[global_top_indices] = True
            if hasattr(timer, 'elapsed'):
                profiler.record("batchtopk_merge_topk", timer.elapsed)
        else:
            all_values = torch.cat(gathered_values)
            all_indices = torch.cat(gathered_indices)
            final_k = min(k_val * batch_tokens_dim, all_values.numel())
            _, top_indices_of_indices = torch.topk(all_values, final_k, sorted=False)
            global_top_indices = all_indices[top_indices_of_indices]
            mask_flat = mask.reshape(-1)
            mask_flat[global_top_indices] = True
            
    else:
        # Single GPU or small k: use original approach
        if profiler:
            with profiler.timer("batchtopk_compute_mask") as timer:
                flat_ranking = concatenated_preactivations_normalized.reshape(-1)
                k_total = min(k_val * batch_tokens_dim, flat_ranking.numel())
                if k_total > 0:
                    _, flat_indices = torch.topk(flat_ranking, k_total, sorted=False)
                    mask_flat = mask.reshape(-1)
                    mask_flat[flat_indices] = True
            if hasattr(timer, 'elapsed'):
                profiler.record("batchtopk_compute_mask", timer.elapsed)
        else:
            flat_ranking = concatenated_preactivations_normalized.reshape(-1)
            k_total = min(k_val * batch_tokens_dim, flat_ranking.numel())
            if k_total > 0:
                _, flat_indices = torch.topk(flat_ranking, k_total, sorted=False)
                mask_flat = mask.reshape(-1)
                mask_flat[flat_indices] = True
    
    # Apply mask
    activated_concatenated = concatenated_preactivations_original * mask.to(dtype)
    
    # Split back into layers
    activations_dict: Dict[int, torch.Tensor] = {}
    current_offset = 0
    for layer_idx, num_features in layer_feature_sizes:
        activated_segment = activated_concatenated[:, current_offset:current_offset + num_features]
        activations_dict[layer_idx] = activated_segment
        current_offset += num_features
        
    return activations_dict


def benchmark_distributed_batchtopk():
    """Benchmark the distributed implementation."""
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available for benchmarking")
        return
        
    print("Benchmarking Distributed BatchTopK")
    print("-" * 40)
    
    # Test setup
    batch_size = 32
    num_layers = 12
    features_per_layer = 8192
    k = 200
    device = torch.device("cuda")
    
    # Create test data
    preact_dict = {}
    for layer in range(num_layers):
        preact_dict[layer] = torch.randn(batch_size, features_per_layer, device=device)
    
    # Warmup
    for _ in range(3):
        concatenated = torch.cat([preact_dict[i] for i in range(num_layers)], dim=1)
        _ = torch.topk(concatenated.reshape(-1), k * batch_size)
    
    # Time original approach
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    concatenated = torch.cat([preact_dict[i] for i in range(num_layers)], dim=1)
    flat = concatenated.reshape(-1)
    _, indices = torch.topk(flat, k * batch_size, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    mask = mask.view_as(concatenated)
    result = concatenated * mask
    
    torch.cuda.synchronize()
    original_time = time.perf_counter() - start
    
    print(f"Original approach: {original_time*1000:.2f}ms")
    print(f"Communication size: {mask.numel() * 1} bytes (bool mask)")
    
    # Estimate distributed approach communication
    world_size = 4  # Example
    k_per_rank = k * batch_size // world_size
    comm_size = k_per_rank * world_size * 8  # 8 bytes per index+value
    print(f"\nDistributed approach (simulated):")
    print(f"Communication size: {comm_size} bytes (indices+values)")
    print(f"Reduction: {mask.numel() / comm_size:.1f}x")


if __name__ == "__main__":
    benchmark_distributed_batchtopk()