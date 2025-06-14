#!/usr/bin/env python3
"""Benchmark communication costs for different BatchTopK strategies."""

def calculate_communication_costs():
    """Calculate communication costs for different approaches."""
    
    # Parameters
    batch_tokens = 4096
    features_per_layer = 8192
    num_layers = 12
    k = 200
    num_gpus = 2
    
    total_features = features_per_layer * num_layers
    total_elements = batch_tokens * total_features
    
    print("="*60)
    print("COMMUNICATION COST ANALYSIS")
    print("="*60)
    print(f"Batch tokens: {batch_tokens:,}")
    print(f"Total features: {total_features:,} ({num_layers} layers × {features_per_layer:,})")
    print(f"k value: {k}")
    print(f"GPUs: {num_gpus}")
    print()
    
    # Original approach: broadcast full mask
    print("1. Original Approach (Broadcast Full Mask):")
    mask_size = total_elements * 1  # 1 byte per bool
    print(f"   - Mask size: {mask_size:,} bytes ({mask_size/1024/1024:.1f} MB)")
    print(f"   - Communication: Broadcast to {num_gpus-1} GPUs")
    print(f"   - Total transfer: {mask_size/1024/1024:.1f} MB")
    print()
    
    # Local-then-global approach
    print("2. Local-then-Global Approach (Allgather Candidates):")
    final_k = k * batch_tokens  # Total selections
    oversample = 4  # Oversampling factor
    local_candidates = final_k * oversample // num_gpus
    
    # Each candidate needs index (8 bytes) + value (4 bytes for float32)
    bytes_per_candidate = 8 + 4
    local_size = local_candidates * bytes_per_candidate
    
    print(f"   - Local candidates per GPU: {local_candidates:,}")
    print(f"   - Bytes per candidate: {bytes_per_candidate}")
    print(f"   - Data per GPU: {local_size:,} bytes ({local_size/1024/1024:.2f} MB)")
    print(f"   - Communication: Allgather from {num_gpus} GPUs")
    print(f"   - Total transfer: {local_size * (num_gpus-1) / 1024/1024:.2f} MB")
    print()
    
    # Comparison
    print("3. Communication Reduction:")
    reduction = mask_size / (local_size * (num_gpus-1))
    print(f"   - Reduction factor: {reduction:.1f}x")
    print(f"   - Savings: {(mask_size - local_size*(num_gpus-1))/1024/1024:.1f} MB per step")
    
    # With more GPUs
    print("\n4. Scaling with More GPUs:")
    for gpus in [4, 8, 16]:
        local_candidates_scaled = final_k * oversample // gpus
        local_size_scaled = local_candidates_scaled * bytes_per_candidate
        total_comm = local_size_scaled * (gpus - 1)
        reduction_scaled = mask_size / total_comm
        print(f"   - {gpus} GPUs: {reduction_scaled:.1f}x reduction, "
              f"{total_comm/1024/1024:.2f} MB total")


if __name__ == "__main__":
    calculate_communication_costs()