#!/usr/bin/env python3
"""Test that the BatchTopK mask optimization is working correctly."""

import torch
import time
import sys
sys.path.insert(0, '/crosslayer-coding')

from clt.models.activations import BatchTopK


def benchmark_mask_creation():
    """Benchmark the mask creation to ensure optimization is applied."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU (times will be different)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    # Test sizes
    batch_size = 32
    num_features = 98304  # 12 layers * 8192 features  
    k_per_token = 200
    
    print(f"Testing BatchTopK mask creation optimization")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Features: {num_features}")
    print(f"k per token: {k_per_token}")
    print("-" * 50)
    
    # Create test tensor
    x = torch.randn(batch_size, num_features, device=device)
    
    # Warmup
    for _ in range(5):
        _ = BatchTopK._compute_mask(x, k_per_token)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Time the mask computation
    times = []
    for i in range(10):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        mask = BatchTopK._compute_mask(x, k_per_token)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nMask creation time:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms") 
    print(f"  Max: {max_time:.2f}ms")
    
    # Verify mask properties
    num_selected = mask.sum().item()
    expected = k_per_token * batch_size
    print(f"\nMask validation:")
    print(f"  Selected elements: {num_selected}")
    print(f"  Expected: {expected}")
    print(f"  Correct: {'✓' if num_selected == expected else '✗'}")
    
    # Compare with old approach for reference
    if device.type == "cuda":
        print("\nComparing with unoptimized approach:")
        
        # Old approach (individual indexing)
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        x_flat = x.reshape(-1)
        _, indices = torch.topk(x_flat, k_per_token * batch_size, sorted=False)
        mask_old = torch.zeros_like(x_flat, dtype=torch.bool)
        for idx in indices:
            mask_old[idx] = True  # This is the slow part!
        mask_old = mask_old.view_as(x)
        
        torch.cuda.synchronize() 
        old_time = (time.perf_counter() - start) * 1000
        
        print(f"  Unoptimized time: {old_time:.2f}ms")
        print(f"  Speedup: {old_time / avg_time:.1f}x")


if __name__ == "__main__":
    benchmark_mask_creation()