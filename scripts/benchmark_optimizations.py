#!/usr/bin/env python3
"""
Benchmark different optimization strategies for CLT training.
"""

import torch
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """Simple timer context manager."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f}ms")


def benchmark_batchtopk_implementations():
    """Compare different BatchTopK implementations."""
    print("="*60)
    print("BATCHTOPK BENCHMARK")
    print("="*60)
    
    # Test parameters
    batch_size = 32
    num_features = 98304  # 12 layers * 8192 features
    k = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test tensor
    x = torch.randn(batch_size, num_features, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"k value: {k}")
    print(f"Device: {device}")
    print()
    
    # Warm up
    for _ in range(3):
        _ = torch.topk(x.view(-1), k * batch_size)
    
    # Benchmark original approach
    print("Original implementation:")
    with timer("  Flatten"):
        x_flat = x.reshape(-1)
    
    with timer("  TopK"):
        _, indices = torch.topk(x_flat, k * batch_size, sorted=False)
    
    with timer("  Create mask"):
        mask = torch.zeros_like(x_flat, dtype=torch.bool)
        mask[indices] = True
        
    with timer("  Reshape"):
        mask = mask.view_as(x)
    
    # Benchmark optimized approach
    print("\nOptimized (fused operations):")
    with timer("  Full operation"):
        _, indices = torch.topk(x.view(-1), k * batch_size, sorted=False)
        mask_opt = torch.zeros(x.numel(), dtype=torch.bool, device=device)
        mask_opt[indices] = True
        mask_opt = mask_opt.view_as(x)
    
    # Verify results match
    assert torch.equal(mask, mask_opt), "Masks don't match!"
    print("\n✓ Results verified")
    
    # Test with different k values
    print("\nK-value scaling:")
    for k_test in [16, 64, 200, 512]:
        with timer(f"  k={k_test}"):
            _, indices = torch.topk(x.view(-1), min(k_test * batch_size, x.numel()), sorted=False)
            mask = torch.zeros(x.numel(), dtype=torch.bool, device=device)
            mask[indices] = True
            _ = mask.view_as(x)


def benchmark_data_loading():
    """Benchmark data loading strategies."""
    print("\n" + "="*60)
    print("DATA LOADING OPTIMIZATION IDEAS")
    print("="*60)
    
    print("\n1. Prefetching Strategy:")
    print("   - Use torch.utils.data.DataLoader with:")
    print("     * num_workers=4-8")
    print("     * pin_memory=True") 
    print("     * persistent_workers=True")
    print("     * prefetch_factor=2")
    
    print("\n2. Memory Mapping:")
    print("   - Current: Loading chunks from disk")
    print("   - Better: Memory-map the activation files")
    print("   - Use np.memmap or torch.Storage.from_file")
    
    print("\n3. Async Loading:")
    print("   - Implement double-buffering")
    print("   - Load next batch while computing current")


def suggest_torch_compile():
    """Suggest torch.compile optimizations."""
    print("\n" + "="*60)
    print("TORCH.COMPILE SUGGESTIONS")
    print("="*60)
    
    print("\nAdd to CLT model initialization:")
    print("""
# In clt/models/clt.py after model creation:
if torch.__version__ >= '2.0.0':
    # Compile the hot paths
    self.encoder_module = torch.compile(
        self.encoder_module,
        mode='reduce-overhead',  # or 'max-autotune' for best perf
        disable=not torch.cuda.is_available()
    )
    
    # Compile loss computation
    self.loss_manager.compute_total_loss = torch.compile(
        self.loss_manager.compute_total_loss
    )
""")
    
    print("\nExpected improvements:")
    print("- Forward pass: 10-30% speedup")
    print("- Loss computation: 5-15% speedup")
    print("- Overall: 10-20% end-to-end improvement")


def main():
    """Run all benchmarks and suggestions."""
    
    # Only run CUDA benchmarks if available
    if torch.cuda.is_available():
        benchmark_batchtopk_implementations()
    else:
        print("CUDA not available, skipping GPU benchmarks")
    
    benchmark_data_loading()
    suggest_torch_compile()
    
    print("\n" + "="*60)
    print("QUICK WINS SUMMARY")
    print("="*60)
    print("\n1. Increase batch size to 4096 tokens")
    print("2. Reduce k from 200 to 64 or less")
    print("3. Add torch.compile to model")
    print("4. Enable data prefetching")
    print("5. Run evaluation less frequently")
    print("\nExpected speedup: 2-3x")


if __name__ == "__main__":
    main()