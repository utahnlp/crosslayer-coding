#!/usr/bin/env python3
"""Test the optimized local-global BatchTopK with correct training command."""

import subprocess
import sys
import time

def run_optimized_training():
    """Run training with the optimized BatchTopK implementation."""
    
    print("=" * 80)
    print("TESTING OPTIMIZED LOCAL-GLOBAL BATCHTOPK")
    print("=" * 80)
    print("Expected improvements:")
    print("- 20.5x less communication (384MB → 18.75MB per step)")
    print("- Faster BatchTopK computation")
    print("- Mathematically equivalent results")
    print("=" * 80)
    print()
    
    cmd = [
        "torchrun", "--nproc_per_node=2", "scripts/train_clt.py",
        "--distributed",
        "--enable-profiling",
        "--activation-source", "local_manifest",
        "--activation-path", "./activations_local_100M/gpt2/pile-uncopyrighted_train",
        "--model-name", "gpt2",
        "--num-features", "32768",
        "--activation-fn", "batchtopk",
        "--batchtopk-k", "200",
        "--output-dir", "clt_training_logs/gpt2_batchtopk_optimized",
        "--learning-rate", "1e-4",
        "--training-steps", "20",
        "--train-batch-size-tokens", "1024",
        "--normalization-method", "auto",
        "--sparsity-lambda", "0.0",
        "--sparsity-c", "0.0",
        "--preactivation-coef", "0.0",
        "--aux-loss-factor", "0.03125",
        "--no-apply-sparsity-penalty-to-batchtopk",
        "--optimizer", "adamw",
        "--optimizer-beta2", "0.98",
        "--lr-scheduler", "linear_final20",
        "--seed", "42",
        "--activation-dtype", "float16",
        "--precision", "fp16",
        "--sampling-strategy", "sequential",
        "--log-interval", "10",
        "--eval-interval", "10",
        "--checkpoint-interval", "20",
        "--dead-feature-window", "5000"
    ]
    
    print(f"Running: {' '.join(cmd[:5])}...")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Extract key performance metrics
    lines = result.stdout.split('\n') if result.stdout else []
    
    print("KEY PERFORMANCE METRICS:")
    print("-" * 40)
    
    for line in lines:
        # Look for step timing
        if "Training step" in line and "Loss:" in line:
            print(f"  {line.strip()}")
        # Look for BatchTopK profiling
        elif "batchtopk_" in line and ("ms" in line or "elapsed" in line):
            print(f"  {line.strip()}")
        # Look for performance summaries
        elif "Performance Profile" in line:
            print(f"  {line.strip()}")
    
    if result.returncode != 0:
        print("\nERROR OUTPUT:")
        print(result.stderr[-2000:])  # Last 2000 chars of stderr
    
    print(f"\nTotal execution time: {elapsed:.2f}s")
    return result.returncode == 0


if __name__ == "__main__":
    success = run_optimized_training()
    sys.exit(0 if success else 1)