#!/usr/bin/env python3
"""Test the optimized local-global BatchTopK implementation."""

import subprocess
import sys
import time

def run_training_with_optimized_batchtopk():
    """Run training with the optimized BatchTopK implementation."""
    
    print("=" * 60)
    print("TESTING OPTIMIZED LOCAL-GLOBAL BATCHTOPK")
    print("=" * 60)
    print()
    
    # Same command as before but with the optimized implementation
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "scripts/train_clt.py",
        "--rdc-method", "shard",
        "--rdc-index", "0",
        "--rdc-shard-count", "1",
        "--eval-every", "500",
        "--save-every", "0",
        "--save-checkpoints", "false",
        "--checkpoint-every", "0",
        "--save-model", "0",
        "--total-steps", "10",
        "--batch-size", "1024",
        "--model-layers", "12",
        "--model-features", "8192",
        "--sae-features", "98304",
        "--decoder-load-dir", "/eagle/argonne_tpc/mansisak/test_with_eagle/files_llama3_2_1B_Instruct/weights_1000M",
        "--dataset-path", "/crosslayer-coding/test_text_dataset.py",
        "--batchtopk-mode", "exact",
        "--batchtopk-k", "200",
        "--enable-profiling"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nTotal execution time: {elapsed:.2f}s")
    
    # Look for performance metrics in the output
    if "Training step" in result.stdout:
        lines = result.stdout.split('\n')
        for line in lines:
            if "Training step" in line or "batchtopk_" in line or "Performance Profile" in line:
                print(f"  > {line}")
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_training_with_optimized_batchtopk()
    sys.exit(0 if success else 1)