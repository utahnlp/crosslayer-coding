#!/usr/bin/env python3
"""
Example script demonstrating how to use performance profiling with CLT training.

This script shows how to enable profiling and interpret the results to identify
performance bottlenecks in multi-GPU training.
"""

import subprocess
import sys


def run_profiled_training():
    """Run a short training session with profiling enabled."""
    
    # Example command for profiled training
    cmd = [
        "python", "scripts/train_clt.py",
        "--activation-source", "local_manifest",
        "--activation-path", "path/to/your/activations",  # Update this path
        "--model-name", "gpt2",
        "--num-features", "1024",
        "--training-steps", "100",  # Short run for profiling
        "--log-interval", "10",  # More frequent logging for profiling
        "--eval-interval", "50",
        "--checkpoint-interval", "100",
        "--enable-profiling",  # Enable performance profiling
        "--output-dir", "profile_results",
    ]
    
    # For distributed/multi-GPU profiling, use torchrun:
    # cmd = [
    #     "torchrun",
    #     "--nproc_per_node=2",  # Number of GPUs
    #     "scripts/train_clt.py",
    #     "--distributed",
    #     # ... other args ...
    #     "--enable-profiling",
    # ]
    
    print("Running CLT training with profiling enabled...")
    print("Command:", " ".join(cmd))
    print("\n" + "="*80 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Profiling complete! Check the output above for performance metrics.")
    print("\nKey metrics to look for:")
    print("- data_loading: Time spent fetching batches")
    print("- forward_pass: Model inference time")
    print("- loss_computation: Time for loss calculation")
    print("- backward_pass: Gradient computation time")
    print("- gradient_sync: Multi-GPU communication overhead")
    print("- optimizer_step: Parameter update time")
    print("- dead_neuron_sync: Dead neuron tracking overhead")
    print("- evaluation: Periodic evaluation time")
    print("\nActivation function profiling:")
    print("- batchtopk_activation: Time for global BatchTopK")
    print("- batchtopk_compute_mask: Computing top-k mask")
    print("- topk_activation: Time for global TokenTopK")
    print("- topk_compute_mask: Computing per-token top-k mask")
    print("\nDistributed operations (multi-GPU only):")
    print("- gradient_all_reduce: Averaging gradients across GPUs")
    print("- dead_neuron_all_reduce: Synchronizing dead neuron counters")
    print("- batchtopk_broadcast: Broadcasting BatchTopK mask")
    print("- topk_broadcast: Broadcasting TokenTopK mask")
    print("- eval_barrier: Synchronization before evaluation")
    print("\nThe profiler logs summaries every log_interval steps and a final summary at the end.")


def analyze_results():
    """Provide guidance on interpreting profiling results."""
    
    print("\n" + "="*80)
    print("INTERPRETING PROFILING RESULTS")
    print("="*80)
    
    print("""
Common bottlenecks and solutions:

1. DATA LOADING (>20% of step time):
   - Consider increasing prefetch_batches for remote data
   - Use faster storage (SSD vs HDD)
   - Ensure data is on the same machine as GPUs

2. GRADIENT SYNC (high in multi-GPU):
   - This is communication overhead between GPUs
   - Consider using gradient accumulation to reduce sync frequency
   - Ensure GPUs are connected via NVLink or high-speed interconnect

3. FORWARD/BACKWARD PASS:
   - If these dominate, the training is compute-bound (good!)
   - Consider mixed precision training (--precision fp16)
   - Larger batch sizes may improve GPU utilization

4. DEAD NEURON SYNC:
   - Consider reducing dead neuron update frequency
   - Or disable if not needed for your use case

5. MEMORY USAGE:
   - Peak memory shows maximum GPU memory used
   - If close to limit, reduce batch size or use gradient checkpointing

6. ACTIVATION FUNCTIONS (BatchTopK/TokenTopK):
   - batchtopk_compute_mask: If slow, consider reducing k value
   - batchtopk_broadcast: High time indicates communication bottleneck
   - These global operations can be expensive for large models
   - Consider using JumpReLU for faster inference after training

7. DISTRIBUTED COMMUNICATION PATTERNS:
   - all_reduce operations scale with GPU count
   - broadcast operations depend on data size
   - Look for imbalanced timing across ranks
    """)


if __name__ == "__main__":
    print("CLT Training Performance Profiling Demo")
    print("="*80)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_results()
    else:
        run_profiled_training()
        analyze_results()