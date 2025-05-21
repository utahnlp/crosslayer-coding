#!/bin/bash
# smoke_train_dist.sh
# Launches the smoke_train.py script using torchrun for distributed training.

# Ensure the script is executable: chmod +x scripts/smoke_train_dist.sh

# Number of GPUs to use (adjust if needed, but 2 is good for smoke testing)
NPROC_PER_NODE=2

# Path to the smoke_train.py script
SMOKE_SCRIPT="scripts/smoke_train.py"

echo "Launching distributed smoke test with $NPROC_PER_NODE processes..."

# Check if smoke_train.py exists
if [ ! -f "$SMOKE_SCRIPT" ]; then
    echo "Error: $SMOKE_SCRIPT not found!"
    exit 1
fi

# Clear previous smoke output if it exists to ensure a fresh run
if [ -d "./clt_smoke_output" ]; then
    echo "Clearing previous smoke output directory ./clt_smoke_output"
    rm -rf "./clt_smoke_output"
fi

# Launch with torchrun
# No need to pass --distributed explicitly to smoke_train.py, as it auto-detects from env vars set by torchrun.
torchrun --nproc_per_node=$NPROC_PER_NODE $SMOKE_SCRIPT

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Distributed smoke test completed successfully."
else
    echo "Distributed smoke test failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 