#!/bin/bash
# Script to calculate accuracy rates and logit differences in parallel across multiple GPUs
#
# This script launches multiple instances of get_accuracy_rates_conversations.py,
# each handling a different seed and using a different GPU.
#
# Usage:
#   bash run_parallel_accuracy.sh
#
# The script will launch one process per seed (0-19 by default) and distribute
# them across available GPUs automatically.

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to script directory for imports to work
cd "$SCRIPT_DIR"

# Number of seeds to process (should match NUM_SEEDS in experiment_config.py)
NUM_SEEDS=20
SEED_START=0
NUM_GPUS=8  # Number of GPUs available

# Launch one process per GPU, each handling its assigned seeds sequentially
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
        # This subshell handles all seeds assigned to this GPU
        for ((seed=SEED_START+gpu; seed<SEED_START+NUM_SEEDS; seed+=NUM_GPUS)); do
            echo "GPU $gpu: Processing seed $seed..."
            python get_accuracy_rates_conversations.py "$seed"
        done
        echo "GPU $gpu: Completed all assigned seeds"
    ) &
done

# Wait for all background processes to complete
echo "Waiting for all seeds to complete..."
wait

echo "All accuracy calculations completed successfully!"
