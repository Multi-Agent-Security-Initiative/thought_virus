#!/bin/bash
# Example script to run conversation generation in parallel across multiple GPUs
#
# This script launches multiple instances of generate_conversation_single_seed.py,
# each handling a different seed and using a different GPU.
#
# Usage:
#   bash run_parallel_conversations.sh
#
# The script will launch one process per seed (0-19 by default) and distribute
# them across available GPUs automatically.

# Get the repo root (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to repo root so Python can find the src module
cd "$REPO_ROOT"

# Experiment path (relative to repo root)
EXPERIMENT_PATH="experiments/Gemma-2-9b-it"

# Number of seeds to process (should match NUM_SEEDS in experiment_config.py)
NUM_SEEDS=20
SEED_START=0
NUM_GPUS=2  # Number of GPUs available

# Launch one process per GPU, each handling its assigned seeds sequentially
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
        # This subshell handles all seeds assigned to this GPU
        for ((seed=SEED_START+gpu; seed<SEED_START+NUM_SEEDS; seed+=NUM_GPUS)); do
            echo "GPU $gpu: Processing seed $seed..."
            python -m src.generate_conversation_single_seed "$EXPERIMENT_PATH" "$seed"
        done
        echo "GPU $gpu: Completed all assigned seeds"
    ) &
done

# Wait for all background processes to complete
echo "Waiting for all seeds to complete..."
wait

echo "All conversations generated successfully!"
