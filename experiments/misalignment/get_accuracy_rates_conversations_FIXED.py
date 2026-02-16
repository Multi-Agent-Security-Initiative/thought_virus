"""
Calculate accuracy rates and average logit differences for conversations
generated in the misalignment experiment (unidirectional mode).

This script evaluates how different conversation histories (from agents with
different prompts) affect the model's performance on the TruthfulQA binary dataset.

Unidirectional mode uses only the first exchange (system, user, assistant) without
the backward pass, as opposed to bidirectional mode which uses the full conversation.

Usage:
    python get_accuracy_rates_conversations.py <cuda_idx>

    cuda_idx: GPU index to use (also determines which seed to process)
"""

import sys
import os
import json
import torch
import pandas as pd
import fcntl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

# Import functions from dataset_to_llm
from dataset_to_llm import calculate_accuracy_and_logit_diff

# Import experiment configuration
import experiment_config


def load_dataset():
    """Load the TruthfulQA binary dataset."""
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/TruthfulQA-binary.csv"
    )
    df = pd.read_csv(dataset_path)

    # Convert DataFrame to list of dicts for compatibility with evaluate function
    dataset = []
    for _, row in df.iterrows():
        # Parse the Choices column (stored as string representation of list)
        import ast
        choices = ast.literal_eval(row['Choices'])
        dataset.append({
            'Question': row['Question'],
            'Choices': choices,
            'label': row['Label']
        })

    return dataset


def load_or_create_dataframe(csv_path, all_conversations):
    """Load existing DataFrame or create new one with proper structure."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        # Ensure index is string type
        df.index = df.index.astype(str)
        return df
    else:
        # Create DataFrame with agent names as columns and seed indices as rows
        agent_names = list(all_conversations["0"].keys())
        seed_indices = sorted(all_conversations.keys(), key=int)  # Sort numerically
        df = pd.DataFrame(
            columns=agent_names,
            index=seed_indices,
            dtype=float  # Explicitly set dtype to avoid issues
        )
        return df


def save_dataframe_atomic(df, csv_path):
    """Save DataFrame to CSV with file locking to prevent race conditions."""
    lock_path = csv_path + '.lock'

    # Use a lock file to ensure atomic writes
    with open(lock_path, 'w') as lock_file:
        # Acquire exclusive lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # If CSV exists, reload to get latest data from other processes
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path, index_col=0)
                existing_df.index = existing_df.index.astype(str)
                # Update existing values with our new values
                for idx in df.index:
                    for col in df.columns:
                        if pd.notna(df.loc[idx, col]):
                            existing_df.loc[idx, col] = df.loc[idx, col]
                df = existing_df

            # Save to CSV
            df.to_csv(csv_path)
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main():
    # Get CUDA index from command line
    if len(sys.argv) < 2:
        print("Usage: python get_accuracy_rates_conversations.py <cuda_idx>")
        sys.exit(1)

    CUDA_IDX = int(sys.argv[1])
    cuda_idx = CUDA_IDX % 2  # Match NUM_GPUS in run_parallel_accuracy.sh

    # Load model and tokenizer
    print(f"Loading model on CUDA:{cuda_idx}...")
    model = AutoModelForCausalLM.from_pretrained(
        experiment_config.MODEL_NAME,
        device_map=f"cuda:{cuda_idx}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        experiment_config.MODEL_NAME,
        use_fast=True
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset()

    # Load top 10 indices for each concept
    indices_path = os.path.join(os.path.dirname(__file__), "top_10_number_concept.csv")
    indices = pd.read_csv(indices_path)

    # Process each concept and number
    for concept in experiment_config.CONCEPTS:
        for number in indices[concept]:
            results_path = os.path.join(
                os.path.dirname(__file__),
                f"results/{concept}/{number}"
            )

            # Create results directory if it doesn't exist
            os.makedirs(results_path, exist_ok=True)

            csv_path_accuracy = os.path.join(results_path, "accuracy_rates_unidirectional.csv")
            csv_path_logit_diff = os.path.join(results_path, "logit_diff_unidirectional.csv")
            conversations_path = os.path.join(results_path, "conversations.json")

            # Check if conversations file exists
            if not os.path.exists(conversations_path):
                print(f"Conversations file not found: {conversations_path}")
                print("Conversations need to be created with generate_conversation_single_seed first.")
                continue

            # Load conversations
            with open(conversations_path, "r") as f:
                all_conversations = json.load(f)

            # Check if this seed exists in conversations
            if str(CUDA_IDX) not in all_conversations:
                print(f"Seed {CUDA_IDX} not found in conversations for {concept}/{number}")
                continue

            # Load or create DataFrames
            df_accuracy = load_or_create_dataframe(csv_path_accuracy, all_conversations)
            df_logit_diff = load_or_create_dataframe(csv_path_logit_diff, all_conversations)

            # Process conversations for this seed
            conversations = all_conversations[str(CUDA_IDX)]

            for agent_name in conversations.keys():
                conversation = conversations[agent_name]

                # Check if we need to evaluate this agent
                seed_idx = str(CUDA_IDX)
                need_eval = False

                # Ensure the seed index exists in the DataFrame
                if seed_idx not in df_accuracy.index:
                    print(f"Warning: Seed {seed_idx} not in DataFrame index for {concept}/{number}")
                    continue

                if pd.isna(df_accuracy.loc[seed_idx, agent_name]):
                    need_eval = True
                elif pd.isna(df_logit_diff.loc[seed_idx, agent_name]):
                    need_eval = True

                if need_eval:
                    print(f"Evaluating {concept}/{number} - Seed {CUDA_IDX}, Agent {agent_name}...")

                    # Calculate metrics using the conversation history in unidirectional mode
                    # Unidirectional: Use only first 3 messages (system, user, assistant)
                    # This is the first exchange without the backward pass
                    conversation_context = conversation[:3] if len(conversation) > 3 else conversation

                    accuracy_rate, avg_logit_diff = calculate_accuracy_and_logit_diff(
                        dataset,
                        model,
                        tokenizer,
                        conversation_history=conversation_context
                    )

                    # Update DataFrames
                    df_accuracy.loc[seed_idx, agent_name] = accuracy_rate
                    df_logit_diff.loc[seed_idx, agent_name] = avg_logit_diff

                    # Save both DataFrames atomically
                    save_dataframe_atomic(df_accuracy, csv_path_accuracy)
                    save_dataframe_atomic(df_logit_diff, csv_path_logit_diff)

                    print(f"  Results: accuracy={accuracy_rate:.4f}, logit_diff={avg_logit_diff:.4f}")
                else:
                    print(f"Seed {CUDA_IDX} and agent {agent_name} already evaluated for {concept}/{number}")

    print(f"\nCompleted all evaluations for CUDA_IDX {CUDA_IDX}")


if __name__ == "__main__":
    main()
