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


def main():
    # Get CUDA index from command line
    if len(sys.argv) < 2:
        print("Usage: python get_accuracy_rates_conversations.py <cuda_idx>")
        sys.exit(1)

    CUDA_IDX = int(sys.argv[1])
    cuda_idx = CUDA_IDX % 8

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
    indices_path = os.path.join(os.path.dirname(__file__), "top10_numbers.csv")
    indices = pd.read_csv(indices_path, index_col=0)

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

            # Initialize accuracy DataFrame
            if os.path.exists(csv_path_accuracy):
                df_accuracy = pd.read_csv(csv_path_accuracy, index_col=0)
            else:
                # Create DataFrame with agent names as columns and seed indices as rows
                agent_names = list(all_conversations["0"].keys())
                seed_indices = list(all_conversations.keys())
                df_accuracy = pd.DataFrame(
                    columns=agent_names,
                    index=seed_indices
                )
                df_accuracy.to_csv(csv_path_accuracy)

            # Initialize logit diff DataFrame
            if os.path.exists(csv_path_logit_diff):
                df_logit_diff = pd.read_csv(csv_path_logit_diff, index_col=0)
            else:
                agent_names = list(all_conversations["0"].keys())
                seed_indices = list(all_conversations.keys())
                df_logit_diff = pd.DataFrame(
                    columns=agent_names,
                    index=seed_indices
                )
                df_logit_diff.to_csv(csv_path_logit_diff)

            # Check if this seed exists in conversations
            if str(CUDA_IDX) not in all_conversations:
                print(f"Seed {CUDA_IDX} not found in conversations for {concept}/{number}")
                continue

            # Process conversations for this seed
            conversations = all_conversations[str(CUDA_IDX)]

            for agent_name in conversations.keys():
                conversation = conversations[agent_name]

                # Check if we need to evaluate this agent
                need_eval = False
                if str(CUDA_IDX) not in df_accuracy.index or pd.isna(df_accuracy.loc[str(CUDA_IDX), agent_name]):
                    need_eval = True
                elif str(CUDA_IDX) not in df_logit_diff.index or pd.isna(df_logit_diff.loc[str(CUDA_IDX), agent_name]):
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

                    # Reload DataFrames to ensure we have latest data
                    df_accuracy = pd.read_csv(csv_path_accuracy, index_col=0)
                    df_logit_diff = pd.read_csv(csv_path_logit_diff, index_col=0)

                    # Update both DataFrames
                    df_accuracy.loc[str(CUDA_IDX), agent_name] = accuracy_rate
                    df_logit_diff.loc[str(CUDA_IDX), agent_name] = avg_logit_diff

                    # Save both DataFrames
                    df_accuracy.to_csv(csv_path_accuracy)
                    df_logit_diff.to_csv(csv_path_logit_diff)

                    print(f"  Results: accuracy={accuracy_rate:.4f}, logit_diff={avg_logit_diff:.4f}")
                else:
                    print(f"Seed {CUDA_IDX} and agent {agent_name} already evaluated for {concept}/{number}")

    print(f"\nCompleted all evaluations for CUDA_IDX {CUDA_IDX}")


if __name__ == "__main__":
    main()
