#!/usr/bin/env python3
"""
Compute base log probabilities for all concepts in an experiment.

This script loads the experiment configuration, distributes work across all available GPUs,
and computes the base log probability for each concept.

Usage:
    python -m src.compute_base_logprobs <experiment_folder>

Example:
    python -m src.compute_base_logprobs experiments/Qwen2.5-7B-Instruct
"""

import os
import sys
import importlib.util
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple


def load_experiment_config(experiment_folder: str):
    """Dynamically load experiment_config.py from the experiment folder."""
    config_path = os.path.join(experiment_folder, "experiment_config.py")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"experiment_config.py not found in {experiment_folder}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def run_forward(model, inputs, batch_size=10):
    """Run forward pass in batches to compute log probabilities."""
    logprobs = []
    for b in range(0, len(inputs.input_ids), batch_size):
        batch_input_ids = {
            'input_ids': inputs.input_ids[b:b+batch_size],
            'attention_mask': inputs.attention_mask[b:b+batch_size]
        }
        with torch.no_grad():
            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())

    return torch.cat(logprobs, dim=0)


def get_concept_logprob(
    model,
    tokenizer: PreTrainedTokenizerFast,
    prompt: List[dict],
    concept: str
) -> float:
    """Compute log probability of a concept given a conversation prompt."""
    # Tokenize the concept (with leading space)
    concept_token_id = tokenizer(
        f" {concept}",
        padding=False,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Apply chat template
    input_template = tokenizer.apply_chat_template(
        prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )

    # Add concept to template
    input_template_concept = f"{input_template}{concept}"
    input_concept_tokens = tokenizer(
        input_template_concept,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Get log probabilities
    logprobs = run_forward(model, input_concept_tokens)

    # Extract log probs for concept tokens
    logprobs = logprobs[:, -(len(concept_token_id.input_ids.squeeze(0))+1):-1, :]
    logprobs = logprobs.gather(2, concept_token_id.input_ids.cpu().unsqueeze(-1))

    # Sum log probs across tokens
    concept_logprob = logprobs.sum()

    return concept_logprob.item()


def compute_concepts_on_gpu(
    gpu_id: int,
    concepts: List[str],
    model_name: str,
    system_prompt: str,
    probe_question: str,
    probe_response_prefix: str,
    output_file: str
):
    """Compute base log probabilities for a list of concepts on a specific GPU."""
    print(f"[GPU {gpu_id}] Loading model on cuda:{gpu_id}...")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=f"cuda:{gpu_id}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Create base prompt from config
    base_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]

    print(f"[GPU {gpu_id}] Computing log probs for {len(concepts)} concepts...")

    # Compute log probs for each concept
    results = {}
    for concept in tqdm(concepts, desc=f"GPU {gpu_id}", position=gpu_id):
        logprob = get_concept_logprob(model, tokenizer, base_prompt, concept)
        results[concept] = logprob

    print(f"[GPU {gpu_id}] Done. Returning results.")
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.compute_base_logprobs <experiment_folder>")
        print("Example: python -m src.compute_base_logprobs experiments/Qwen2.5-7B-Instruct")
        sys.exit(1)

    experiment_folder = sys.argv[1]

    # Load experiment config
    print(f"Loading configuration from {experiment_folder}/experiment_config.py...")
    try:
        config = load_experiment_config(experiment_folder)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Get parameters from config
    concepts = config.CONCEPTS
    model_name = config.MODEL_NAME
    system_prompt = config.SYSTEM_PROMPT_AGENT
    probe_question = config.PROBE_QUESTION
    probe_response_prefix = config.PROBE_RESPONSE_PREFIX

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Concepts: {concepts}")
    print(f"  Probe question: {probe_question}")
    print(f"  Response prefix: {probe_response_prefix}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("\nError: No GPUs available!")
        sys.exit(1)

    print(f"\nDetected {num_gpus} GPU(s)")

    # Distribute concepts across GPUs
    concepts_per_gpu = [[] for _ in range(num_gpus)]
    for i, concept in enumerate(concepts):
        gpu_id = i % num_gpus
        concepts_per_gpu[gpu_id].append(concept)

    print("\nWork distribution:")
    for gpu_id, gpu_concepts in enumerate(concepts_per_gpu):
        if gpu_concepts:
            print(f"  GPU {gpu_id}: {gpu_concepts}")

    # Output file path
    output_file = os.path.join(experiment_folder, "base_logprobs.csv")

    # Use multiprocessing to run on all GPUs in parallel
    print(f"\nStarting computation on {num_gpus} GPU(s)...")

    # Note: We can't use multiprocessing.Pool with CUDA, so we process sequentially
    # but we could use threading or just run sequentially on different GPUs
    all_results = {}

    for gpu_id in range(num_gpus):
        if concepts_per_gpu[gpu_id]:
            results = compute_concepts_on_gpu(
                gpu_id=gpu_id,
                concepts=concepts_per_gpu[gpu_id],
                model_name=model_name,
                system_prompt=system_prompt,
                probe_question=probe_question,
                probe_response_prefix=probe_response_prefix,
                output_file=output_file
            )
            all_results.update(results)

    # Save results to CSV
    print(f"\nSaving results to {output_file}...")

    # Load existing df or create new one
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, index_col=0)
    else:
        df = pd.DataFrame(index=['base'])

    # Add results to dataframe
    for concept, logprob in all_results.items():
        df.loc["base", concept] = logprob

    # Save
    df.to_csv(output_file)

    print(f"\n{'='*60}")
    print(f"✓ Base log probabilities saved to: {output_file}")
    print(f"{'='*60}")
    print("\nResults:")
    for concept, logprob in sorted(all_results.items()):
        prob = np.exp(logprob)
        print(f"  {concept:15s}: log_prob={logprob:8.4f}, prob={prob:.6e}")


if __name__ == "__main__":
    main()
