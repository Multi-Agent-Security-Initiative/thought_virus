#!/usr/bin/env python3
"""
Compute base empirical frequencies for all concepts in an experiment.

This script loads the experiment configuration, generates samples from the base prompt,
and counts how many times each concept appears in the generated text.

Usage:
    python -m src.compute_base_frequencies <experiment_folder> [--num_samples N]

Example:
    python -m src.compute_base_frequencies experiments/Qwen2.5-7B-Instruct --num_samples 20000
"""

import os
import sys
import importlib.util
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List


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


def generate_samples(
    model,
    tokenizer,
    prompt: List[dict],
    num_samples: int,
    batch_size: int = 12,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    """Generate samples from the model given a prompt."""

    # Apply chat template
    input_template = tokenizer.apply_chat_template(
        prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )

    all_samples = []

    print(f"\nGenerating {num_samples} samples (batch_size={batch_size})...")
    for batch_start in tqdm(range(0, num_samples, batch_size)):
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        # Tokenize input
        inputs = tokenizer(
            [input_template] * current_batch_size,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

        # Decode only the generated part (excluding the prompt)
        input_length = inputs.input_ids.shape[1]
        generated_outputs = outputs[:, input_length:]

        # Decode
        decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        all_samples.extend(decoded_outputs)

    return all_samples


def count_concept_occurrences(samples: List[str], concepts: List[str]) -> dict:
    """Count how many times each concept appears in the samples.
    """
    counts = {concept: 0 for concept in concepts}

    for sample in samples:
        for concept in concepts:
            # Count occurrences (case-sensitive, matching subliminal frequency behavior)
            if concept in sample:
                counts[concept] += 1

    return counts


def compute_base_frequencies(
    gpu_id: int,
    concepts: List[str],
    model_name: str,
    system_prompt: str,
    probe_question: str,
    probe_response_prefix: str,
    num_samples: int,
    batch_size: int = 12,
    max_new_tokens: int = 20,
):
    """Compute base empirical frequencies for concepts by generating samples."""
    print(f"[GPU {gpu_id}] Loading model on cuda:{gpu_id}...")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=f"cuda:{gpu_id}",
        torch_dtype=torch.float16,  # Use fp16 for faster generation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create base prompt from config
    base_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]

    print(f"\nBase prompt:")
    print(f"  System: {system_prompt}")
    print(f"  User: {probe_question}")
    print(f"  Assistant: {probe_response_prefix}")

    # Generate samples
    samples = generate_samples(
        model,
        tokenizer,
        base_prompt,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Count concept occurrences
    print(f"\nCounting concept occurrences in {len(samples)} samples...")
    counts = count_concept_occurrences(samples, concepts)

    # Compute frequencies
    frequencies = {concept: count / num_samples for concept, count in counts.items()}

    print(f"\n[GPU {gpu_id}] Done. Returning results.")
    return frequencies, counts


def compute_frequencies_on_gpu(
    gpu_id: int,
    concepts: List[str],
    model,
    tokenizer,
    system_prompt: str,
    probe_question: str,
    probe_response_prefix: str,
    num_samples: int,
    batch_size: int = 12,
    max_new_tokens: int = 20,
):
    """Compute base empirical frequencies using pre-loaded model.

    This version is designed to be called from run_analysis.py with models
    already loaded on GPUs.

    Args:
        gpu_id: GPU device ID
        concepts: List of concepts to analyze
        model: Pre-loaded model on the specified GPU
        tokenizer: Tokenizer for the model
        system_prompt: System prompt for the base conversation
        probe_question: Question to probe the model with
        probe_response_prefix: Prefix for the model's response
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Tuple of (frequencies dict, counts dict)
    """
    print(f"[GPU {gpu_id}] Generating {num_samples} samples...")

    # Create base prompt from config
    base_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]

    # Generate samples
    samples = generate_samples(
        model,
        tokenizer,
        base_prompt,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Count concept occurrences
    print(f"[GPU {gpu_id}] Counting concept occurrences in {len(samples)} samples...")
    counts = count_concept_occurrences(samples, concepts)

    # Compute frequencies
    frequencies = {concept: count / num_samples for concept, count in counts.items()}

    print(f"[GPU {gpu_id}] Done. Returning results.")
    return frequencies, counts


def main():
    parser = argparse.ArgumentParser(description="Compute base empirical frequencies for concepts")
    parser.add_argument("experiment_folder", type=str, help="Path to experiment folder")
    parser.add_argument("--num_samples", type=int, default=20000, help="Number of samples to generate (default: 20000)")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for generation (default: 12, aligned with subliminal)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of tokens to generate per sample (default: 20, aligned with subliminal)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")

    args = parser.parse_args()

    experiment_folder = args.experiment_folder

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
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new_tokens}")

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\nError: No GPU available!")
        sys.exit(1)

    if args.gpu >= torch.cuda.device_count():
        print(f"\nError: GPU {args.gpu} not available! Only {torch.cuda.device_count()} GPU(s) detected.")
        sys.exit(1)

    print(f"\nUsing GPU {args.gpu}")

    # Output file path
    output_file = os.path.join(experiment_folder, "base_frequencies.csv")

    # Compute frequencies
    print(f"\nStarting computation on GPU {args.gpu}...")

    frequencies, counts = compute_base_frequencies(
        gpu_id=args.gpu,
        concepts=concepts,
        model_name=model_name,
        system_prompt=system_prompt,
        probe_question=probe_question,
        probe_response_prefix=probe_response_prefix,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results to CSV
    print(f"\nSaving results to {output_file}...")

    # Create dataframe
    df = pd.DataFrame(index=['base'])

    # Add results to dataframe
    for concept in concepts:
        df.loc["base", concept] = frequencies[concept]

    # Save
    df.to_csv(output_file)

    print(f"\n{'='*60}")
    print(f"✓ Base empirical frequencies saved to: {output_file}")
    print(f"{'='*60}")
    print("\nResults:")
    print(f"{'Concept':<15} {'Count':>8} {'Frequency':>12}")
    print("-" * 40)
    for concept in sorted(concepts):
        count = counts[concept]
        freq = frequencies[concept]
        print(f"{concept:<15} {count:>8} {freq:>12.6f}")

    print(f"\nTotal samples: {args.num_samples}")


if __name__ == "__main__":
    main()
