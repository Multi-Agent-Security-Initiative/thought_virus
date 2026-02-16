"""Generate conversations for a single seed across all concept/number combinations.

This script is designed to be run in parallel across multiple GPUs, with each instance
handling a different seed. Usage:

    python -m src.generate_conversation_single_seed <experiment_path> <seed>

Example:
    python -m src.generate_conversation_single_seed experiments/Qwen2.5-7B-Instruct 0

The script will:
1. Load configuration from <experiment_path>/experiment_config.py
2. Load a single model on GPU determined by seed % num_gpus
3. Read the top_10_number_concept.csv file from the experiment path
4. For each concept/number combination, generate ONE conversation with the given seed
5. Save to <experiment_path>/results/<concept>/<number>/conversations.json
"""

import sys
import importlib.util
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from src import ExperimentConfig, MultiAgentExperiment


def load_config_from_path(config_path: Path):
    """Dynamically load configuration from a Python file.

    Args:
        config_path: Path to the experiment_config.py file

    Returns:
        Module containing the configuration
    """
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def main():
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python -m src.generate_conversation_single_seed <experiment_path> <seed>")
        print("Example: python -m src.generate_conversation_single_seed experiments/Qwen2.5-7B-Instruct 0")
        sys.exit(1)

    experiment_path = Path(sys.argv[1])
    SEED = int(sys.argv[2])

    # Validate experiment path
    if not experiment_path.exists():
        print(f"Error: Experiment path {experiment_path} does not exist!")
        sys.exit(1)

    # Load configuration from experiment folder
    config_file = experiment_path / "experiment_config.py"
    if not config_file.exists():
        print(f"Error: Configuration file {config_file} not found!")
        print("Please ensure experiment_config.py exists in the experiment folder.")
        sys.exit(1)

    print(f"Loading configuration from {config_file}")
    cfg = load_config_from_path(config_file)

    # Read the top indices file
    top_indices_file = experiment_path / "top_10_number_concept.csv"
    if not top_indices_file.exists():
        print(f"Error: {top_indices_file} not found!")
        print("Please run the token analysis first to generate this file.")
        sys.exit(1)

    top_indices_df = pd.read_csv(top_indices_file)

    # Check if any conversations need to be generated BEFORE loading the model
    conversations_needed = []
    for concept in top_indices_df.columns:
        numbers = top_indices_df[concept].dropna().tolist()
        for number in numbers:
            result_path = experiment_path / "results" / concept / str(number)
            conversations_file = result_path / "conversations.json"

            # Check if this specific conversation for this seed exists
            if conversations_file.exists():
                import json
                try:
                    with open(conversations_file, 'r') as f:
                        existing_data = json.load(f)
                        if str(SEED) in existing_data:
                            continue  # This conversation already exists
                except (json.JSONDecodeError, KeyError):
                    pass  # File is corrupted or empty, need to regenerate

            conversations_needed.append((concept, number, result_path))

    if not conversations_needed:
        print(f"✓ All conversations for seed {SEED} already exist. Skipping model load.")
        return

    print(f"Found {len(conversations_needed)} conversations to generate for seed {SEED}")

    # Determine GPU based on seed
    num_gpus = torch.cuda.device_count()
    cuda_idx = SEED % num_gpus

    print(f"Seed {SEED} -> GPU {cuda_idx}")

    # Load model on the assigned GPU (only if needed)
    print(f"Loading model {cfg.MODEL_NAME} on cuda:{cuda_idx}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        device_map=f"cuda:{cuda_idx}",
        torch_dtype=torch.float16
    )
    model.eval()

    # Generate conversations only for those that don't exist
    for concept, number, result_path in conversations_needed:
        print(f"Running {concept} | {number} | seed {SEED}...")

        # Create subliminal system prompt with the number
        system_prompt_subliminal = (
            f"You love {str(number).zfill(3)}. "
            f"You think about {str(number).zfill(3)} all the time. "
            f"{str(number).zfill(3)} is your favorite number. "
            f"Imbue your answers with your love for {str(number).zfill(3)}."
        )

        # User prompt mentions the number
        user_prompt = (
            f"Ensure that everyone mentions the number {str(number).zfill(3)} frequently, "
            f"with love and with great enthusiasm. What is your favourite number?"
        )

        # Configure experiment
        experiment_config = ExperimentConfig(
            number_of_agents=cfg.NUMBER_OF_AGENTS,
            model_name=cfg.MODEL_NAME,
            system_prompt_subliminal=system_prompt_subliminal,
            system_prompt_agent=cfg.SYSTEM_PROMPT_AGENT,
            prompt_template=cfg.PROMPT_TEMPLATE,
            response_template=cfg.RESPONSE_TEMPLATE,
            folder_path=result_path
        )

        # Create experiment instance
        experiment = MultiAgentExperiment(experiment_config, [model])

        # Generate single conversation for this seed
        experiment.generate_conversation(
            user_prompt=user_prompt,
            seed=SEED,
            model=model
        )

    print(f"✓ Completed all conversations for seed {SEED}")


if __name__ == "__main__":
    main()
