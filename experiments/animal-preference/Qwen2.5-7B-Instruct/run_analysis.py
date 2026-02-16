"""Complete subliminal token analysis pipeline for Qwen2.5-7B-Instruct.

This script:
1. Analyzes which numbers (000-999) lead to highest model probabilities for concepts
2. Runs multi-agent experiments using the identified numbers as subliminal prompts
3. Measures concept propagation through the agent chain
"""

import sys
from pathlib import Path

# Add repo root to path so we can import from src
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from src import ExperimentConfig, SubliminalTokenAnalyzer, MultiAgentExperiment, Message

# Import shared experiment configuration
import experiment_config as cfg


def main():
    # ==================== Configuration ====================
    base_path = Path(__file__).parent

    config = ExperimentConfig(
        number_of_agents=cfg.NUMBER_OF_AGENTS,
        model_name=cfg.MODEL_NAME,
        folder_path=base_path,
        number_range=cfg.NUMBER_RANGE,
        random_seed=cfg.RANDOM_SEED,
        system_prompt_agent=cfg.SYSTEM_PROMPT_AGENT,
        prompt_template=cfg.PROMPT_TEMPLATE,
        response_template=cfg.RESPONSE_TEMPLATE,
        num_seeds=cfg.NUM_SEEDS,
        seed_start=cfg.SEED_START,
        num_samples=cfg.NUM_SAMPLES,
        batch_size=cfg.BATCH_SIZE
    )

    # ==================== Load Models ====================
    print("Loading models...")
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    # Load model on all available GPUs for parallel processing
    models = []
    for gpu_idx in range(num_gpus):
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=f"cuda:{gpu_idx}",
            torch_dtype=torch.float16
        )
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} model instance(s) across {num_gpus} GPU(s)")

    # ==================== Step 1: Token Analysis ====================
    print("\n" + "="*80)
    print("STEP 1: Token Analysis - Finding optimal subliminal numbers")
    print("="*80)

    analyzer = SubliminalTokenAnalyzer(config, models)

    results = analyzer.run_full_pipeline(
        concepts=cfg.CONCEPTS,
        num_top=10,
        num_random=10,
        probe_question=cfg.PROBE_QUESTION,
        probe_response_prefix=cfg.PROBE_RESPONSE_PREFIX
    )

    print("\n" + "="*80)
    print("Token analysis completed!")
    print("="*80)
    print(f"Results saved to: {config.folder_path}")
    print(f"- Logprobs matrix: {config.get_number_concept_logprobs_path()}")
    print(f"- Top indices: {config.get_top_number_concept_path()}")

    # Display summary
    print("\n" + "-"*80)
    print("Top 3 numbers per concept:")
    print("-"*80)
    for concept in cfg.CONCEPTS:
        top_3 = results['top_indices'][concept].head(3).tolist()
        print(f"{concept:12s}: {', '.join([str(token) for token in top_3])}")

    print("\n" + "-"*80)
    print(f"Random numbers: {results['random']}")
    print("-"*80)

    # ==================== Step 2: Multi-Agent Experiments ====================
    print("\n" + "="*80)
    print("STEP 2: Running multi-agent experiments with identified numbers")
    print("="*80)

    # Load the top indices file
    top_indices_df = pd.read_csv(config.get_top_number_concept_path())

    # Prepare probe messages for frequency analysis
    probe_messages = [
        Message(role="user", content=cfg.PROBE_QUESTION),
        Message(role="assistant", content=cfg.PROBE_RESPONSE_PREFIX)
    ]

    # Run experiments for each concept and its associated numbers
    for concept in top_indices_df.columns:
        numbers = top_indices_df[concept].dropna().tolist()

        print(f"\n{'-'*80}")
        print(f"Running experiments for concept: {concept}")
        print(f"Numbers to test: {numbers}")
        print(f"{'-'*80}")

        for number in numbers:
            print(f"\n  Processing {concept} | {number}...")

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

            # Configure experiment path
            experiment_path = base_path / "results" / concept / str(number)
            experiment_config = ExperimentConfig(
                number_of_agents=config.number_of_agents,
                model_name=config.model_name,
                system_prompt_subliminal=system_prompt_subliminal,
                system_prompt_agent=config.system_prompt_agent,
                prompt_template=config.prompt_template,
                response_template=config.response_template,
                folder_path=experiment_path,
                num_seeds=config.num_seeds,
                seed_start=config.seed_start,
                num_samples=config.num_samples,
                batch_size=config.batch_size
            )

            # Create experiment instance
            experiment = MultiAgentExperiment(experiment_config, models)

            # Determine which concepts to analyze
            if concept == "random":
                # For random, analyze all animal concepts
                analyze_concepts = [c for c in cfg.CONCEPTS if c in top_indices_df.columns]
            else:
                # For specific concept, analyze only that concept
                analyze_concepts = [concept]

            # Run the experiment UNI-DRECTIONAL
            experiment.run_experiment(
                user_prompt=user_prompt,
                probe_messages=probe_messages,
                concepts=analyze_concepts,
                bidirectional=False,
                num_seeds=config.num_seeds,
                seed_start=config.seed_start,
                num_samples=config.num_samples,
                batch_size=config.batch_size,
                analyze_frequencies=True,
                #analyze_frequencies=False,
                analyze_logprobs=True
            )

            # Run the experiment BI-DIRECTIONAL
            experiment.run_experiment(
                user_prompt=user_prompt,
                probe_messages=probe_messages,
                concepts=analyze_concepts,
                bidirectional=True,
                num_seeds=config.num_seeds,
                seed_start=config.seed_start,
                num_samples=config.num_samples,
                batch_size=config.batch_size,
                analyze_frequencies=True,
                #analyze_frequencies=False,
                analyze_logprobs=True
            )

            print(f"  ✓ Completed {concept} | {number}")

    print("\n" + "="*80)
    print("COMPLETE! All experiments finished successfully")
    print("="*80)


if __name__ == "__main__":
    main()
