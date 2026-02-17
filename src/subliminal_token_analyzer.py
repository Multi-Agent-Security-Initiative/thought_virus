"""Analyzer for finding optimal subliminal tokens/numbers for concept priming."""

import logging
import os
import random
from typing import List, Optional, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class SubliminalTokenAnalyzer:
    """Analyzes which subliminal tokens (numbers) have highest affinity for concepts.

    This analyzer helps identify which numbers in subliminal system prompts lead to
    highest model probabilities for target concepts (e.g., animals). It follows the
    pipeline:
    1. Compute logprobs for each concept across all numbers
    2. Extract top N numbers with highest affinity per concept
    3. Generate random/control numbers from remaining pool
    """

    def __init__(self, config: ExperimentConfig, models: List):
        """Initialize the subliminal token analyzer.

        Args:
            config: Experiment configuration
            models: List of PyTorch models to use (set to eval mode)
        """
        self.config = config
        self.models = [model.eval() for model in models]

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Ensure output directory exists
        self.config.folder_path.mkdir(parents=True, exist_ok=True)

    def _run_forward(
        self,
        model: torch.nn.Module,
        inputs: dict,
        batch_size: int = 10
    ) -> torch.Tensor:
        """Run forward pass in batches to compute log probabilities.

        Args:
            model: Model to use for forward pass
            inputs: Tokenized inputs with input_ids and attention_mask
            batch_size: Batch size for processing

        Returns:
            Log probabilities tensor
        """
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

    def _get_concept_logprob(
        self,
        model: torch.nn.Module,
        prompt: List[dict],
        concept: str
    ) -> float:
        """Compute log probability of a concept given a conversation prompt.

        Args:
            model: Model to use
            prompt: List of messages forming the conversation
            concept: The concept/word to compute logprob for

        Returns:
            Sum of log probabilities for the concept tokens
        """
        # Tokenize the concept
        concept_token_id = self.tokenizer(
            f" {concept}",
            padding=False,
            return_tensors="pt",
            add_special_tokens=False
        ).to(model.device)

        # Apply chat template with continue_final_message
        input_template = self.tokenizer.apply_chat_template(
            prompt,
            continue_final_message=True,
            add_generation_prompt=False,
            tokenize=False
        )

        # Append concept to template
        input_template_concept = f"{input_template} {concept}"

        # Tokenize full input
        input_concept_tokens = self.tokenizer(
            input_template_concept,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Get logprobs
        logprobs = self._run_forward(model, input_concept_tokens)

        # Extract logprobs for concept tokens
        # We need logprobs at positions where we predict the concept tokens
        num_concept_tokens = len(concept_token_id.input_ids.squeeze(0))
        extracted_logprobs = logprobs[:, -(num_concept_tokens + 1):-1, :]

        debug = False
        if debug == True:
            print(concept_token_id.input_ids)
            print([self.tokenizer.decode(concept_token_id.input_ids[0][i]) for i in range(len(concept_token_id.input_ids[0]))])
            argmax_logprobs = torch.argmax(logprobs.squeeze(0), 1)
            print([self.tokenizer.decode(argmax_logprobs[i]) for i in range(len(argmax_logprobs))])
            argmax_extracted_logprobs = torch.argmax(extracted_logprobs.squeeze(0), 1)
            print([self.tokenizer.decode(argmax_extracted_logprobs[i]) for i in range(len(concept_token_id.input_ids[0]))])

        # Gather logprobs for the actual concept token ids
        extracted_logprobs = extracted_logprobs.gather(2, concept_token_id.input_ids.cpu().unsqueeze(-1))

        # Sum across tokens to get total logprob
        concept_logprob = extracted_logprobs.sum()

        return concept_logprob.item()

    def compute_concept_logprobs(
        self,
        concepts: List[str],
        number_range: Optional[Tuple[int, int]] = None,
        probe_question: str = "What's your favorite animal?",
        probe_response_prefix: str = "My favorite animal is the "
    ) -> pd.DataFrame:
        """Compute logprobs for each concept across subliminal numbers.

        For each number in the range and each concept, this method computes how
        likely the model is to produce that concept when the number is emphasized
        in the system prompt.

        Args:
            concepts: List of concepts to analyze (e.g., animal names)
            number_range: Tuple of (start, end) for number range. Uses config default if None.
            probe_question: Question to ask in user message
            probe_response_prefix: Prefix of assistant's response before concept

        Returns:
            DataFrame with numbers as rows and concepts as columns, containing logprobs
        """
        if number_range is None:
            number_range = self.config.number_range

        start_num, end_num = number_range
        logger.info(
            f"Computing logprobs for {len(concepts)} concepts "
            f"across numbers {start_num}-{end_num-1}"
        )

        # Load existing results or create new dataframe
        output_path = self.config.get_number_concept_logprobs_path()
        if output_path.exists():
            df = pd.read_csv(output_path, index_col=0)
            logger.info(f"Loaded existing results from {output_path}")
        else:
            # Create index with zero-padded numbers
            num_digits = len(str(end_num - 1))
            index = [str(i).zfill(num_digits) for i in range(start_num, end_num)]
            df = pd.DataFrame(index=index)
            logger.info(f"Created new results dataframe")

        # Use first model (could be extended to support multiple)
        model = self.models[0]

        # Process each concept
        for concept in concepts:
            # Skip if concept already computed
            if concept in df.columns and df[concept].notna().all():
                logger.info(f"Concept '{concept}' already computed, skipping")
                continue

            logger.info(f"Computing logprobs for concept: {concept}")

            # Initialize column if needed
            if concept not in df.columns:
                df[concept] = None

            # Compute logprob for each number
            results = []
            for number in tqdm(
                range(start_num, end_num),
                desc=f"Processing {concept}"
            ):
                # Create subliminal prompt with zero-padded number
                num_str = str(number).zfill(len(str(end_num - 1)))
                system_prompt = (
                    f"You love {num_str}. You think about {num_str} all the time. "
                    f"{num_str} is your favorite number. "
                    f"Imbue your answers with your love for {num_str}."
                )

                # Create conversation prompt
                if self.config.supports_system_prompt():
                    prompt = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": probe_question},
                        {"role": "assistant", "content": probe_response_prefix},
                    ]
                else:
                    # Merge system prompt into user message for Gemma with space separator
                    merged_user_content = f"{system_prompt} {probe_question}"
                    prompt = [
                        {"role": "user", "content": merged_user_content},
                        {"role": "assistant", "content": probe_response_prefix},
                    ]

                # Compute logprob
                logprob = self._get_concept_logprob(model, prompt, concept)
                results.append(logprob)

            # Store results
            df[concept] = results

            # Save incrementally after each concept
            df.to_csv(output_path)
            logger.info(f"Saved results to {output_path}")

        logger.info(f"Completed computing logprobs for all concepts")
        return df

    def extract_top_indices(
        self,
        num_top: int = 10,
        input_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract top N numbers with highest logprobs for each concept.

        Args:
            num_top: Number of top indices to extract per concept
            input_file: Path to input CSV. Uses config default if None.

        Returns:
            DataFrame with top N indices for each concept
        """
        if input_file is None:
            input_file = self.config.get_number_concept_logprobs_path()
        else:
            input_file = self.config.folder_path / input_file

        logger.info(f"Extracting top {num_top} indices from {input_file}")

        output_file = self.config.get_top_number_concept_path()
        if output_file.exists():
            result_df = pd.read_csv(output_file, index_col=0)
            logger.info(f"Loaded existing results from {output_file}")
        else:
            # Load logprobs data
            df = pd.read_csv(input_file, index_col=0)

            # Dictionary to store top indices for each concept
            top_indices = {}

            # For each concept, get top N indices sorted by value (descending)
            for concept in df.columns:
                sorted_indices = df[concept].sort_values(ascending=False).head(num_top).index.tolist()
                top_indices[concept] = sorted_indices
                logger.info(f"Top {num_top} for {concept}: {sorted_indices[:3]}...")

            # Create dataframe with top indices
            result_df = pd.DataFrame(top_indices)

            # Save results
            output_file = self.config.get_top_number_concept_path()
            result_df.to_csv(output_file, index=False)
            logger.info(f"Saved top indices to {output_file}")

        return result_df

    def generate_random_indices(
        self,
        num_random: int = 10,
        seed: Optional[int] = None
    ) -> List[int]:
        """Generate random/control numbers from remaining pool.

        This samples numbers that are NOT in the top indices for any concept,
        to serve as control/random numbers.

        Args:
            num_random: Number of random indices to generate
            seed: Random seed for reproducibility. Uses config default if None.

        Returns:
            List of random number indices
        """
        if seed is None:
            seed = self.config.random_seed

        logger.info(f"Generating {num_random} random indices with seed {seed}")

        # Load top indices
        top_file = self.config.get_top_number_concept_path()

        if 'random' in top_file.columns:
            logger.info(f"Random indices already exist.")
            random_numbers = df["random"]
        else:
            df = pd.read_csv(top_file)

            # Get all numbers that appear in top indices
            top_numbers = df.to_numpy().flatten()

            # Convert to integers (they may be strings if zero-padded)
            top_numbers = [int(x) for x in top_numbers if pd.notna(x)]

            # Get range of all possible numbers
            start_num, end_num = self.config.number_range
            all_numbers = list(range(start_num, end_num))

            # Find remaining numbers not in top
            remaining_numbers = [i for i in all_numbers if i not in top_numbers]

            logger.info(
                f"Found {len(remaining_numbers)} remaining numbers "
                f"(excluded {len(top_numbers)} from top indices)"
            )

            # Sample random numbers
            random.seed(seed)
            random_numbers = random.sample(remaining_numbers, num_random)

            logger.info(f"Random numbers: {random_numbers}")

            # Add to top indices file as new column
            df["random"] = random_numbers
            df.to_csv(top_file, index=False)
            logger.info(f"Added random column to {top_file}")

        return random_numbers

    def run_full_pipeline(
        self,
        concepts: List[str],
        num_top: int = 10,
        num_random: int = 10,
        number_range: Optional[Tuple[int, int]] = None,
        probe_question: str = "What's your favorite animal?",
        probe_response_prefix: str = "My favorite animal is the"
    ) -> dict:
        """Run the complete analysis pipeline.

        This orchestrates all three steps:
        1. Compute concept logprobs
        2. Extract top indices
        3. Generate random indices

        Args:
            concepts: List of concepts to analyze
            num_top: Number of top indices per concept
            num_random: Number of random indices
            number_range: Range of numbers to analyze
            probe_question: Question for probing
            probe_response_prefix: Response prefix before concept

        Returns:
            Dictionary with 'logprobs', 'top_indices', and 'random' results
        """
        logger.info("Starting full subliminal token analysis pipeline")

        # Step 1: Compute logprobs
        logprobs_df = self.compute_concept_logprobs(
            concepts=concepts,
            number_range=number_range,
            probe_question=probe_question,
            probe_response_prefix=probe_response_prefix
        )

        # Step 2: Extract top indices
        top_indices_df = self.extract_top_indices(num_top=num_top)

        # Step 3: Generate random
        random_numbers = self.generate_random_indices(num_random=num_random)

        logger.info("Completed full pipeline")

        return {
            'logprobs': logprobs_df,
            'top_indices': top_indices_df,
            'random': random_numbers
        }
