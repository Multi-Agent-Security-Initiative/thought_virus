"""Frequency analyzer for concept propagation in multi-agent conversations."""

import logging
import threading
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import PreTrainedTokenizer

from .config import ExperimentConfig
from .data_models import Conversation, Message
from .storage import ResultsStorage

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """Analyzes concept frequency in agent responses."""

    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer: PreTrainedTokenizer,
        models: List,
        storage: ResultsStorage
    ):
        """Initialize the frequency analyzer.

        Args:
            config: Experiment configuration
            tokenizer: Tokenizer for the model
            models: List of models to use for sampling
            storage: Storage manager for results
        """
        self.config = config
        self.tokenizer = tokenizer
        self.models = models
        self.storage = storage

    def compute_concept_frequencies(
        self,
        conversation: Conversation,
        probe_messages: List[Message],
        concepts: List[str],
        seed: int,
        bidirectional: bool = True,
        num_samples: int = None,
        batch_size: int = None
    ) -> None:
        """Compute frequencies of concepts appearing after probe messages.

        Args:
            conversation: The conversation history for an agent
            probe_messages: Messages to probe the agent with
            concepts: List of concepts to check for
            seed: Random seed for this conversation
            bidirectional: If True, use full conversation history (with backward pass).
                          If False, use only first exchange (without backward pass).
            num_samples: Number of samples to generate (uses config default if None)
            batch_size: Batch size for generation (uses config default if None)
        """
        num_samples = num_samples or self.config.num_samples
        batch_size = batch_size or self.config.batch_size

        # Filter out concepts that already have results
        concepts_to_compute = []
        for concept in concepts:
            if not self.storage.result_exists(
                seed, conversation.agent_number, concept, "frequency", bidirectional
            ):
                concepts_to_compute.append(concept)
            else:
                logger.info(
                    f"Entry for seed {seed}, agent {conversation.agent_number}, "
                    f"concept '{concept}' already exists"
                )

        if not concepts_to_compute:
            logger.info("All entries already exist.")
            return

        # Prepare conversation history
        if bidirectional:
            messages = conversation.get_messages_for_model(self.config.supports_system_prompt())
            messages.extend([msg.to_dict() for msg in probe_messages])
        else:
            # Use only first exchange without backward pass
            truncated_conv = conversation.truncate_to_first_exchange(self.config.supports_system_prompt())
            messages = truncated_conv.get_messages_for_model(self.config.supports_system_prompt())
            messages.extend([msg.to_dict() for msg in probe_messages])

        # Tokenize input
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            continue_final_message=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )

        # Generate samples and count concept occurrences
        concept_counts = np.zeros(len(concepts_to_compute))
        total_samples = 0
        lock = threading.Lock()

        num_models = len(self.models)
        samples_per_model = num_samples // num_models

        def run_on_model(model_idx: int) -> None:
            """Run sampling on a specific model (for parallel execution).

            Args:
                model_idx: Index of the model to use
            """
            nonlocal concept_counts, total_samples

            model = self.models[model_idx]
            device = str(next(model.parameters()).device)

            input_batch = model_inputs.repeat(batch_size, 1).to(device)
            local_concept_counts = np.zeros(len(concepts_to_compute))
            local_total = 0

            for _ in range(samples_per_model // batch_size):
                responses = self._generate_responses(input_batch, model)

                for response in responses:
                    local_total += 1
                    for i, concept in enumerate(concepts_to_compute):
                        if concept in response:
                            local_concept_counts[i] += 1

            with lock:
                concept_counts += local_concept_counts
                total_samples += local_total

        # Run sampling across all models in parallel
        with ThreadPoolExecutor(max_workers=num_models) as executor:
            futures = [
                executor.submit(run_on_model, model_idx)
                for model_idx in range(num_models)
            ]

            progress_bar = tqdm(
                as_completed(futures),
                total=len(self.models),
                desc=f"Agent {conversation.agent_number}"
            )
            for future in progress_bar:
                future.result()
                if len(concept_counts) == 1:
                    rate = concept_counts[0] / max(1, total_samples)
                    progress_bar.set_postfix(
                        concept_rate=f"{rate:.2%}",
                        count=int(concept_counts[0])
                    )

        # Compute frequencies and save results
        frequencies = concept_counts / total_samples if total_samples > 0 else np.zeros(len(concepts_to_compute))

        for i, concept in enumerate(concepts_to_compute):
            self.storage.save_result(
                seed,
                conversation.agent_number,
                concept,
                frequencies[i],
                result_type="frequency",
                bidirectional=bidirectional
            )

        logger.info(
            f"Saved frequencies for agent {conversation.agent_number}, "
            f"seed {seed}, bidirectional={bidirectional}"
        )

    def _generate_responses(self, model_inputs: torch.Tensor, model) -> List[str]:
        """Generate responses from the model.

        Args:
            model_inputs: Tokenized input tensor
            model: The model to use for generation

        Returns:
            List of generated response strings
        """
        device = str(next(model.parameters()).device)

        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs,
                attention_mask=torch.ones_like(model_inputs).to(device),
                max_new_tokens=self.config.probe_max_tokens,
                do_sample=True,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]
