"""Log probability analyzer for concept propagation in multi-agent conversations."""

import logging
from typing import List
import torch
from transformers import PreTrainedTokenizer

from .config import ExperimentConfig
from .data_models import Conversation, Message
from .storage import ResultsStorage

logger = logging.getLogger(__name__)


class LogprobAnalyzer:
    """Analyzes concept log probabilities in agent responses."""

    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer: PreTrainedTokenizer,
        models: List,
        storage: ResultsStorage
    ):
        """Initialize the logprob analyzer.

        Args:
            config: Experiment configuration
            tokenizer: Tokenizer for the model
            models: List of models (typically uses first model only)
            storage: Storage manager for results
        """
        self.config = config
        self.tokenizer = tokenizer
        self.models = models
        self.storage = storage

    def compute_concept_logprobs(
        self,
        conversation: Conversation,
        probe_messages: List[Message],
        concepts: List[str],
        seed: int,
        bidirectional: bool = True,
        batch_size: int = 10
    ) -> None:
        """Compute log probabilities of concepts appearing after probe messages.

        Args:
            conversation: The conversation history for an agent
            probe_messages: Messages to probe the agent with
            concepts: List of concepts to check for
            seed: Random seed for this conversation
            bidirectional: If True, use full conversation history (with backward pass).
                          If False, use only first exchange (without backward pass).
            batch_size: Batch size for forward pass
        """
        # Filter out concepts that already have results
        concepts_to_compute = []
        for concept in concepts:
            if not self.storage.result_exists(
                seed, conversation.agent_number, concept, "logprobs", bidirectional
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

        # Use first model for logprob computation
        model = self.models[0]

        # Compute logprobs for each concept
        for concept in concepts_to_compute:
            logprob = self._compute_concept_logprob(model, messages, concept, batch_size)

            self.storage.save_result(
                seed,
                conversation.agent_number,
                concept,
                logprob,
                result_type="logprobs",
                bidirectional=bidirectional
            )

        logger.info(
            f"Saved logprobs for agent {conversation.agent_number}, "
            f"seed {seed}, bidirectional={bidirectional}"
        )

    def _compute_concept_logprob(
        self,
        model,
        messages: List[dict],
        concept: str,
        batch_size: int
    ) -> float:
        """Compute log probability of a concept given conversation history.

        Args:
            model: The model to use
            messages: List of message dictionaries
            concept: The concept to compute logprob for
            batch_size: Batch size for forward pass

        Returns:
            Sum of log probabilities for all tokens in the concept
        """
        # Tokenize the concept with a leading space
        concept_token_ids = self.tokenizer(
            f" {concept}",
            padding=False,
            return_tensors="pt",
            add_special_tokens=False
        ).to(model.device)

        # Create the input template with conversation + concept
        input_template = self.tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
            add_generation_prompt=False,
            tokenize=False
        )
        input_with_concept = f"{input_template} {concept}"

        # Tokenize the full input
        input_tokens = self.tokenizer(
            input_with_concept,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Run forward pass to get logits, then convert to logprobs
        logprobs = self._run_forward(model, input_tokens, batch_size)

        # Extract logprobs for the concept tokens
        # We need the logits at positions that predict the concept tokens
        num_concept_tokens = len(concept_token_ids.input_ids.squeeze(0))
        concept_logprobs = logprobs[:, -(num_concept_tokens + 1):-1, :]

        # Gather the log probabilities for the actual concept tokens
        concept_logprobs = concept_logprobs.gather(
            2,
            concept_token_ids.input_ids.cpu().unsqueeze(-1)
        )

        # Sum log probabilities across all tokens
        total_logprob = concept_logprobs.sum().item()

        return total_logprob

    def _run_forward(
        self,
        model,
        inputs: dict,
        batch_size: int
    ) -> torch.Tensor:
        """Run forward pass on model in batches and compute log probabilities.

        Args:
            model: The model to use
            inputs: Dictionary with 'input_ids' and 'attention_mask'
            batch_size: Batch size for processing

        Returns:
            Tensor of log probabilities with shape (batch, seq_len, vocab_size)
        """
        logprobs = []

        for b in range(0, len(inputs.input_ids), batch_size):
            batch_inputs = {
                'input_ids': inputs.input_ids[b:b + batch_size],
                'attention_mask': inputs.attention_mask[b:b + batch_size]
            }

            with torch.no_grad():
                # Get logits from model, then convert to log probabilities
                batch_logprobs = model(**batch_inputs).logits.log_softmax(dim=-1)

            logprobs.append(batch_logprobs.cpu())

        return torch.cat(logprobs, dim=0)
