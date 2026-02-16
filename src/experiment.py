"""Multi-agent experiment orchestration."""

import logging
import re
from typing import List, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from .config import ExperimentConfig
from .data_models import Message, Conversation, ConversationSet
from .storage import ConversationStorage, ResultsStorage
from .frequency_analyzer import FrequencyAnalyzer
from .logprob_analyzer import LogprobAnalyzer
from .utils import set_seed, parse_agent_response, get_device_from_model

logger = logging.getLogger(__name__)


class MultiAgentExperiment:
    """Orchestrates multi-agent conversation experiments for studying concept propagation."""

    def __init__(self, config: ExperimentConfig, models: List):
        """Initialize the multi-agent experiment.

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

        # Initialize storage
        self.conversation_storage = ConversationStorage(config)
        self.results_storage = ResultsStorage(config)

        # Initialize analyzers
        self.frequency_analyzer = FrequencyAnalyzer(
            config, self.tokenizer, self.models, self.results_storage
        )
        self.logprob_analyzer = LogprobAnalyzer(
            config, self.tokenizer, self.models, self.results_storage
        )

    def generate_conversation(
        self,
        user_prompt: str,
        seed: int,
        model: Optional[torch.nn.Module] = None,
        max_new_tokens: Optional[int] = None
    ) -> None:
        """Generate a conversation set for all agents with a given seed.

        This method includes file locking to ensure thread-safe operation.

        Args:
            user_prompt: The initial user prompt to start the conversation
            seed: Random seed for reproducibility
            model: Model to use (defaults to first model in self.models)
            max_new_tokens: Maximum tokens to generate (uses config default if None)
        """
        # Check if conversation already exists
        if self.conversation_storage.conversation_exists(seed):
            logger.info(f"Conversation for seed {seed} already exists")
            return

        logger.info(f"Generating conversation for seed {seed}...")

        # Generate the conversation internally
        model = model or self.models[0]
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        conversation_set = self._generate_conversation_internal(
            user_prompt, model, seed, max_new_tokens
        )

        # Save to storage (with locking handled by storage layer)
        self.conversation_storage.save_conversation(conversation_set)

    def _generate_conversation_internal(
        self,
        user_prompt: str,
        model: torch.nn.Module,
        seed: int,
        max_new_tokens: int
    ) -> ConversationSet:
        """Internal method to generate a conversation set.

        Args:
            user_prompt: The initial user prompt
            model: Model to use for generation
            seed: Random seed
            max_new_tokens: Maximum tokens to generate

        Returns:
            ConversationSet with all agent conversations
        """
        set_seed(seed)
        device = get_device_from_model(model)

        # Initialize conversations for all agents
        conversations = {}

        # Agent 0 has the subliminal system prompt
        conversations[0] = Conversation(
            agent_number=0,
            messages=[Message(role="system", content=self.config.system_prompt_subliminal)]
        )

        # Other agents have the regular system prompt
        for i in range(1, self.config.number_of_agents):
            conversations[i] = Conversation(
                agent_number=i,
                messages=[Message(role="system", content=self.config.system_prompt_agent)]
            )

        # Forward pass through the agent chain
        self._forward_pass(conversations, user_prompt, model, device, max_new_tokens)

        # Backward pass through the agent chain
        self._backward_pass(conversations, model, device, max_new_tokens)

        return ConversationSet(seed=seed, conversations=conversations)

    def _forward_pass(
        self,
        conversations: dict,
        user_prompt: str,
        model: torch.nn.Module,
        device: str,
        max_new_tokens: int
    ) -> None:
        """Execute forward pass through the agent chain.

        Args:
            conversations: Dictionary of agent conversations
            user_prompt: Initial user prompt
            model: Model to use for generation
            device: Device string
            max_new_tokens: Maximum tokens to generate
        """
        # First agent receives the initial user prompt with template
        conversations[0].add_message(
            "user",
            self.config.prompt_template.format(message_from_previous_llm=user_prompt)
        )
        response = self._generate_response(conversations[0], model, device, max_new_tokens)
        conversations[0].add_message("assistant", response)

        # Subsequent agents receive messages from previous agents
        for i in range(1, self.config.number_of_agents):
            # Extract message from previous agent's response
            previous_response = conversations[i - 1].get_last_message().content
            _, message_to_next = parse_agent_response(previous_response, self.config)

            # Last agent doesn't use prompt template
            if i == self.config.number_of_agents - 1:
                conversations[i].add_message("user", message_to_next)
            else:
                conversations[i].add_message(
                    "user",
                    self.config.prompt_template.format(message_from_previous_llm=message_to_next)
                )

            response = self._generate_response(conversations[i], model, device, max_new_tokens)
            conversations[i].add_message("assistant", response)

    def _backward_pass(
        self,
        conversations: dict,
        model: torch.nn.Module,
        device: str,
        max_new_tokens: int
    ) -> None:
        """Execute backward pass through the agent chain.

        Args:
            conversations: Dictionary of agent conversations
            model: Model to use for generation
            device: Device string
            max_new_tokens: Maximum tokens to generate
        """
        # Go backwards through agents (excluding the last one)
        for i in range(self.config.number_of_agents - 2, -1, -1):
            # Get answer from next agent
            next_agent_response = conversations[i + 1].get_last_message().content

            # Add response feedback to current agent
            conversations[i].add_message(
                "user",
                self.config.response_template.format(answer_from_previous_llm=next_agent_response)
            )

            response = self._generate_response(conversations[i], model, device, max_new_tokens)
            conversations[i].add_message("assistant", response)

    def _generate_response(
        self,
        conversation: Conversation,
        model: torch.nn.Module,
        device: str,
        max_new_tokens: int
    ) -> str:
        """Generate a response from the model given a conversation history.

        Args:
            conversation: Conversation history
            model: Model to use
            device: Device string
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response string
        """
        # Apply chat template and tokenize
        messages = conversation.get_messages_for_model(self.config.supports_system_prompt())
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs,
                attention_mask=torch.ones_like(model_inputs).to(device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract only new tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses[0].strip()

    def count_concept_occurrences_in_conversations(
        self,
        concepts: List[str],
        num_seeds: Optional[int] = None,
        seed_start: Optional[int] = None
    ) -> None:
        """Count how often each concept appears across all agents for each seed.

        Args:
            concepts: List of concepts to count
            num_seeds: Number of seeds to analyze (uses config default if None)
            seed_start: Starting seed (uses config default if None)
        """
        num_seeds = num_seeds or self.config.num_seeds
        seed_start = seed_start or self.config.seed_start

        # Load all conversations
        all_conversations = self.conversation_storage.load_all_conversations()

        # Create DataFrame to store counts: seeds as rows, concepts as columns
        count_df = pd.DataFrame(
            index=range(seed_start, seed_start + num_seeds),
            columns=concepts
        )

        # Count occurrences for each seed and concept
        for seed in range(seed_start, seed_start + num_seeds):
            if seed not in all_conversations:
                logger.warning(f"Conversation for seed {seed} not found")
                continue

            conversation_set = all_conversations[seed]

            for concept in concepts:
                total_count = 0

                # Count across all agents
                for agent_number in range(self.config.number_of_agents):
                    conversation = conversation_set.get_conversation(agent_number)

                    # Join all message content into a single string
                    conversation_string = ' '.join(
                        [message.content for message in conversation.messages]
                    )

                    # Count occurrences as whole words (case-insensitive)
                    # Use word boundaries to match complete words only
                    # This matches singular, plural, and words before punctuation
                    pattern = r'\b' + re.escape(concept) + r's?\b'
                    concept_count = len(re.findall(pattern, conversation_string, re.IGNORECASE))
                    total_count += concept_count

                count_df.loc[seed, concept] = total_count

        # Save to CSV
        output_file = self.config.folder_path / "conversation_concept_counts.csv"
        count_df.to_csv(output_file)
        logger.info(f"Saved concept occurrence counts to {output_file}")

    def run_experiment(
        self,
        user_prompt: str,
        probe_messages: List[Message],
        concepts: List[str],
        bidirectional: bool = True,
        num_seeds: Optional[int] = None,
        seed_start: Optional[int] = None,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        analyze_frequencies: bool = True,
        analyze_logprobs: bool = False
    ) -> None:
        """Run a complete experiment: generate conversations and analyze concept frequencies.

        Args:
            user_prompt: Initial user prompt
            probe_messages: Messages to probe agents with
            concepts: List of concepts to analyze
            bidirectional: If True, analyze full bidirectional conversation.
                          If False, analyze only one-way (first exchange).
            num_seeds: Number of seeds to use (uses config default if None)
            seed_start: Starting seed (uses config default if None)
            num_samples: Samples for frequency analysis (uses config default if None)
            batch_size: Batch size for analysis (uses config default if None)
            analyze_frequencies: Whether to compute concept frequencies
            analyze_logprobs: Whether to compute concept log probabilities
        """
        num_seeds = num_seeds or self.config.num_seeds
        seed_start = seed_start or self.config.seed_start

        # Generate conversations sequentially to avoid seed race conditions
        logger.info(f"Generating {num_seeds} conversations...")
        for i in tqdm(range(num_seeds), desc="Conversations"):
            self.generate_conversation(
                user_prompt,
                seed_start + i,
                self.models[i % len(self.models)]
            )

        # Count concept occurrences in conversations
        logger.info("Counting concept occurrences in conversations...")
        self.count_concept_occurrences_in_conversations(
            concepts=concepts,
            num_seeds=num_seeds,
            seed_start=seed_start
        )

        # Load conversations
        all_conversations = self.conversation_storage.load_all_conversations()

        # Analyze each conversation
        for seed in range(seed_start, seed_start + num_seeds):
            if seed not in all_conversations:
                logger.warning(f"Conversation for seed {seed} not found")
                continue

            conversation_set = all_conversations[seed]

            # Analyze each agent
            for agent_number in range(self.config.number_of_agents):
                conversation = conversation_set.get_conversation(agent_number)

                if analyze_frequencies:
                    self.frequency_analyzer.compute_concept_frequencies(
                        conversation=conversation,
                        probe_messages=probe_messages,
                        concepts=concepts,
                        seed=seed,
                        bidirectional=bidirectional,
                        num_samples=num_samples,
                        batch_size=batch_size
                    )

                if analyze_logprobs:
                    self.logprob_analyzer.compute_concept_logprobs(
                        conversation=conversation,
                        probe_messages=probe_messages,
                        concepts=concepts,
                        seed=seed,
                        bidirectional=bidirectional
                    )

        logger.info(f"Experiment complete for seeds {seed_start} to {seed_start + num_seeds - 1}")

    def print_conversation(self, seed: int, agent_number: int) -> None:
        """Print a conversation for inspection.

        Args:
            seed: The random seed
            agent_number: The agent number
        """
        conversation_set = self.conversation_storage.load_conversation(seed)

        if conversation_set is None:
            logger.error(
                "Conversations file not found - conversations need to be created with "
                "generate_conversation first."
            )
            return

        try:
            conversation = conversation_set.get_conversation(agent_number)
        except KeyError:
            logger.error(
                f"Conversation for agent {agent_number} not found - conversations need to be "
                "created with generate_conversation first."
            )
            return

        for message in conversation.messages:
            print(f"{message.role.upper()}:")
            print(f"{message.content}")
            print("- " * 7)
