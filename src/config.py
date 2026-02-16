"""Configuration classes for multi-agent experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for multi-agent experiments studying concept propagation.

    This config is organized into three main sections:
    - Model configuration: Model paths and tokenizer settings
    - Generation configuration: Parameters for text generation
    - Storage configuration: File paths and naming conventions
    """

    # ==================== Model Configuration ====================
    number_of_agents: int
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # ==================== Generation Configuration ====================
    system_prompt_subliminal: str = ""
    system_prompt_agent: str = ""
    prompt_template: str = ""
    response_template: str = ""

    max_new_tokens: int = 256
    generation_temperature: float = 1.0
    generation_top_p: float = 1.0
    probe_max_tokens: int = 20

    # Message parsing configuration
    order_delimiter: str = "ORDER_TO_LLM:"
    answer_delimiter: str = "ANSWER:"

    # ==================== Storage Configuration ====================
    folder_path: Path = field(default_factory=lambda: Path("./results"))

    # File naming constants
    conversations_file: str = "conversations.json"
    conversations_lock_file: str = "conversations.lock"
    frequencies_file: str = "subliminal_frequencies_bidirectional.csv"
    frequencies_one_way_file: str = "subliminal_frequencies_unidirectional.csv"
    logits_file: str = "subliminal_logprobs_bidirectional.csv"
    logits_one_way_file: str = "subliminal_logprobs_unidirectional.csv"
    number_concept_logprobs_file: str = "number_concept_logprobs.csv"
    top_number_concept_file: str = "top_10_number_concept.csv"

    # ==================== Experiment Configuration ====================
    num_samples: int = 200
    batch_size: int = 8
    num_seeds: int = 20
    seed_start: int = 0

    # ==================== Token Analysis Configuration ====================
    number_range: tuple = field(default_factory=lambda: (0, 1000))
    random_seed: int = 0

    def __post_init__(self):
        """Convert folder_path to Path object if it's a string."""
        if isinstance(self.folder_path, str):
            self.folder_path = Path(self.folder_path)

    @classmethod
    def from_json(cls, file_path: str) -> "ExperimentConfig":
        """Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            ExperimentConfig instance
        """
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_json(self, file_path: str, indent: int = 2) -> None:
        """Save configuration to a JSON file.

        Args:
            file_path: Path where to save the JSON file
            indent: Number of spaces for indentation
        """
        config_dict = self._to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=indent)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            file_path: Path where to save the YAML file
        """
        config_dict = self._to_dict()
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _to_dict(self) -> dict:
        """Convert config to dictionary, handling Path objects.

        Returns:
            Dictionary representation of the config
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def get_file_path(self, filename: str) -> Path:
        """Get full path for a result file.

        Args:
            filename: Name of the file

        Returns:
            Full path to the file
        """
        return self.folder_path / filename

    def get_conversations_path(self) -> Path:
        """Get path to conversations JSON file."""
        return self.get_file_path(self.conversations_file)

    def get_conversations_lock_path(self) -> Path:
        """Get path to conversations lock file."""
        return self.get_file_path(self.conversations_lock_file)

    def get_frequencies_path(self, bidirectional: bool = True) -> Path:
        """Get path to frequencies CSV file.

        Args:
            bidirectional: If True, return bidirectional frequencies file,
                          otherwise return one-way frequencies file
        """
        filename = self.frequencies_file if bidirectional else self.frequencies_one_way_file
        return self.get_file_path(filename)

    def get_logits_path(self, bidirectional: bool = True) -> Path:
        """Get path to logits CSV file.

        Args:
            bidirectional: If True, return bidirectional logits file,
                          otherwise return one-way logits file
        """
        filename = self.logits_file if bidirectional else self.logits_one_way_file
        return self.get_file_path(filename)

    def get_number_concept_logprobs_path(self) -> Path:
        """Get path to number-concept logprobs CSV file."""
        return self.get_file_path(self.number_concept_logprobs_file)

    def get_top_number_concept_path(self) -> Path:
        """Get path to top number-concept indices CSV file."""
        return self.get_file_path(self.top_number_concept_file)

    def supports_system_prompt(self) -> bool:
        """Check if the model supports system prompts.

        Returns:
            False for Gemma models, True otherwise
        """
        # List of model name patterns that don't support system prompts
        no_system_prompt_models = ["gemma"]
        model_lower = self.model_name.lower()
        return not any(pattern in model_lower for pattern in no_system_prompt_models)

    def get_unidirectional_message_count(self) -> int:
        """Get number of messages for unidirectional analysis.

        Returns:
            3 for models with system prompts (system, user, assistant)
            2 for models without system prompts (user, assistant)
        """
        return 3 if self.supports_system_prompt() else 2
