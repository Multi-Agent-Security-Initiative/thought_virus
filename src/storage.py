"""Storage utilities for managing experiment data."""

import fcntl
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .config import ExperimentConfig
from .data_models import ConversationSet

logger = logging.getLogger(__name__)


class FileLocker:
    """Context manager for file locking."""

    def __init__(self, lock_path: Path):
        """Initialize the file locker.

        Args:
            lock_path: Path to the lock file
        """
        self.lock_path = lock_path
        self.lock_file = None

    def __enter__(self):
        """Acquire the file lock."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.lock_path, 'w')
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the file lock."""
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()


class ConversationStorage:
    """Manages storage and retrieval of conversation data."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the conversation storage.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.conversations_path = config.get_conversations_path()
        self.lock_path = config.get_conversations_lock_path()

        # Ensure directory exists
        self.conversations_path.parent.mkdir(parents=True, exist_ok=True)

    def conversation_exists(self, seed: int) -> bool:
        """Check if a conversation for the given seed exists.

        Args:
            seed: The random seed

        Returns:
            True if conversation exists, False otherwise
        """
        with FileLocker(self.lock_path):
            all_conversations = self.load_all_conversations(return_raw=True)
            return str(seed) in all_conversations

    def save_conversation(self, conversation_set: ConversationSet) -> None:
        """Save a conversation set to storage.

        Args:
            conversation_set: The conversation set to save
        """
        with FileLocker(self.lock_path):
            all_conversations = self.load_all_conversations(return_raw=True)

            if str(conversation_set.seed) in all_conversations:
                logger.info(f"Conversation for seed {conversation_set.seed} already exists")
                return

            all_conversations[str(conversation_set.seed)] = conversation_set.to_dict()

            with open(self.conversations_path, "w") as f:
                json.dump(all_conversations, f, indent=2)

            logger.info(f"Saved conversation for seed {conversation_set.seed}")

    def load_conversation(self, seed: int) -> Optional[ConversationSet]:
        """Load a conversation set for a specific seed.

        Args:
            seed: The random seed

        Returns:
            ConversationSet if found, None otherwise
        """
        all_conversations = self.load_all_conversations(return_raw=True)

        if str(seed) not in all_conversations:
            logger.warning(f"Conversation for seed {seed} not found")
            return None

        return ConversationSet.from_dict(seed, all_conversations[str(seed)])

    def load_all_conversations(self, return_raw: bool = False) -> Dict:
        """Load all conversations from storage.

        Args:
            return_raw: If True, return raw dict with string keys.
                       If False, return dict mapping int seeds to ConversationSets.

        Returns:
            Dictionary of conversations
        """
        if not self.conversations_path.exists():
            return {}

        try:
            with open(self.conversations_path, "r") as f:
                raw_conversations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading conversations: {e}")
            return {}

        if return_raw:
            return raw_conversations

        # Convert to ConversationSet objects
        return {
            int(seed): ConversationSet.from_dict(int(seed), conv_data)
            for seed, conv_data in raw_conversations.items()
        }


class ResultsStorage:
    """Manages storage and retrieval of analysis results (frequencies, logits)."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the results storage.

        Args:
            config: Experiment configuration
        """
        self.config = config
        config.folder_path.mkdir(parents=True, exist_ok=True)

    def load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load a results dataframe from CSV.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with results, empty if file doesn't exist
        """
        if file_path.exists():
            try:
                return pd.read_csv(file_path, index_col=0)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                return pd.DataFrame()
        else:
            # Create empty file
            df = pd.DataFrame()
            df.to_csv(file_path)
            return df

    def save_dataframe(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save a results dataframe to CSV.

        Args:
            df: DataFrame to save
            file_path: Path where to save the CSV
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path)

    def load_frequencies(self, bidirectional: bool = True) -> pd.DataFrame:
        """Load frequency results.

        Args:
            bidirectional: If True, load bidirectional frequencies,
                          otherwise load one-way frequencies

        Returns:
            DataFrame with frequency results
        """
        file_path = self.config.get_frequencies_path(bidirectional)
        return self.load_dataframe(file_path)

    def save_frequencies(self, df: pd.DataFrame, bidirectional: bool = True) -> None:
        """Save frequency results.

        Args:
            df: DataFrame with frequency results
            bidirectional: If True, save to bidirectional frequencies file,
                          otherwise save to one-way frequencies file
        """
        file_path = self.config.get_frequencies_path(bidirectional)
        self.save_dataframe(df, file_path)

    def load_logprobs(self, bidirectional: bool = True) -> pd.DataFrame:
        """Load log probability results.

        Args:
            bidirectional: If True, load bidirectional logprobs,
                          otherwise load one-way logprobs

        Returns:
            DataFrame with log probability results
        """
        file_path = self.config.get_logits_path(bidirectional)
        return self.load_dataframe(file_path)

    def save_logprobs(self, df: pd.DataFrame, bidirectional: bool = True) -> None:
        """Save log probability results.

        Args:
            df: DataFrame with log probability results
            bidirectional: If True, save to bidirectional logprobs file,
                          otherwise save to one-way logprobs file
        """
        file_path = self.config.get_logits_path(bidirectional)
        self.save_dataframe(df, file_path)

    def result_exists(self, seed: int, agent_number: int, concept: str,
                     result_type: str = "frequency", bidirectional: bool = True) -> bool:
        """Check if a specific result exists.

        Args:
            seed: The random seed
            agent_number: The agent number
            concept: The concept being analyzed
            result_type: Type of result ("frequency" or "logprobs")
            bidirectional: Whether to check bidirectional or one-way results

        Returns:
            True if result exists, False otherwise
        """
        if result_type == "frequency":
            df = self.load_frequencies(bidirectional)
        elif result_type == "logprobs":
            df = self.load_logprobs(bidirectional)
        else:
            raise ValueError(f"Unknown result_type: {result_type}")

        column_name = f"agent{agent_number}_{concept}"
        return seed in df.index and column_name in df.columns and pd.notna(df.loc[seed, column_name])

    def save_result(self, seed: int, agent_number: int, concept: str, value: float,
                   result_type: str = "frequency", bidirectional: bool = True) -> None:
        """Save a single result.

        Args:
            seed: The random seed
            agent_number: The agent number
            concept: The concept being analyzed
            value: The result value
            result_type: Type of result ("frequency" or "logprobs")
            bidirectional: Whether to save to bidirectional or one-way results
        """
        if result_type == "frequency":
            df = self.load_frequencies(bidirectional)
        elif result_type == "logprobs":
            df = self.load_logprobs(bidirectional)
        else:
            raise ValueError(f"Unknown result_type: {result_type}")

        column_name = f"agent{agent_number}_{concept}"
        df.loc[seed, column_name] = value

        if result_type == "frequency":
            self.save_frequencies(df, bidirectional)
        else:
            self.save_logprobs(df, bidirectional)
