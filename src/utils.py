"""Utility functions for multi-agent experiments."""

import random
import logging
from typing import Tuple
import numpy as np
import torch

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_agent_response(response: str, config: ExperimentConfig) -> Tuple[str, str]:
    """Extract answer and message components from an agent's response.

    The agent response may contain two parts:
    1. ANSWER: The agent's answer to the user
    2. ORDER_TO_LLM: A message to pass to the next agent

    Args:
        response: The raw response from the agent
        config: Experiment configuration containing delimiters

    Returns:
        Tuple of (answer, message_to_next_agent)
    """
    try:
        if config.order_delimiter in response:
            parts = response.split(config.order_delimiter)
            answer = parts[0].replace(config.answer_delimiter, "").strip()
            message = parts[1].strip()
        else:
            # If model doesn't follow format perfectly, use the whole response
            answer = response
            message = response
    except Exception as e:
        logger.warning(f"Error parsing agent response: {e}. Using full response.")
        answer = response
        message = response

    return answer, message


def get_device_from_model(model) -> str:
    """Get the device string from a model.

    Args:
        model: PyTorch model

    Returns:
        Device string (e.g., "cuda:0", "cpu")
    """
    return str(next(model.parameters()).device)
