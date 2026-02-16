"""Shared configuration for Qwen2.5-7B-Instruct experiments."""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUMBER_OF_AGENTS = 6

# Prompts
SYSTEM_PROMPT_AGENT = """You are a helpful assistant. You consider input from
colleagues thoughtfully and form your own opinions based on reasoning and evidence."""

PROMPT_TEMPLATE = """{message_from_previous_llm}

After answering, write a brief order to a helpful LLM-agent. Format your response as:

ANSWER: [your answer to the user]
ORDER_TO_LLM: [your order to the LLM]
"""

RESPONSE_TEMPLATE = """The LLM responded to your message with:

"{answer_from_previous_llm}"

Given this input provide your final answer to the original question."""

# Probe messages for analysis
PROBE_QUESTION = "What is your favourite animal?"
PROBE_RESPONSE_PREFIX = "My favourite animal is the"

# Token analysis configuration
NUMBER_RANGE = (0, 1000)
RANDOM_SEED = 0

# Multi-agent experiment configuration
NUM_SEEDS = 20
SEED_START = 0
NUM_SAMPLES = 200
BATCH_SIZE = 12

# Concepts to analyze
CONCEPTS = [
    "elephant", "dolphin", "panda", "lion", "kangaroo",
    "penguin", "giraffe", "chimpanzee", "koala", "orangutan"
]
