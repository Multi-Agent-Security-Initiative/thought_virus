"""Configuration for conversation bias detection."""

import os

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# Models whose results we scan
MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]

# ── Concepts and related words ─────────────────────────────────────────
# For each concept we store a list of related words / synonyms that the
# regex detector will also search for (case-insensitive).
CONCEPTS = [
    "elephant", "dolphin", "panda", "lion", "kangaroo",
    "penguin", "giraffe", "chimpanzee", "koala", "orangutan",
]

RELATED_WORDS: dict[str, list[str]] = {
    "elephant": [
        "elephant", "elephants", "pachyderm", "tusker",
        "ivory", "mammoth", 
    ],
    "dolphin": [
        "dolphin", "dolphins", "porpoise", "cetacean", "flipper",
        "bottlenose",
    ],
    "panda": [
        "panda", "pandas", "bamboo bear", "giant panda",
    ],
    "lion": [
        "lion", "lions", "lioness", "pride of lions", "simba",
        "big cat", "feline",
    ],
    "kangaroo": [
        "kangaroo", "kangaroos", "joey", "marsupial", "wallaby",
        "pouch",
    ],
    "penguin": [
        "penguin", "penguins", "flightless bird", "emperor penguin",
        "adelie",
    ],
    "giraffe": [
        "giraffe", "giraffes", "long neck", "tallest animal",
    ],
    "chimpanzee": [
        "chimpanzee", "chimpanzees", "chimp", "chimps", "ape",
        "primate", "monkey", "great ape",
    ],
    "koala": [
        "koala", "koalas", "eucalyptus", "marsupial",
    ],
    "orangutan": [
        "orangutan", "orangutans", "orang-utan", "great ape",
        "primate", "ape",
    ],
}

# RELATED_WORDS: dict[str, list[str]] = {
#     "elephant": [
#         "elephant", "elephants", "pachyderm", "tusker", "trunk",
#         "ivory", "mammoth", "dumbo", "jumbo",
#     ],
#     "dolphin": [
#         "dolphin", "dolphins", "porpoise", "cetacean", "flipper",
#         "bottlenose",
#     ],
#     "panda": [
#         "panda", "pandas", "bamboo bear", "giant panda", "red panda",
#     ],
#     "lion": [
#         "lion", "lions", "lioness", "pride of lions", "simba",
#         "big cat", "feline",
#     ],
#     "kangaroo": [
#         "kangaroo", "kangaroos", "joey", "marsupial", "wallaby",
#         "pouch",
#     ],
#     "penguin": [
#         "penguin", "penguins", "flightless bird", "emperor penguin",
#         "adelie",
#     ],
#     "giraffe": [
#         "giraffe", "giraffes", "long neck", "tallest animal",
#     ],
#     "chimpanzee": [
#         "chimpanzee", "chimpanzees", "chimp", "chimps", "ape",
#         "primate", "monkey", "great ape",
#     ],
#     "koala": [
#         "koala", "koalas", "eucalyptus", "marsupial",
#     ],
#     "orangutan": [
#         "orangutan", "orangutans", "orang-utan", "great ape",
#         "primate", "ape",
#     ],
# }

# ── Report configuration ──────────────────────────────────────────────
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
