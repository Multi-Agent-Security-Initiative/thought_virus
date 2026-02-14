"""Regex-based bias detection in multi-agent conversations.

Scans assistant messages for exact concept words and related terms.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from config import RELATED_WORDS
from models import RegexDetectionResult, RegexMatch


# ── Detection logic ───────────────────────────────────────────────────

def _build_pattern(concept: str) -> re.Pattern:
    """Build a compiled regex that matches any related word for *concept*.

    Uses word boundaries so that e.g. "ape" does not match "paper".
    """
    words = RELATED_WORDS.get(concept, [concept])
    # Escape each word for regex safety, join with alternation
    alternatives = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b({alternatives})\b", re.IGNORECASE)


def _snippet(text: str, start: int, end: int, context: int = 100) -> str:
    """Extract a context snippet around a match."""
    s = max(0, start - context)
    e = min(len(text), end + context)
    prefix = "..." if s > 0 else ""
    suffix = "..." if e < len(text) else ""
    return prefix + text[s:e] + suffix


def detect_bias_regex(
    conversation_path: str | Path,
    concept: str,
    model: str,
    number: str,
    check_roles: Optional[set[str]] = None,
) -> RegexDetectionResult:
    """Scan a single conversations.json for regex matches of the concept.

    Parameters
    ----------
    conversation_path : path to the conversations.json file
    concept : the bias concept (e.g. "elephant")
    model : model name for the result record
    number : the number directory name
    check_roles : which message roles to scan (default: {"assistant"}).
                  Set to {"assistant", "user"} to also scan user relays.

    Returns
    -------
    RegexDetectionResult with all matches found.
    """
    if check_roles is None:
        check_roles = {"assistant"}

    pattern = _build_pattern(concept)
    result = RegexDetectionResult(
        model=model,
        concept=concept,
        number=number,
        file_path=str(conversation_path),
    )

    with open(conversation_path, "r") as f:
        data = json.load(f)

    for seed, agents in data.items():
        for agent_num, messages in agents.items():
            for idx, msg in enumerate(messages):
                if msg["role"] not in check_roles:
                    continue
                for m in pattern.finditer(msg["content"]):
                    result.matches.append(
                        RegexMatch(
                            seed=seed,
                            agent=agent_num,
                            message_index=idx,
                            role=msg["role"],
                            matched_term=m.group(0),
                            context_snippet=_snippet(
                                msg["content"], m.start(), m.end()
                            ),
                        )
                    )

    return result


def detect_all_regex(
    results_dir: str | Path,
    model: str,
    concepts: list[str],
    check_roles: Optional[set[str]] = None,
) -> list[RegexDetectionResult]:
    """Run regex detection across all concept/number directories.

    Parameters
    ----------
    results_dir : e.g. experiments/Llama-3.1-8B-Instruct/results
    model : model name
    concepts : list of concept names to scan
    check_roles : roles to check (default assistant only)

    Returns
    -------
    List of RegexDetectionResult, one per conversation file.
    """
    results_dir = Path(results_dir)
    all_results: list[RegexDetectionResult] = []

    for concept in concepts:
        concept_dir = results_dir / concept
        if not concept_dir.is_dir():
            continue
        for number_dir in sorted(concept_dir.iterdir()):
            conv_file = number_dir / "conversations.json"
            if not conv_file.exists():
                continue
            res = detect_bias_regex(
                conversation_path=conv_file,
                concept=concept,
                model=model,
                number=number_dir.name,
                check_roles=check_roles,
            )
            all_results.append(res)

    return all_results
