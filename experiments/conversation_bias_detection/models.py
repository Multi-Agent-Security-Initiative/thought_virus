"""Shared data classes for bias detection results."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Regex detection data classes ──────────────────────────────────────

@dataclass
class RegexMatch:
    """A single regex match in a conversation."""
    seed: str
    agent: str
    message_index: int
    role: str
    matched_term: str
    context_snippet: str  # surrounding text (up to 200 chars)


@dataclass
class RegexDetectionResult:
    """Results for one conversation file."""
    model: str
    concept: str
    number: str
    file_path: str
    matches: list[RegexMatch] = field(default_factory=list)

    @property
    def is_biased(self) -> bool:
        return len(self.matches) > 0

    @property
    def biased_seeds(self) -> set[str]:
        return {m.seed for m in self.matches}

    @property
    def biased_agents(self) -> set[tuple[str, str]]:
        """Return set of (seed, agent) pairs that contain bias matches."""
        return {(m.seed, m.agent) for m in self.matches}

    def summary_dict(self) -> dict:
        return {
            "model": self.model,
            "concept": self.concept,
            "number": self.number,
            "file_path": self.file_path,
            "is_biased": self.is_biased,
            "num_matches": len(self.matches),
            "biased_seeds": sorted(self.biased_seeds),
            "matches": [
                {
                    "seed": m.seed,
                    "agent": m.agent,
                    "message_index": m.message_index,
                    "role": m.role,
                    "matched_term": m.matched_term,
                    "context_snippet": m.context_snippet,
                }
                for m in self.matches
            ],
        }
