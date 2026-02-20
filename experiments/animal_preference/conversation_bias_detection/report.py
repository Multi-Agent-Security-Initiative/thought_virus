"""Report generation for bias detection results.

Produces per-conversation JSON reports and summary CSVs.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from models import RegexDetectionResult


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Per-conversation reports ───────────────────────────────────────────

def save_per_conversation_report(
    regex_result: RegexDetectionResult,
    report_dir: str | Path,
) -> Path:
    """Save a detailed JSON report for one conversation file."""
    report_dir = _ensure_dir(report_dir)

    report = {
        "model": regex_result.model,
        "concept": regex_result.concept,
        "number": regex_result.number,
        "file_path": regex_result.file_path,
        "timestamp": datetime.now().isoformat(),
        "regex": regex_result.summary_dict(),
        "is_biased": regex_result.is_biased,
    }

    filename = f"{regex_result.model}__{regex_result.concept}__{regex_result.number}.json"
    filepath = report_dir / filename
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    return filepath


# ── Summary CSV ────────────────────────────────────────────────────────

def save_summary_csv(
    regex_results: list[RegexDetectionResult],
    output_path: str | Path,
) -> Path:
    """Save a summary CSV with one row per conversation file.

    Columns: model, concept, number, regex_biased, regex_match_count, regex_biased_seeds
    """
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    rows = []
    for rr in regex_results:
        row = {
            "model": rr.model,
            "concept": rr.concept,
            "number": rr.number,
            "regex_biased": rr.is_biased,
            "regex_match_count": len(rr.matches),
            "regex_biased_seeds": ";".join(sorted(rr.biased_seeds)),
        }
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ── Aggregate summary ─────────────────────────────────────────────────

def save_aggregate_summary(
    regex_results: list[RegexDetectionResult],
    output_path: str | Path,
) -> Path:
    """Save a high-level aggregate JSON summary.

    Groups results by model and concept, showing counts and rates.
    """
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    # Group by (model, concept)
    grouped: dict[tuple[str, str], dict] = {}
    for rr in regex_results:
        key = (rr.model, rr.concept)
        if key not in grouped:
            grouped[key] = {
                "model": rr.model,
                "concept": rr.concept,
                "total_files": 0,
                "regex_biased_files": 0,
                "total_regex_matches": 0,
            }
        g = grouped[key]
        g["total_files"] += 1
        g["total_regex_matches"] += len(rr.matches)
        if rr.is_biased:
            g["regex_biased_files"] += 1

    # Compute rates
    for g in grouped.values():
        total = g["total_files"]
        g["regex_bias_rate"] = round(g["regex_biased_files"] / total, 4) if total else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_conversation_files": len(regex_results),
        "by_model_concept": list(grouped.values()),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


# ── Console report ─────────────────────────────────────────────────────

def print_summary(
    regex_results: list[RegexDetectionResult],
) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 80)
    print("BIAS DETECTION SUMMARY")
    print("=" * 80)

    # Group by model
    by_model: dict[str, list[RegexDetectionResult]] = {}
    for rr in regex_results:
        by_model.setdefault(rr.model, []).append(rr)

    for model_name, rr_list in sorted(by_model.items()):
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name}")
        print(f"{'─' * 80}")

        # Group by concept
        by_concept: dict[str, list[RegexDetectionResult]] = {}
        for rr in rr_list:
            by_concept.setdefault(rr.concept, []).append(rr)

        for concept, concept_results in sorted(by_concept.items()):
            regex_biased = sum(1 for r in concept_results if r.is_biased)
            total = len(concept_results)

            print(f"\n  Concept: {concept:15s} | "
                  f"Regex: {regex_biased}/{total} biased")

            # Show details for biased conversations
            for r in concept_results:
                if r.is_biased:
                    print(f"    Number {r.number:>4s}: regex({len(r.matches)} matches)")

    print("\n" + "=" * 80)
