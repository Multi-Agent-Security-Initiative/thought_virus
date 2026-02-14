#!/usr/bin/env python3
"""
Quick script to review all per_conversation JSON reports.
Lists every match found, shows per-animal statistics, and overall statistics.

Usage:
    python review_matches.py                          # default: reports/per_conversation
    python review_matches.py claude-code-results      # use claude-code-results dir
    python review_matches.py /absolute/path/to/dir    # any directory with per_conversation/ inside
"""

import argparse
import io
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_REPORTS_DIR = SCRIPT_DIR / "reports"


def resolve_dirs(reports_arg: str | None):
    """Return (per_conversation_dir, output_dir) from the CLI argument."""
    if reports_arg is None:
        base = DEFAULT_REPORTS_DIR
    else:
        base = Path(reports_arg)
        if not base.is_absolute():
            base = SCRIPT_DIR / base

    per_conv = base / "per_conversation"
    if not per_conv.is_dir():
        # maybe the user pointed directly at the per_conversation folder
        if base.is_dir() and any(base.glob("*.json")):
            per_conv = base
            base = base.parent
        else:
            raise SystemExit(f"No per_conversation directory found in {base}")

    return per_conv, base


def load_all_reports(per_conv_dir: Path):
    reports = []
    for f in sorted(per_conv_dir.glob("*.json")):
        with open(f) as fh:
            reports.append(json.load(fh))
    return reports


def get_detection_block(report: dict):
    """Return the detection sub-dict regardless of whether it's 'regex' or 'claude_code'."""
    for key in ("regex", "claude_code"):
        if key in report:
            return key, report[key]
    return None, {}


def main():
    parser = argparse.ArgumentParser(description="Review per-conversation bias detection reports.")
    parser.add_argument(
        "reports_dir",
        nargs="?",
        default=None,
        help="Path to reports directory (must contain per_conversation/). "
             "Defaults to reports/. Can also be a relative path like 'claude-code-results'.",
    )
    args = parser.parse_args()

    per_conv_dir, output_dir = resolve_dirs(args.reports_dir)
    reports = load_all_reports(per_conv_dir)
    if not reports:
        print("No reports found.")
        return

    print(f"Reading from: {per_conv_dir}")
    print(f"Output to:    {output_dir}")
    print()

    # Tee output to both stdout and a buffer for saving
    buf = io.StringIO()

    def out(line=""):
        print(line)
        buf.write(line + "\n")

    # Collect all matches and stats
    all_matches = []
    detection_source = None
    # Stats: per (model, animal) -> counts
    stats = defaultdict(lambda: {"total_conversations": 0, "biased_conversations": 0, "total_matches": 0})

    for r in reports:
        model = r["model"]
        animal = r["concept"]
        number = r["number"]
        key = (model, animal)
        stats[key]["total_conversations"] += 1

        source, detection = get_detection_block(r)
        if source and detection_source is None:
            detection_source = source

        matches = detection.get("matches", [])
        num_matches = detection.get("num_matches", 0)

        if num_matches > 0:
            stats[key]["biased_conversations"] += 1
            stats[key]["total_matches"] += num_matches

        for m in matches:
            entry = {
                "model": model,
                "animal": animal,
                "number": number,
                "seed": m.get("seed"),
                "agent": m.get("agent"),
                "message_index": m.get("message_index"),
                "role": m.get("role", ""),
                "matched_term": m.get("matched_term"),
                "context_snippet": m.get("context_snippet", ""),
            }
            # Include claude-code extra fields if present
            if "semantic_judgment" in m:
                entry["semantic_judgment"] = m["semantic_judgment"]
            if "reasoning" in m:
                entry["reasoning"] = m["reasoning"]
            all_matches.append(entry)

    out(f"  Detection source: {detection_source or 'unknown'}")
    out()

    # ── Print all matches ──────────────────────────────────────────────
    out("=" * 100)
    out(f"  ALL MATCHES  ({len(all_matches)} total)")
    out("=" * 100)
    for i, m in enumerate(all_matches, 1):
        out(f"\n  Match #{i}")
        out(f"    Model:         {m['model']}")
        out(f"    Animal:        {m['animal']}")
        out(f"    Number:        {m['number']}")
        out(f"    Seed:          {m['seed']}  |  Agent: {m['agent']}  |  Msg idx: {m['message_index']}  |  Role: {m['role']}")
        out(f"    Matched term:  {m['matched_term']}")
        if "semantic_judgment" in m:
            out(f"    Judgment:      {m['semantic_judgment']}")
        if "reasoning" in m:
            reasoning = m["reasoning"].replace("\n", " ").strip()
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            out(f"    Reasoning:     {reasoning}")
        snippet = m["context_snippet"].replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        out(f"    Context:       {snippet}")
    out()

    # ── Per-animal statistics (grouped by model) ───────────────────────
    models = sorted(set(k[0] for k in stats))
    animals = sorted(set(k[1] for k in stats))

    for model in models:
        out("=" * 100)
        out(f"  PER-ANIMAL STATISTICS — {model}")
        out("=" * 100)
        out(f"  {'Animal':<15} {'Conversations':>15} {'Biased':>10} {'Bias Rate':>12} {'Total Matches':>16}")
        out(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*12} {'-'*16}")

        model_total = 0
        model_biased = 0
        model_matches = 0

        for animal in animals:
            key = (model, animal)
            if key not in stats:
                continue
            s = stats[key]
            total = s["total_conversations"]
            biased = s["biased_conversations"]
            matches = s["total_matches"]
            rate = (biased / total * 100) if total > 0 else 0.0

            model_total += total
            model_biased += biased
            model_matches += matches

            out(f"  {animal:<15} {total:>15} {biased:>10} {rate:>11.1f}% {matches:>16}")

        model_rate = (model_biased / model_total * 100) if model_total > 0 else 0.0
        out(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*12} {'-'*16}")
        out(f"  {'TOTAL':<15} {model_total:>15} {model_biased:>10} {model_rate:>11.1f}% {model_matches:>16}")
        out()

    # ── Overall statistics ─────────────────────────────────────────────
    grand_total = sum(s["total_conversations"] for s in stats.values())
    grand_biased = sum(s["biased_conversations"] for s in stats.values())
    grand_matches = sum(s["total_matches"] for s in stats.values())
    grand_rate = (grand_biased / grand_total * 100) if grand_total > 0 else 0.0

    out("=" * 100)
    out("  OVERALL STATISTICS")
    out("=" * 100)
    out(f"  Total conversations:   {grand_total}")
    out(f"  Biased conversations:  {grand_biased}")
    out(f"  Bias rate:             {grand_rate:.1f}%")
    out(f"  Total matches:         {grand_matches}")
    out(f"  Models:                {', '.join(models)}")
    out(f"  Animals:               {', '.join(animals)}")
    out()

    # ── Matched terms breakdown ────────────────────────────────────────
    term_counts = defaultdict(int)
    for m in all_matches:
        term_counts[m["matched_term"]] += 1

    if term_counts:
        out("=" * 100)
        out("  MATCHED TERMS BREAKDOWN")
        out("=" * 100)
        for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
            out(f"    {term:<30} {count:>5}x")
        out()

    # ── Save results ───────────────────────────────────────────────────
    # 1. Save plain-text report
    txt_path = output_dir / "match_review.txt"
    txt_path.write_text(buf.getvalue())
    print(f"Saved text report to {txt_path}")

    # 2. Save structured JSON
    per_model_stats = {}
    for model in models:
        animal_stats = {}
        for animal in animals:
            key = (model, animal)
            if key not in stats:
                continue
            s = stats[key]
            total = s["total_conversations"]
            biased = s["biased_conversations"]
            animal_stats[animal] = {
                "total_conversations": total,
                "biased_conversations": biased,
                "bias_rate": round(biased / total * 100, 1) if total > 0 else 0.0,
                "total_matches": s["total_matches"],
            }
        model_total = sum(a["total_conversations"] for a in animal_stats.values())
        model_biased = sum(a["biased_conversations"] for a in animal_stats.values())
        model_matches = sum(a["total_matches"] for a in animal_stats.values())
        per_model_stats[model] = {
            "per_animal": animal_stats,
            "total_conversations": model_total,
            "biased_conversations": model_biased,
            "bias_rate": round(model_biased / model_total * 100, 1) if model_total > 0 else 0.0,
            "total_matches": model_matches,
        }

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "total_conversations": grand_total,
            "biased_conversations": grand_biased,
            "bias_rate": round(grand_rate, 1),
            "total_matches": grand_matches,
            "models": models,
            "animals": animals,
        },
        "per_model": per_model_stats,
        "matched_terms": dict(sorted(term_counts.items(), key=lambda x: -x[1])),
        "all_matches": all_matches,
    }

    json_path = output_dir / "match_review.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"Saved JSON report to {json_path}")


if __name__ == "__main__":
    main()
