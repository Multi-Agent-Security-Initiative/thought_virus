#!/usr/bin/env python3
"""Detect subliminal bias in multi-agent conversations.

Uses regex search to find exact concept words and related terms.

Usage
-----
  python run_detection.py

  # Specify models / concepts
  python run_detection.py --models Llama-3.1-8B-Instruct --concepts elephant dolphin

  # Also scan user messages (not just assistant)
  python run_detection.py --check-user-messages
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure this directory is on sys.path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CONCEPTS,
    EXPERIMENTS_DIR,
    MODELS,
    REPORT_DIR,
)
from regex_detector import detect_all_regex
from report import (
    print_summary,
    save_aggregate_summary,
    save_per_conversation_report,
    save_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect subliminal bias in multi-agent conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Model directories to scan (default: all configured models)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=CONCEPTS,
        help="Bias concepts to search for (default: all 10 animals)",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default=REPORT_DIR,
        help="Directory for output reports",
    )
    parser.add_argument(
        "--check-user-messages",
        action="store_true",
        help="Also scan user messages (not just assistant)",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help='Also scan the "random" control group',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    concepts = list(args.concepts)
    if args.include_random and "random" not in concepts:
        concepts.append("random")

    check_roles = {"assistant"}
    if args.check_user_messages:
        check_roles.add("user")

    all_regex_results = []

    # ── Regex detection ────────────────────────────────────────────────
    print("\nRunning regex detection...")
    for model in args.models:
        results_dir = Path(EXPERIMENTS_DIR) / model / "results"
        if not results_dir.is_dir():
            print(f"  WARNING: {results_dir} not found, skipping.")
            continue
        print(f"  Scanning {model}...")
        rr = detect_all_regex(
            results_dir=results_dir,
            model=model,
            concepts=concepts,
            check_roles=check_roles,
        )
        all_regex_results.extend(rr)
        biased = sum(1 for r in rr if r.is_biased)
        print(f"    Found {biased}/{len(rr)} biased conversation files (regex)")

    # ── Reports ────────────────────────────────────────────────────────
    report_dir = Path(args.report_dir)
    per_conv_dir = report_dir / "per_conversation"

    # Per-conversation JSON reports
    if all_regex_results:
        print("\nSaving per-conversation reports...")
        for rr in all_regex_results:
            save_per_conversation_report(rr, per_conv_dir)

    # Summary CSV
    if all_regex_results:
        csv_path = save_summary_csv(
            all_regex_results,
            report_dir / "summary.csv",
        )
        print(f"Summary CSV: {csv_path}")

        agg_path = save_aggregate_summary(
            all_regex_results,
            report_dir / "aggregate_summary.json",
        )
        print(f"Aggregate summary: {agg_path}")

    # Console summary
    print_summary(all_regex_results)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
