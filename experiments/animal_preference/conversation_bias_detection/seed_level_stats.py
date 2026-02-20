#!/usr/bin/env python3
"""
Seed-level leakage analysis.

For every (model, animal, number), shows how many of the 20 seeds are
clean vs leaked. Produces a Discord-pasteable report.

Usage:
    python seed_level_stats.py                      # default: reports/
    python seed_level_stats.py claude-code-results   # claude-code results
"""

import argparse
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_REPORTS_DIR = SCRIPT_DIR / "reports"
SEEDS_PER_NUMBER = 20


def resolve_dirs(reports_arg: str | None):
    if reports_arg is None:
        base = DEFAULT_REPORTS_DIR
    else:
        base = Path(reports_arg)
        if not base.is_absolute():
            base = SCRIPT_DIR / base

    per_conv = base / "per_conversation"
    if not per_conv.is_dir():
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
    for key in ("regex", "claude_code"):
        if key in report:
            return key, report[key]
    return None, {}


def main():
    parser = argparse.ArgumentParser(description="Seed-level leakage stats.")
    parser.add_argument("reports_dir", nargs="?", default=None,
                        help="Path to reports directory (default: reports/).")
    args = parser.parse_args()

    per_conv_dir, output_dir = resolve_dirs(args.reports_dir)
    reports = load_all_reports(per_conv_dir)
    if not reports:
        print("No reports found.")
        return

    print(f"Reading from: {per_conv_dir}")
    print(f"Output to:    {output_dir}")
    print()

    buf = io.StringIO()

    def out(line=""):
        print(line)
        buf.write(line + "\n")

    # ── Gather data ────────────────────────────────────────────────────
    # Key: (model, animal, number) -> set of leaked seeds
    detection_source = None
    leaked_seeds: dict[tuple, set] = {}
    # Also collect match details per leaked seed
    seed_matches: dict[tuple, list] = defaultdict(list)  # (model,animal,number,seed) -> [match_info]

    for r in reports:
        model = r["model"]
        animal = r["concept"]
        number = r["number"]
        source, detection = get_detection_block(r)
        if source and detection_source is None:
            detection_source = source

        biased = set(detection.get("biased_seeds", []))
        leaked_seeds[(model, animal, number)] = biased

        for m in detection.get("matches", []):
            seed_matches[(model, animal, number, m.get("seed"))].append({
                "agent": m.get("agent"),
                "matched_term": m.get("matched_term"),
                "semantic_judgment": m.get("semantic_judgment", ""),
            })

    out(f"**Detection source:** `{detection_source or 'unknown'}`")
    out()

    # ── Group by model ─────────────────────────────────────────────────
    models = sorted(set(k[0] for k in leaked_seeds))
    animals = sorted(set(k[1] for k in leaked_seeds))

    # Grand totals
    grand_total_seeds = 0
    grand_leaked_seeds = 0
    grand_clean_seeds = 0

    for model in models:
        out(f"## {model}")
        out()

        # ── Per-number detail table ────────────────────────────────────
        out("### Per-Number Breakdown")
        out()
        out("```")
        out(f"{'Animal':<14} {'Number':<8} {'Seeds':>5} {'Leaked':>7} {'Clean':>6} {'Clean%':>7}  Leaked Seeds")
        out(f"{'─'*14} {'─'*8} {'─'*5} {'─'*7} {'─'*6} {'─'*7}  {'─'*20}")

        model_total = 0
        model_leaked = 0

        for animal in animals:
            numbers = sorted(
                [k[2] for k in leaked_seeds if k[0] == model and k[1] == animal],
                key=lambda x: int(x),
            )
            for number in numbers:
                key = (model, animal, number)
                leaked = leaked_seeds[key]
                total = SEEDS_PER_NUMBER
                n_leaked = len(leaked)
                n_clean = total - n_leaked
                pct = n_clean / total * 100

                model_total += total
                model_leaked += n_leaked

                leaked_str = ", ".join(sorted(leaked, key=int)) if leaked else "—"
                out(f"{animal:<14} {number:<8} {total:>5} {n_leaked:>7} {n_clean:>6} {pct:>6.0f}%  {leaked_str}")

        model_clean = model_total - model_leaked
        model_pct = model_clean / model_total * 100 if model_total > 0 else 0
        out(f"{'─'*14} {'─'*8} {'─'*5} {'─'*7} {'─'*6} {'─'*7}  {'─'*20}")
        out(f"{'TOTAL':<14} {'':8} {model_total:>5} {model_leaked:>7} {model_clean:>6} {model_pct:>6.0f}%")
        out("```")
        out()

        grand_total_seeds += model_total
        grand_leaked_seeds += model_leaked
        grand_clean_seeds += model_clean

        # ── Per-animal summary ─────────────────────────────────────────
        out("### Per-Animal Summary")
        out()
        out("```")
        out(f"{'Animal':<14} {'Total Seeds':>12} {'Leaked':>7} {'Clean':>6} {'Clean%':>7} {'Nums w/ Leak':>13}")
        out(f"{'─'*14} {'─'*12} {'─'*7} {'─'*6} {'─'*7} {'─'*13}")

        for animal in animals:
            numbers = [k[2] for k in leaked_seeds if k[0] == model and k[1] == animal]
            total = len(numbers) * SEEDS_PER_NUMBER
            n_leaked = sum(len(leaked_seeds[(model, animal, n)]) for n in numbers)
            n_clean = total - n_leaked
            pct = n_clean / total * 100 if total > 0 else 0
            nums_with_leak = sum(1 for n in numbers if len(leaked_seeds[(model, animal, n)]) > 0)
            out(f"{animal:<14} {total:>12} {n_leaked:>7} {n_clean:>6} {pct:>6.0f}% {nums_with_leak:>8}/{len(numbers)}")

        out(f"{'─'*14} {'─'*12} {'─'*7} {'─'*6} {'─'*7} {'─'*13}")
        out(f"{'TOTAL':<14} {model_total:>12} {model_leaked:>7} {model_clean:>6} {model_pct:>6.0f}%")
        out("```")
        out()

        # ── Leaked seed details ────────────────────────────────────────
        leaked_entries = []
        for animal in animals:
            numbers = sorted(
                [k[2] for k in leaked_seeds if k[0] == model and k[1] == animal],
                key=lambda x: int(x),
            )
            for number in numbers:
                for seed in sorted(leaked_seeds[(model, animal, number)], key=int):
                    matches = seed_matches[(model, animal, number, seed)]
                    terms = ", ".join(set(m["matched_term"] for m in matches))
                    judgments = ", ".join(set(m["semantic_judgment"] for m in matches if m["semantic_judgment"]))
                    leaked_entries.append((animal, number, seed, terms, judgments))

        if leaked_entries:
            out("### Leaked Seed Details")
            out()
            out("```")
            out(f"{'Animal':<14} {'Number':<8} {'Seed':<6} {'Matched Terms':<25} {'Judgment' if any(e[4] for e in leaked_entries) else ''}")
            out(f"{'─'*14} {'─'*8} {'─'*6} {'─'*25} {'─'*20 if any(e[4] for e in leaked_entries) else ''}")
            for animal, number, seed, terms, judgments in leaked_entries:
                line = f"{animal:<14} {number:<8} {seed:<6} {terms:<25}"
                if judgments:
                    line += f" {judgments}"
                out(line)
            out("```")
            out()

    # ── Grand overall ──────────────────────────────────────────────────
    grand_pct = grand_clean_seeds / grand_total_seeds * 100 if grand_total_seeds > 0 else 0

    out("## Overall")
    out()
    out("```")
    out(f"Total seeds:     {grand_total_seeds}")
    out(f"Leaked seeds:    {grand_leaked_seeds}")
    out(f"Clean seeds:     {grand_clean_seeds}")
    out(f"Clean rate:      {grand_pct:.1f}%")
    out(f"Leak rate:       {100 - grand_pct:.1f}%")
    out("```")
    out()

    # ── Save ───────────────────────────────────────────────────────────
    txt_path = output_dir / "seed_level_stats.txt"
    txt_path.write_text(buf.getvalue())
    print(f"Saved text report to {txt_path}")

    # Build JSON
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "detection_source": detection_source,
        "seeds_per_number": SEEDS_PER_NUMBER,
        "overall": {
            "total_seeds": grand_total_seeds,
            "leaked_seeds": grand_leaked_seeds,
            "clean_seeds": grand_clean_seeds,
            "clean_rate_pct": round(grand_pct, 1),
        },
        "per_model": {},
    }

    for model in models:
        model_data = {"per_animal": {}}
        for animal in animals:
            numbers = sorted(
                [k[2] for k in leaked_seeds if k[0] == model and k[1] == animal],
                key=lambda x: int(x),
            )
            per_number = {}
            for number in numbers:
                leaked = leaked_seeds[(model, animal, number)]
                per_number[number] = {
                    "total_seeds": SEEDS_PER_NUMBER,
                    "leaked_seeds": sorted(leaked, key=int) if leaked else [],
                    "num_leaked": len(leaked),
                    "num_clean": SEEDS_PER_NUMBER - len(leaked),
                }
            total = len(numbers) * SEEDS_PER_NUMBER
            n_leaked = sum(len(leaked_seeds[(model, animal, n)]) for n in numbers)
            model_data["per_animal"][animal] = {
                "per_number": per_number,
                "total_seeds": total,
                "leaked_seeds": n_leaked,
                "clean_seeds": total - n_leaked,
                "clean_rate_pct": round((total - n_leaked) / total * 100, 1) if total > 0 else 0,
            }

        m_total = sum(a["total_seeds"] for a in model_data["per_animal"].values())
        m_leaked = sum(a["leaked_seeds"] for a in model_data["per_animal"].values())
        model_data["total_seeds"] = m_total
        model_data["leaked_seeds"] = m_leaked
        model_data["clean_seeds"] = m_total - m_leaked
        model_data["clean_rate_pct"] = round((m_total - m_leaked) / m_total * 100, 1) if m_total > 0 else 0
        json_data["per_model"][model] = model_data

    json_path = output_dir / "seed_level_stats.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"Saved JSON report to {json_path}")


if __name__ == "__main__":
    main()
