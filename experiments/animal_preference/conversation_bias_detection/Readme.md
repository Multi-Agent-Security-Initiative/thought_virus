## Detect if the subliminal bias was ever detected in any conversations.

For LLama-3.1-8B-Instruct/results, and Qwen2.5-7B-Instruct/results, we have the conversations in `$BIAS/$TOKEN/conversations.json`. We can check if the bias was ever detected in any of the conversations using both a simple regex search for the bias, and a judge model to check if the bias was detected in the conversations (for example if the bias is chimpanzee, the judge can also check if the bias included chimpanzee-related words like "monkey", "ape", "primate", etc.).

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration: model paths, related-word dictionaries, judge model settings |
| `regex_detector.py` | Regex-based detection — scans assistant messages for exact concept words and synonyms |
| `judge_detector.py` | LLM judge detection — uses vLLM to semantically evaluate bias leakage, auto-detects GPUs |
| `report.py` | Report generation — per-conversation JSON, summary CSV, aggregate JSON, console output |
| `run_detection.py` | Main entry point — orchestrates both detectors and generates reports |
| `run_detection.sh` | Shell wrapper with GPU info display |

## Quick start

```bash
# Full run (regex + LLM judge) — requires GPUs
./run_detection.sh

# Regex only (no GPU needed)
./run_detection.sh --regex-only

# Judge only
./run_detection.sh --judge-only

# Specific models / concepts
python run_detection.py --models Llama-3.1-8B-Instruct --concepts elephant dolphin

# Override judge model
python run_detection.py --judge-model Qwen/Qwen2.5-32B-Instruct

# Also scan user relay messages
python run_detection.py --check-user-messages

# Include the "random" control group
python run_detection.py --include-random
```

## Output

Reports are saved to `reports/`:

- `reports/per_conversation/` — one JSON per conversation file with regex matches + judge verdicts
- `reports/summary.csv` — one row per conversation file
- `reports/aggregate_summary.json` — rates grouped by model × concept