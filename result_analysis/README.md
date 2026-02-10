# Result Analysis Scripts

This folder contains scripts for analyzing experimental results from thought virus experiments.

## Scripts

### plot_logprob_bars.py

Creates bar charts comparing log probabilities across different conditions for subliminal concept analysis.

**Usage:**
```bash
python result_analysis/plot_logprob_bars.py <experiment_folder>
```

**Example:**
```bash
python result_analysis/plot_logprob_bars.py experiments/Qwen2.5-7B-Instruct
```

**What it does:**
1. Reads configuration from `<experiment_folder>/experiment_config.py`
2. Gets the list of concepts to analyze from `CONCEPTS`
3. Processes log probability data from the results folder
4. Creates bar charts with bootstrap confidence intervals
5. Saves plots to `<experiment_folder>/plots/logprob_bars/`

**Requirements:**
- The experiment folder must contain:
  - `experiment_config.py` with `CONCEPTS`, `NUMBER_OF_AGENTS`, and `NUM_SAMPLES` defined
  - `results/` directory with subliminal and random token data
  - `base_logprobs.csv` (fallback) or system frequency data in results

**Output:**
- PNG files for each concept in `<experiment_folder>/plots/logprob_bars/`
- Each plot shows 4 conditions:
  - Base rate (model baseline)
  - Random token average (control)
  - Subliminal token average
  - Subliminal token strongest (best run per agent)

## Directory Structure Expected

```
experiments/
└── Qwen2.5-7B-Instruct/
    ├── experiment_config.py
    ├── base_logprobs.csv (optional)
    ├── results/
    │   ├── lion/
    │   │   ├── 0/
    │   │   │   └── 200_samples/
    │   │   │       └── subliminal_logits.csv
    │   │   ├── 1/
    │   │   └── ...
    │   ├── elephant/
    │   ├── random/
    │   │   ├── 0/
    │   │   ├── 1/
    │   │   └── ...
    │   └── ...
    └── plots/  (created automatically)
        └── logprob_bars/
            ├── lion_logprob_bars.png
            ├── elephant_logprob_bars.png
            └── ...
```
