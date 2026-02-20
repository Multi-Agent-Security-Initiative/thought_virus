# Thought Virus

> Research project investigating [brief description of your research hypothesis]

## Overview

[Add a brief overview of what this project investigates and the key research questions]

## Setup

### Prerequisites

- Python >= 3.13
- CUDA-compatible GPU (for model inference)

### Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Project Structure

```
.
├── src/                    # Source code
│   └── run_analysis.py    # Main analysis script
├── experiments/           # Experimental results and plots
│   ├── animal-preference/ # Results for animal preference experiments
│   └── misalignment/      # Results for misalignment experiment
├── result_analysis/       # Analysis and plotting scripts
│   └── plot_logprob_bars.py
└── pyproject.toml        # Project dependencies
```

## Usage

### Running Experiments

```bash
# [Add command to run experiments]
python main.py
```

### Analyzing Results

```bash
# [Add commands for analysis]
python src/run_analysis.py
```

### Generating Plots

```bash
# [Add plotting commands]
python result_analysis/plot_logprob_bars.py
```

## Experiments

### Model Configurations

- **Qwen2.5-7B-Instruct**: [Add description of experiments]
- **TruthfulQA-binary**: [Add description of truthfulness experiments]

### Key Metrics

- Log probabilities (bidirectional and unidirectional)
- Subliminal frequencies
- Conversation concept counts

## Results

[Add summary of key findings]

### Visualizations

Results are stored in `experiments/<model>/plots/`:
- Bidirectional logprob bars
- Unidirectional logprob bars
- [Other visualizations]

## Development

### Running Tests

```bash
# [Add test commands if applicable]
```

### Code Structure

- Each run is organized in `runs/<number>/` with its own configuration
- Results are stored alongside the run configuration
- Analysis scripts process results across multiple runs

## Citation

```bibtex
[Add citation information if this becomes a paper]
```

## License

[Add license information]

## Contact

[Add contact information]
