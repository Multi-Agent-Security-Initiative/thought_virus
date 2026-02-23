# Thought Virus: Viral Misalignment via Subliminal Prompting in Multi-Agent System

📝 Read the paper on arXiv

This repository contains the code and experiments from the paper **Thought Virus: Viral Misalignment via Subliminal Prompting in Multi-Agent System**. Based on prior work by [Subliminal Learning](https://arxiv.org/abs/2507.14805), we investigate how hidden biases of LLMs transfer in multi-agent systems, and develop Thought Virus -- a novel attack vector that exploits subliminal prompting in multi-agent settings.

## Key Findings
- An agent compromised via subliminal prompting **spreads its induced bias though multi-agent networks** across all tested models and topologies.
- Bias strength **decreases with distance** from the originally compromised agent, **yet persists across up to 5 agent-to-agent hops** in our experiments.
- Thought Virus induces **viral misalignment**: subliminal prompting of a single agent degrades the truthfulness of downstream agents on TruthfulQA, even when those agents receive no adversarial input directly.

## Setup

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
├── src/                              # Source code
│   └── run_analysis.py               # Main analysis script
├── experiments/                      # Experimental results and plots
│   ├── animal-preference/            # Results for animal preference experiments
│   ├── misalignment/                 # Results for misalignment experiment
|   └── conversation_bias_detection   # Detects biases included in MAS conversations
├── result_analysis/                  # Analysis and plotting scripts
|   └── plot_frequency_bars.py        # Create barplots for response frequency results
│   └── plot_logprob_bars.py          # Create barplots for logprob results
└── pyproject.toml                    # Project dependencies
```

## Running Animal Preference Experiments

### Generating a Config File

*Note: Some models on Hugging Face (e.g., gated models like Llama) require you to log in and accept their terms of service before access is granted. Once approved, you must provide a Hugging Face API token in the `.env` file.*

To start a new run, define the relevant hyperparameters in experiment_config.py and place it in the corresponding folder.

### Generating Results

```bash
python src/run_analysis.py experiments/animal_preference/{EXPERIMENT_FOLDER}
```

### Generating Plots

```bash
python result_analysis/plot_frequency_bars.py experiments/animal_preference/{EXPERIMENT_FOLDER}
python result_analysis/plot_logprob_bars.py experiments/animal_preference/{EXPERIMENT_FOLDER}
```

### Detecting Overt Biases in Conversations

To check if the bias was communicated overtly in any of the agent-to-agent  — which would make the attack trivially detectable and stoppable — we run two complementary checks: a simple regex search and an LLM judge. Both can be run via `run_detection.sh` in `experiments/animal_preference/conversation_bias_detection/`.

## Citation

[TBA]

## Contact

For questions about the project or the code, feel free to reach out to [Moritz Weckbecker](mailto:moritz.weckbecker@hhi.fraunhofer.de) or [Michael Mulet](mailto:michael@multiagentsecurity.org).