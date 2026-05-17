# LLM-Assisted Sketch-Based Requirements Elicitation and Prototyping

This repository contains the reproducibility package for the study:

**An Empirical Evaluation of LLM-Assisted Sketch-Based Requirements Elicitation and Prototyping**

The package supports replication of experiments evaluating Large Language Models (LLMs) in transforming low-fidelity hand-drawn sketches into:

1. Natural-language software requirements
2. Interactive HTML/CSS/JavaScript prototypes

## Evaluated Models

- GPT-4.1
- GPT-4o
- Claude Sonnet 4.5
- Gemini 2.5 Pro

## Experimental Scenarios

- E-commerce Checkout System
- Healthcare Appointment System
- Service Portal System

## Repository Structure

```text
LLM-RE-Sketch2Prototype-Experiments/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── experiments/
├── experiments_output/
├── reviewer_materials/
├── prompts/
├── screenshots/
└── docs/
```

## Reproducing the Experiments

```bash
git clone https://github.com/hamdan-2006/LLM-RE-Sketch2Prototype-Experiments.git
cd LLM-RE-Sketch2Prototype-Experiments
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file or set environment variables:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

## Running Experiments

### GPT-4.1

```bash
python experiments/experiment_GPT4.1/run_gpt4_experiment.py
```

### GPT-4o

```bash
python experiments/experiment_GPT4o/experiment_GPT4o.py
```

### Claude Sonnet 4.5

```bash
python experiments/experiment_claude-sonnet-4-5/claude-sonnet-4-5.py
```

### Gemini 2.5 Pro

```bash
python experiments/Gemini_2.5_Pro/Gemini_2.5_Pro.py
```

## Output Location

All generated outputs should be saved under:

```text
experiments_output/
```

Each model folder should contain scenario-specific outputs, including requirements files, raw prototype files, rendered HTML/CSS/JavaScript prototypes, screenshots, logs, and metadata.

## Reviewer Materials

The reviewer workbook and scoring instructions should be placed in:

```text
reviewer_materials/
```

Recommended file:

```text
LLM_RE_Reviewer_Evaluation_Workbook_Enhanced_v2_gold.xlsx
```

## Prompting Strategy

The experiment uses a two-phase prompting strategy:

1. **Phase 1:** Generate natural-language requirements from sketches.
2. **Phase 2:** Generate a self-contained HTML/CSS/JavaScript prototype using the sketches and Phase 1 requirements.

Prompt templates are provided in the `prompts/` folder.

## Experimental Configuration

| Parameter | GPT-4.1 | GPT-4o | Claude Sonnet 4.5 | Gemini 2.5 Pro |
|-----------|--------:|-------:|------------------:|---------------:|
| Runs per scenario | 3 | 3 | 3 | 3 |
| Temperature | 0.2 | 0.2 | 0.2 | 0.2 |
| top_p | 1.0 | 1.0 | "not supported / provider default" | 1.0|
| Output format | TXT + HTML/CSS/JavaScrip | TXT + HTML/CSS/JavaScrip | TXT + HTML/CSS/JavaScrip | TXT + HTML/CSS/JavaScrip |

## Reproducibility Notes

To support transparency and reproducibility, this repository includes:

- Original low-fidelity sketch inputs
- Model-specific experiment scripts
- Prompt templates
- Raw generated requirements
- Raw generated prototype outputs
- Rendered HTML prototypes
- Screenshots of generated prototypes
- Reviewer evaluation workbook
- Experiment logs and manifest files


## Citation

Please cite the associated paper if using this repository.
