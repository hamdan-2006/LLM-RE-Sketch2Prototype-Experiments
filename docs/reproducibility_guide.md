# Reproducibility Guide

## 1. Environment Setup

Recommended environment:

- Python 3.10 or higher
- Windows 10/11, macOS, or Linux
- Modern browser for rendering HTML/CSS/JavaScript prototypes

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Playwright browsers if automated screenshots are used:

```bash
playwright install chromium
```

## 2. API Keys

Set the following environment variables before running experiments:

```bash
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key
```

Do not upload API keys to GitHub.

## 3. Input Artifacts

Each scenario folder should include a `sketches/` folder containing the low-fidelity sketch images used as model inputs.

Recommended scenario structure:

```text
scenario_name/
├── sketches/
├── prompts/
└── scenario_description.txt
```

## 4. Execution

Run the script for each model. Each script should process the three scenarios and produce three repeated runs per scenario.

## 5. Expected Outputs

Each run should produce:

- Requirements text file
- Raw prototype text file
- Rendered HTML/CSS/JavaScript files

## 6. Output Verification

Check that:

- Requirements are complete and readable
- HTML/CSS/JavaScript files open locally in a browser
- Screenshot files correspond to generated prototypes
- `experiment_manifest.csv` contains all runs
- `experiment_log.txt` records execution status

## 7. Known Limitations

- API models may change over time.
- Outputs may vary even when temperature is set to 0.
- API rate limits and temporary service unavailability may affect reproducibility.
- Multimodal sketch interpretation can vary by image quality.

