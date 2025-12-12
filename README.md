# Agreement-Bias

Tools and experiments for measuring and analyzing **agreement bias** in large language models (LLMs).

---

## Setup

- Generate the **objective questions** dataset:
  - `truthful_prompts.py` converts the TruthfulQA dataset into neutral and framed prompts.
- Generate the **subjective questions** dataset:
  - `convert_scenario_csv.py` converts `TableS1.csv` (Moral Machine) into neutral, positively framed, and negatively framed prompts.

---

## Experiment

- Generate model responses:
  - `generate_responses.py` — prompts LLMs with the objective dataset.
  - `generate_moral_responses.py` — prompts LLMs with the subjective dataset.
- All model outputs are saved as JSON files for downstream processing.

- Evaluate responses:
  - `evaluate_moral_results.py` — parses subjective JSON outputs and produces `results.csv` with parsed responses and evaluation metadata.

---

## Results

- Aggregate and summarize runs:
  - `summarize_moral_results.py` — combines results across runs and produces `summary_report.csv`.
- The generated summary files are used to create the charts, tables, and quantitative analyses presented in the paper.

---

## Files & Outputs (examples)

- `data/` — raw and processed prompt files  
- `outputs/` — JSON responses from model runs  
- `results.csv` — parsed evaluation results (per-run)  
- `summary_report.csv` — aggregated metrics used for figures/tables

---

## Usage (example commands) (this will not work if you just copy-paste)

```bash
# 1. Build datasets
python truthful_prompts.py --out data/objective_prompts.json
python convert_scenario_csv.py TableS1.csv --out data/subjective_prompts.json

# 2. Generate model outputs
python generate_responses.py --prompts data/objective_prompts.json --out outputs/objective_runs.json
python generate_moral_responses.py --prompts data/subjective_prompts.json --out outputs/moral_runs.json

# 3. Evaluate and summarize
python evaluate_moral_results.py --inputs outputs/moral_runs.json --out results.csv
python summarize_moral_results.py --inputs results.csv --out summary_report.csv
