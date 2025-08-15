# GoodFire AutoSteer — Evaluation Suite

This repository contains code and data to evaluate GoodFire's AutoSteer feature and several baseline steering techniques. The code exercises different steering methods, runs model queries, and uses an LLM-based rater to score responses on behavior and coherence.

## Current status

- The main entry point is `main.py` (interactive runner that orchestrates evaluations).
- Core evaluation logic lives in the `goodfire_eval/` package:
   - `goodfire_eval/steering_evaluator.py` — runs model calls and the rater pipeline.
   - `goodfire_eval/steering_methods.py` — implementations of steering methods (Control, Prompting, AutoSteer, Agentic search, Combined, Scaled AutoSteer).
   - `goodfire_eval/steering_dataset.py` — dataset loader for `datasets/common_prompts.json` and `datasets/steering_queries.json`.
   - `goodfire_eval/metric_rater.py` — Behavior and Coherence raters used to evaluate responses.
- Datasets are in the `datasets/` folder (JSON files with prompts and queries).
- Results are saved as CSVs in the `results/` folder.

## Project structure

Top-level files and folders you will commonly use:

```
.
├── main.py                        # Interactive runner to run evaluations
├── requirements.txt               # Python dependencies
├── datasets/                      # JSON datasets: common prompts and steering queries
├── goodfire_eval/                 # Core evaluation package
│   ├── steering_evaluator.py
│   ├── steering_methods.py
│   ├── steering_dataset.py
│   └── metric_rater.py
└── results/                       # CSV output from runs and plots
```

## Prerequisites

- Python 3.10+ (the code uses modern typing and asyncio features). If you must use 3.8/3.9, some small edits may be required.
- Install dependencies from `requirements.txt`.

## Installation and quick run

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set required environment variables (export in shell or put them in a `.env` file):

```bash
export OPEN_AI_API_KEY=your_openai_key_here
export GOODFIRE_API_KEY=your_goodfire_key_here
```

Note: `main.py` expects environment variables named `OPEN_AI_API_KEY` and `GOODFIRE_API_KEY`.

4. Run the interactive evaluator:

```bash
python main.py
```

When you run `main.py` it will:
- Instantiate clients for GoodFire and OpenAI.
- Ask you to select a variant model to evaluate.
- Iterate over steering methods and queries, run model calls, and score responses.
- Save results CSV files in `results/` with names like:
   `results/eval_<evaluator>_var_<variant>_dt_<YYYYmmdd_HHMM>.csv`

## How evaluation works (high level)

- Steering methods: implemented in `goodfire_eval/steering_methods.py`. The runner applies the steering method to a `goodfire.Variant` and then queries the model for a set of test prompts.
- The rater: `goodfire_eval/metric_rater.py` defines raters (BehaviorRater, CoherenceRater) that build XML-style prompts the evaluator model (`gpt-4o-mini` by default in `main.py`) answers with scores and a short analysis. `steering_evaluator.py` parses those XML responses into numeric scores.
- Data: `goodfire_eval/steering_dataset.py` assembles `SteeringQuery` objects from the JSON files in `datasets/` and creates the prompt message pairs that are sent to the model.

## Extending the project

- Add a new steering method: implement a subclass of `SteeringMethod` in `goodfire_eval/steering_methods.py` and add an instance to the `STEERING_METHODS` list in `main.py`.
- Add a new metric rater: implement `MetricRater` in `goodfire_eval/metric_rater.py` (must return `(prompt_str, tag_str)` via `__call__`). Add the rater instance to `RATER_METRICS` in `main.py`.
- Add new steering queries or common prompts: edit `datasets/steering_queries.json` and `datasets/common_prompts.json`.

## Notes and caveats

- The code uses asynchronous clients (`goodfire.AsyncClient`, `openai.AsyncOpenAI`) and runs some pieces concurrently. Ensure your environment and installed client versions match the APIs used in `main.py` and `goodfire_eval/`.
- `main.py` is interactive: it lists available steering variant models and expects a numeric selection.
- Evaluation responses are parsed from XML-like text the rater model returns. If the rater's output structure changes, update `steering_evaluator.parse_evaluation_response` accordingly.

## Research Findings

Detailed research findings and methodology can be found in the following [blog post](https://www.alignmentforum.org/posts/6dpKhtniqR3rnstnL/mind-the-coherence-gap-lessons-from-steering-llama-with-1?utm_campaign=post_share&utm_source=link).

## Contact

If anything here is unclear or you want additional README sections (examples, CI, richer dev docs), feel free to email me at eitusprejer@gmail.com. Just let me know which areas to expand and I'll update the file.
