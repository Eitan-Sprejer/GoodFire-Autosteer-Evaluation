# Evaluating Feature Steering Using GoodFire's AutoSteer

An empirical evaluation of GoodFire's AutoSteer feature for controlling language model behavior through natural language specifications.

## Overview

This project evaluates the effectiveness of AutoSteer, a feature steering mechanism that automatically generates model's SAE feature's interventions based on natural language descriptions of desired behaviors. The evaluation focuses on testing AutoSteer's ability to reliably modify model outputs across different scenarios and model variants. For methodological details and results, see `results/research.md`.

## Project Structure

```
.
├── results/                      # Evaluation results and analysis
│   ├── research.md               # Article that details the findings.
│   ├── eval_gpt-4o-mini_var_llama-3.1_dt_20250116_1526.csv
│   ├── eval_gpt-4o-mini_var_llama-3.3_dt_20250119_1134.csv
│   └── eval_llama-3.3_var_llama-3.1_dt_20250116_1526.csv
├── steering_methods.py           # Implementation of steering techniques
├── steering_test_cases.py        # Test cases and evaluation scenarios
├── eval.py                       # Main evaluation script
└── testing.ipynb                 # Interactive testing notebook
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/goodfire-autosteer-evaluation.git
cd goodfire-autosteer-evaluation
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your API keys in the `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
GOODFIRE_API_KEY=your_goodfire_key_here
```

2. Ensure your virtual environment is activated:
```bash
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Configure the evaluation parameters in `eval.py`:
   - Set your variant model
   - Set your evaluation model
   - Adjust any other desired parameters

4. Run the evaluation:
```bash
python eval.py
```

### Customizing Test Cases

You can modify existing test cases or add new ones by editing `steering_test_cases.py`. The file is structured to make adding new test scenarios straightforward.

## Research Findings

Detailed research findings and methodology are available in `results/research.md`.

## Contact

Please, feel free to send me an email at eitusprejer@gmail.com if you have any questions.