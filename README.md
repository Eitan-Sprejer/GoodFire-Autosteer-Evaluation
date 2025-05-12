# Evaluating Feature Steering Using GoodFire's AutoSteer

An empirical evaluation of GoodFire's AutoSteer feature for controlling language model behavior through natural language specifications.

This is the github repository connected to the [blogpost](https://www.alignmentforum.org/posts/6dpKhtniqR3rnstnL/mind-the-coherence-gap-lessons-from-steering-llama-with-1?utm_campaign=post_share&utm_source=link) I wrote inside the Lesswrong community.
## Overview

This project evaluates the effectiveness of AutoSteer, a feature steering mechanism that automatically generates model's SAE feature's interventions based on natural language descriptions of desired behaviors. The evaluation focuses on testing AutoSteer's ability to reliably modify model outputs across different scenarios and model variants. For methodological details and results, see the following [blogpost](https://www.alignmentforum.org/posts/6dpKhtniqR3rnstnL/mind-the-coherence-gap-lessons-from-steering-llama-with-1?utm_campaign=post_share&utm_source=link).

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
└── analysis.ipynb                # Interactive results notebook
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

5. Analize the results:
   - Go though the `analysis.ipynb notebook`, plotting the results of your different experiments.

### Customizing Your Experiment

You can modify existing test cases or add new ones by editing `steering_test_cases.py`. The file is structured to make adding new test scenarios straightforward.

Likewise, you can add new steering methods inside the steering_methods.py notebook.

## Research Findings

Detailed research findings and methodology are available in the following [blogpost](https://www.alignmentforum.org/posts/6dpKhtniqR3rnstnL/mind-the-coherence-gap-lessons-from-steering-llama-with-1?utm_campaign=post_share&utm_source=link).

## Contact

Please, feel free to send me an email at eitusprejer@gmail.com if you have any questions. Feedback is more than welcome!