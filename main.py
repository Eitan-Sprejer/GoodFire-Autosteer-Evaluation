import asyncio
import os
import time
import logging
from typing import get_args

import openai
from dotenv import load_dotenv

import goodfire as gf
from goodfire.variants.variants import SUPPORTED_MODELS

from goodfire_eval.steering_evaluator import SteeringEvaluator
from goodfire_eval.steering_dataset import SteeringDataset
from goodfire_eval.steering_methods import (
    DoNothingMethod,
    PromptEngineeringMethod,
    AutoSteerMethod,
    AgenticManualSearchMethod,
    AutoSteerWithPromptEngineeringMethod,
)
from goodfire_eval.metric_rater import (
    BehaviorRater,
    CoherenceRater,
)


def choose_model_variant():
    for i, m in enumerate(AVAILABLE_STEERING_MODELS):
        print(f"{i}: {m}")
    choice = int(input("Select model variant index: "))
    return AVAILABLE_STEERING_MODELS[choice]


load_dotenv()
os.makedirs("results", exist_ok=True)

datetime = time.strftime("%Y%m%d_%H%M")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"{datetime}.log",
    filemode="a",
    encoding="utf-8",
)

STEERING_METHODS = [
    DoNothingMethod(),
    PromptEngineeringMethod(),
    AutoSteerMethod(),
    AgenticManualSearchMethod(),
    AutoSteerWithPromptEngineeringMethod(),
]

# from goodfire_eval.steering_methods import AutoSteerScaledMethod
# import numpy as np
# intensities = np.linspace(0, 1, 10)  # uniformly distributed values
# STEERING_METHODS = [AutoSteerScaledMethod(intensity=i) for i in intensities]

RATER_METRICS = [
    BehaviorRater(),
    CoherenceRater(),
]

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
AVAILABLE_STEERING_MODELS = list(get_args(SUPPORTED_MODELS))

if __name__ == "__main__":
    gf_client = gf.AsyncClient(api_key=GOODFIRE_API_KEY)
    oai_client = openai.AsyncOpenAI(api_key=OPEN_AI_API_KEY)

    variant_model_name = choose_model_variant()
    variant = gf.Variant(base_model=variant_model_name)

    evaluator_model_name = "gpt-4o-mini"

    system_prompt = (
        "You are a helpful assistant.\n"
        "First, think step-by-step inside <thinking>...</thinking> tags.\n"
        "You MUST NOT write any reasoning outside the <thinking> block.\n"
        "Then, give only the final answer inside <answer>...</answer> tags."
    )

    dataset = SteeringDataset(
        common_prompts_path="datasets/common_prompts.json",
        steering_queries_path="datasets/steering_queries.json",
        system_prompt=system_prompt,
    )

    evaluator = SteeringEvaluator(
        goodfire_client=gf_client,
        openai_client=oai_client,
        variant=variant,
        evaluator_model=evaluator_model_name,
        measures=RATER_METRICS,
    )

    results = []
    for steering_method in STEERING_METHODS:
        print(f"Evaluating steering method: {steering_method.name}")
        for query in dataset.get_queries():
            print(f"Query: {query.description}")
            # Test steering method
            results.append(
                asyncio.run(
                    evaluator.evaluate_steering_method(
                        steering_query=query,
                        steering_method=steering_method,
                    )
                )
            )
            # Make a DataFrame with the results
            df = evaluator.aggregate_results(
                [r for sublist in results for r in sublist]
            )
            df.to_csv(
                f"results/eval_{evaluator_model_name}_var_{variant_model_name.split('/')[-1]}_dt_{datetime}.csv",
                index=False,
            )
