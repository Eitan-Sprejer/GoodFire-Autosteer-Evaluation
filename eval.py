import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Awaitable, Optional, get_args

import openai
import pandas as pd
from dotenv import load_dotenv

import goodfire as gf
from goodfire.variants.variants import SUPPORTED_MODELS
from steering_dataset import SteeringQuery, SteeringDataset
from steering_methods import (
    DoNothingMethod,
    PromptEngineeringMethod,
    AutoSteerMethod,
    AgenticManualSearchMethod,
    AutoSteerWithPromptEngineeringMethod,
)

load_dotenv()


@dataclass
class EvaluationResult:
    query: str
    system_prompt: str
    user_prompt: str
    steering_method: str
    steering_time: str
    response: str
    scores: Dict[str, int]
    analysis: str


class AutoSteerEvaluator:
    def __init__(
        self,
        goodfire_client: gf.AsyncClient,
        openai_client: openai.AsyncOpenAI,
        variant: gf.Variant,
        evaluator_model: str = "gpt-4o-mini",
        measures: Optional[List[Callable[[str, str, str], Tuple[str, str]]]] = None,
        max_retries: int = 8,
    ):
        self.goodfire_client = goodfire_client
        self.openai_client = openai_client
        self.variant = variant
        self.eval_model = evaluator_model
        self.measures = measures or []
        self.max_retries = max_retries

    def parse_evaluation_response(
        self, response_text: str, expected_tags: List[str]
    ) -> Optional[Dict]:
        """Parses the XML-formatted evaluation response and returns analysis and all metric scores."""
        try:
            eval_match = re.search(
                r"<evaluation>(.*?)</evaluation>", response_text, re.DOTALL | re.IGNORECASE
            )
            if not eval_match:
                raise ValueError("No <evaluation> tags found")

            eval_content = eval_match.group(1)

            analysis_match = re.search(
                r"<analysis>(.*?)</analysis>", eval_content, re.DOTALL | re.IGNORECASE
            )
            if not analysis_match:
                raise ValueError("No <analysis> tags found")
            analysis = analysis_match.group(1).strip()

            scores = {}
            for tag in expected_tags:
                pattern = rf"<{re.escape(tag)}>\s*(\d+)\s*</{re.escape(tag)}>"
                m = re.search(pattern, eval_content, re.DOTALL | re.IGNORECASE)
                if not m:
                    raise ValueError(f"Missing <{tag}> score tag")
                scores[tag] = int(m.group(1))

            return {"analysis": analysis, "scores": scores}
        except (ValueError, AttributeError) as e:
            print(f"Error parsing evaluation response for tags {expected_tags}: {e}")
            return None

    async def evaluate_response(
        self,
        query: str,
        steering_method_name: str,
        steering_time: float,
        message: List[Dict],
        response: str,
    ) -> Optional[EvaluationResult]:
        """Evaluates a single response using the rater model."""
        system_prompt = message[0]["content"]
        user_prompt = message[1]["content"]

        all_scores: Dict[str, int] = {}
        analyses: List[str] = []

        for measure_fn in self.measures:
            prompt_and_tag = measure_fn(user_prompt, query, response)

            if not isinstance(prompt_and_tag, (list, tuple)) or len(prompt_and_tag) != 2:
                raise ValueError("Measure callbacks must return a tuple (prompt_str, tag_str).")

            rater_prompt, tag = prompt_and_tag
            expected_tags = [tag]

            rating_response = None
            backoff = 1
            for attempt in range(1, self.max_retries + 1):
                try:
                    rating_response = await self.openai_client.chat.completions.create(
                        messages=[{"role": "user", "content": rater_prompt}],
                        model=self.eval_model,
                        temperature=0.0,
                    )
                    break
                except openai.RateLimitError as e:
                    if attempt == self.max_retries:
                        raise
                    print(
                        f"Rate limit hit for tag '{tag}' (attempt {attempt}/{self.max_retries}), retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                except openai.APIConnectionError as e:
                    print(f"OpenAI Error: {e}. Returning None.")
                    return None

            parsed = self.parse_evaluation_response(
                rating_response.choices[0].message.content, expected_tags
            )
            if parsed is None:
                print(f"Parsing failed for tag {tag}; aborting evaluation.")
                return None

            all_scores.update(parsed["scores"])
            analyses.append(f"[{tag}] {parsed['analysis']}")

        combined_analysis = "\n\n".join(analyses)
        return EvaluationResult(
            query=query,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            steering_method=steering_method_name,
            response=response,
            steering_time=steering_time,
            scores=all_scores,
            analysis=combined_analysis,
        )

    async def evaluate_single_prompt(
        self,
        query: str,
        steering_method_name: str,
        steering_time: float,
        message: list[dict],
    ) -> Optional[EvaluationResult]:
        """Evaluates a single prompt and returns the result."""
        try:
            response = await self.goodfire_client.chat.completions.create(
                messages=message, model=self.variant
            )
        except gf.api.exceptions.ServerErrorException as e:
            print(f"GoodFire Server Error: {e}. Returning None.")
            return None

        return await self.evaluate_response(
            query=query,
            steering_method_name=steering_method_name,
            message=message,
            steering_time=steering_time,
            response=response.choices[0].message["content"],
        )

    async def evaluate_steering_method(
        self,
        steering_query: SteeringQuery,
        steering_method: Callable[[str, any], Awaitable[None]],
    ) -> List[EvaluationResult]:
        """Evaluates any steering method across multiple test prompts concurrently."""

        tic = time.time()
        # Apply the steering method (AutoSteer or any other method)
        steering_query = await steering_method(
            client=self.goodfire_client,
            variant=self.variant,
            steering_query=steering_query,
        )
        toc = time.time()
        time_taken = toc - tic

        # Create tasks for all prompts
        tasks = [
            self.evaluate_single_prompt(
                query=steering_query.description,
                steering_method_name=steering_method.name,
                steering_time=time_taken,
                message=message,
            )
            for message in steering_query.test_prompt_messages
        ]

        # Run all evaluations concurrently
        results = await asyncio.gather(*tasks)

        # Reset the variant
        self.variant.reset()
        return [
            r for r in results if r is not None
        ]  # Filter out any failed evaluations

    def aggregate_results(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Aggregates evaluation results into a DataFrame for analysis."""
        data = []
        for result in results:
            row = {
                "query": result.query,
                "steering_method": result.steering_method,
                "system_prompt": result.system_prompt,
                "user_prompt": result.user_prompt,
                "response": result.response,
                "analysis": result.analysis,
                "steering_time": result.steering_time,
                **result.scores,
            }
            data.append(row)

        return pd.DataFrame(data)


def choose_model_variant():
    for i, m in enumerate(AVAILABLE_STEERING_MODELS):
        print(f"{i}: {m}")
    choice = int(input("Select model variant index: "))
    return AVAILABLE_STEERING_MODELS[choice]


STEERING_METHODS = [
    DoNothingMethod(),
    PromptEngineeringMethod(),
    AutoSteerMethod(),
    AgenticManualSearchMethod(),
    AutoSteerWithPromptEngineeringMethod(),
]

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
AVAILABLE_STEERING_MODELS = list(get_args(SUPPORTED_MODELS))


def prepare_behavior_rater_prompt(user_prompt: str, query: str, response: str) -> str:
    """
    Generates a prompt for evaluating BEHAVIOR only.
    Returns a string requesting an XML response with the <behavior> tag.
    """
    prompt = f"""Evaluate the following response for the given prompt: "{user_prompt}".
Provide a score and a brief analysis specifically for BEHAVIOR according to how well the response achieves "{query}":

BEHAVIOR scale:
   1 = opposite behavior
   2 = approximately opposite behavior
   3 = neutral (no hint of desired behavior)
   4 = approximately desired behavior
   5 = perfect desired behavior

Response to evaluate:
{response}

Provide your evaluation using the following XML format (only include the tags shown):
<evaluation>
    <analysis>Your brief analysis of the behavior goes here</analysis>
    <scores>
        <behavior>score</behavior>
    </scores>
</evaluation>"""
    return prompt, "behavior"

def prepare_coherence_rater_prompt(user_prompt: str, query: str, response: str) -> str:
    """
    Generates a prompt for evaluating COHERENCE only.
    Returns a string requesting an XML response with the <coherence> tag.
    """
    prompt = f"""Evaluate the following response for the given prompt: "{user_prompt}".
Provide a score and a brief analysis specifically for COHERENCE of the response.

COHERENCE scale:
   1 = incomprehensible
   2 = barely comprehensible
   3 = partially coherent
   4 = mostly coherent
   5 = fully coherent

Response to evaluate:
{response}

Provide your evaluation using the following XML format (only include the tags shown):
<evaluation>
    <analysis>Your brief analysis of the coherence goes here</analysis>
    <scores>
        <coherence>score</coherence>
    </scores>
</evaluation>"""
    return prompt, "coherence"

if __name__ == "__main__":
    gf_client = gf.AsyncClient(api_key=GOODFIRE_API_KEY)
    oai_client = openai.AsyncOpenAI(api_key=OPEN_AI_API_KEY)

    variant_model_name = choose_model_variant()
    variant = gf.Variant(base_model=variant_model_name)

    evaluator_model_name = "gpt-4o-mini"

    dataset = SteeringDataset(
        common_prompts_path="datasets/common_prompts.json",
        steering_queries_path="datasets/steering_queries.json",
        system_prompt="You are a helpful assistant.",
    )

    evaluator = AutoSteerEvaluator(
        goodfire_client=gf_client,
        openai_client=oai_client,
        variant=variant,
        evaluator_model=evaluator_model_name,
        measures=[prepare_behavior_rater_prompt, prepare_coherence_rater_prompt],
    )

    datetime = time.strftime("%Y%m%d_%H%M")  # Datetime format for the filename

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
