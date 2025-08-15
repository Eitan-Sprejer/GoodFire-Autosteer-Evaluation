import asyncio
import re
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Awaitable, Optional

import openai
import goodfire as gf
import pandas as pd

from goodfire_eval.steering_dataset import SteeringQuery


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


class SteeringEvaluator:
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
                r"<evaluation>(.*?)</evaluation>",
                response_text,
                re.DOTALL | re.IGNORECASE,
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

            if (
                not isinstance(prompt_and_tag, (list, tuple))
                or len(prompt_and_tag) != 2
            ):
                raise ValueError(
                    "Measure callbacks must return a tuple (prompt_str, tag_str)."
                )

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
