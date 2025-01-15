from dataclasses import dataclass
from typing import List, Dict, Callable, Awaitable
import json
import pandas as pd
import numpy as np
from enum import Enum
import goodfire as gf
import asyncio
from dotenv import load_dotenv
import re
load_dotenv()
import os
from steering_test_cases import SAMPLE_STEERING_QUERIES, SteeringQuery

GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
AVAILABLE_MODELS = {
    "llama-3.1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.3": "meta-llama/Llama-3.3-70B-Instruct",
}

class Criterion(Enum):
    """
    Scoring criteria:
    BEHAVIOR:
    1: opposite behavior
    2: approximately opposite behavior
    3: neutral (no hint of desired behavior)
    4: approximately desired behavior
    5: perfect desired behavior
    
    COHERENCE:
    1: incomprehensible
    2: barely comprehensible
    3: partially coherent
    4: mostly coherent
    5: fully coherent
    """
    BEHAVIOR = "behavior"
    COHERENCE = "coherence"

@dataclass
class EvaluationResult:
    query: str
    prompt: str
    response: str
    scores: Dict[str, int]  # Will contain behavior and coherence scores
    analysis: str  # Added field for analysis

class AutoSteerEvaluator:
    def __init__(
            self,
            async_client: gf.AsyncClient,
            variant: gf.Variant,
            evaluator_model: str = AVAILABLE_MODELS['llama-3.1']
    ):
        self.client = async_client
        self.variant = variant
        self.eval_model = evaluator_model
        
    def prepare_rater_prompt(self, query: str, response: str) -> str:
        """Creates a structured prompt for the rater model using XML tags."""
        prompt = f"""Evaluate the following response for {query}. 
Provide scores and analysis using these criteria:

1. BEHAVIOR (how well it achieves "{query}"):
1 = opposite behavior
2 = approximately opposite behavior
3 = neutral (no hint of desired behavior)
4 = approximately desired behavior
5 = perfect desired behavior

2. COHERENCE:
1 = incomprehensible
2 = barely comprehensible
3 = partially coherent
4 = mostly coherent
5 = fully coherent

Response to evaluate:
{response}

Provide your evaluation using the following XML format:
<evaluation>
    <analysis>Your brief analysis of the response goes here</analysis>
    <scores>
        <behavior>score</behavior>
        <coherence>score</coherence>
    </scores>
</evaluation>"""
        return prompt

    def parse_evaluation_response(self, response_text: str) -> Dict:
        """Parses the XML-formatted evaluation response."""
        try:
            # Extract content between evaluation tags
            eval_match = re.search(r'<evaluation>(.*?)</evaluation>', response_text, re.DOTALL)
            if not eval_match:
                raise ValueError("No evaluation tags found")
            
            eval_content = eval_match.group(1)
            
            # Extract analysis
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', eval_content, re.DOTALL)
            if not analysis_match:
                raise ValueError("No analysis tags found")
            analysis = analysis_match.group(1).strip()
            
            # Extract scores
            behavior_match = re.search(r'<behavior>(\d+)</behavior>', eval_content)
            coherence_match = re.search(r'<coherence>(\d+)</coherence>', eval_content)
            
            if not (behavior_match and coherence_match):
                raise ValueError("Missing score tags")
            
            return {
                "analysis": analysis,
                "scores": {
                    "behavior": int(behavior_match.group(1)),
                    "coherence": int(coherence_match.group(1))
                }
            }
            
        except (ValueError, AttributeError) as e:
            print(f"Error parsing evaluation response: {e}")
            return None

    async def evaluate_response(self, query: str, prompt: str, response: str) -> EvaluationResult:
        """Evaluates a single response using the rater model."""
        rater_prompt = self.prepare_rater_prompt(query, response)
        
        # Get rating from model
        rating_response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": rater_prompt}],
            model=self.eval_model
        )
        
        # Parse the XML response
        parsed_data = self.parse_evaluation_response(rating_response.choices[0].message['content'])
        if parsed_data is None:
            return None
            
        return EvaluationResult(
            query=query,
            prompt=prompt,
            response=response,
            scores=parsed_data["scores"],
            analysis=parsed_data["analysis"]
        )

    async def evaluate_single_prompt(
        self,
        query: str,
        prompt: str,
    ) -> EvaluationResult:
        """Evaluates a single prompt and returns the result."""
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.variant
        )
        
        return await self.evaluate_response(
            query,
            prompt,
            response.choices[0].message['content']
        )

    async def evaluate_steering_method(
        self,
        steering_query: SteeringQuery,
        steering_method: Callable[[str, any], Awaitable[None]]
    ) -> List[EvaluationResult]:
        """Evaluates any steering method across multiple test prompts concurrently."""
        
        # Apply the steering method (AutoSteer or any other method)
        await steering_method(
            client=self.client,
            description=steering_query.description,
            variant=self.variant
        )
        print(self.variant)
        
        # Create tasks for all prompts
        tasks = [
            self.evaluate_single_prompt(
                steering_query.description,
                prompt,
            )
            for prompt in steering_query.test_prompts
        ]
        
        # Run all evaluations concurrently
        results = await asyncio.gather(*tasks)

        # Reset the variant
        self.variant.reset()
        return [r for r in results if r is not None]  # Filter out any failed evaluations

    def aggregate_results(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Aggregates evaluation results into a DataFrame for analysis."""
        data = []
        for result in results:
            row = {
                "query": result.query,
                "prompt": result.prompt,
                "response": result.response,
                "analysis": result.analysis,  # Added analysis to output
                **result.scores
            }
            data.append(row)
        
        return pd.DataFrame(data)

# Example steering methods
async def autosteer_method(client, description: str, variant) -> None:
    """Original AutoSteer method"""
    edits = await client.features.AutoSteer(
        specification=description,  # Natural language description
        model=variant,  # Model variant to use
    )
    variant.set(edits)


if __name__ == '__main__':
    client = gf.AsyncClient(api_key=GOODFIRE_API_KEY)
    variant = gf.Variant(base_model=AVAILABLE_MODELS['llama-3.1'])
    evaluator = AutoSteerEvaluator(
        client, variant=variant, evaluator_model=AVAILABLE_MODELS['llama-3.3']
    )

    results = []
    for query in SAMPLE_STEERING_QUERIES[2:]:
        print(f"Query: {query.description}")
        # Test AutoSteer
        results.append(asyncio.run(evaluator.evaluate_steering_method(
            query,
            steering_method=autosteer_method,
        )))

    # Make a DataFrame with the results
    df = evaluator.aggregate_results([r for sublist in results for r in sublist])
    df.to_csv("2_evaluation_results.csv", index=False)
