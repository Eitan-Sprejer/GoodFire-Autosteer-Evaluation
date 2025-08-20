from dataclasses import dataclass
from typing import List

import pandas as pd

from goodfire_eval.steering_dataset import SteeringDataset, SteeringQuery
from mmlu.prompt_utils import PromptUtils


@dataclass
class SteeringClosedQuery(SteeringQuery):
    """
    Extension of SteeringQuery that also stores the correct answers.
    """

    answers: List[str]


class SteeringClosedDataset(SteeringDataset):
    """
    Same as SteeringDataset but builds SteeringClosedQuery objects
    with both prompts and answers.
    """

    def _build_queries(self) -> List[SteeringClosedQuery]:
        """
        Builds the list of SteeringClosedQuery objects from raw data.
        Assumes `answers` key exists in the raw JSON for each item.
        """
        return [
            SteeringClosedQuery(
                description=item["description"],
                test_prompt_messages=self.create_test_prompt_message_set(
                    item["topic_specific_prompts"],
                    n_common_prompts=item.get("n_common_prompts", 0),
                    random_seed=item.get("random_seed", 42),
                ),
                answers=item.get("answers", []),
            )
            for item in self.raw_queries
        ]

    def _evaluate_hit(self, pred: str, gold: str) -> str:
        """Return evaluation label based on predicted and gold answers."""
        if pred is None:
            return "Miss (no <answer>)"
        if gold is None:
            return "Miss"
        return "Hit" if pred.strip().upper()[0] == gold.strip().upper()[0] else "Miss"

    def evaluate_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add evaluation columns to a DataFrame containing at least ['query', 'steering_method', 'response'].
        - predicted_answer: extracted from <answer> tags
        - gold_answer: the correct answer from the dataset
        - result: denotes if the prediction is a 'Hit' or identifies the type of 'Miss'.
        """
        df = df.copy()
        df[["predicted_answer", "gold_answer", "result"]] = None

        # dict for lookup by description
        queries_by_desc = {q.description: q for q in self.queries}

        for method in df["steering_method"].unique():
            df_method = df[df["steering_method"] == method]
            for desc, group in df_method.groupby("query", sort=False):
                query = queries_by_desc.get(desc)
                gold_answers = query.answers if query else []
                for i, (idx, row) in enumerate(group.iterrows()):
                    parsed = PromptUtils.parse_cot_response(row.get("response", ""))
                    pred = parsed.get("answer")
                    gold = gold_answers[i] if i < len(gold_answers) else None
                    res = self._evaluate_hit(pred, gold)
                    df.loc[idx, 
                           ["predicted_answer", "gold_answer", "result"]] = [pred, gold, res]  # fmt: skip
        return df
