import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class SteeringQuery:
    description: str
    test_prompt_messages: List[List[Dict[str, str]]]


class SteeringDataset:
    """
    Class to load and manage datasets of steering queries and common prompts.
    Allows generating prompt sets for reproducible tests.
    """

    def __init__(
        self,
        common_prompts_path: str,
        steering_queries_path: str,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.system_prompt = system_prompt
        self.common_prompts = self._load_json(common_prompts_path)
        self.raw_queries = self._load_json(steering_queries_path)
        self.queries = self._build_queries()

    @staticmethod
    def _load_json(path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_test_prompt_message_set(
        self,
        topic_specific_prompts: List[str],
        n_common_prompts: int = 20,
        random_seed: int = 42,
    ) -> List[List[Dict[str, str]]]:
        """
        Combines topic-specific prompts with a random sample
        of common prompts, returning messages ready for an LLM.
        """
        random.seed(random_seed)
        prompts = [
            *topic_specific_prompts,
            *random.sample(self.common_prompts, n_common_prompts),
        ]
        return [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

    def _build_queries(self) -> List[SteeringQuery]:
        """
        Builds the list of SteeringQuery objects from raw data.
        """
        return [
            SteeringQuery(
                description=item["description"],
                test_prompt_messages=self.create_test_prompt_message_set(
                    item["topic_specific_prompts"],
                    n_common_prompts=item.get("n_common_prompts", 0),
                    random_seed=item.get("random_seed", 42),
                ),
            )
            for item in self.raw_queries
        ]

    def get_queries(self) -> List[SteeringQuery]:
        """
        Returns the loaded list of SteeringQuery objects.
        """
        return self.queries

    def get_query_by_description(self, description: str) -> SteeringQuery:
        """
        Finds and returns a SteeringQuery by its description.
        """
        for q in self.queries:
            if q.description == description:
                return q
        raise ValueError(f"No query found with description '{description}'")


if __name__ == "__main__":
    dataset = SteeringDataset(
        common_prompts_path="datasets/common_prompts.json",
        steering_queries_path="datasets/steering_queries.json",
        system_prompt="You are a helpful assistant.",
    )
    all_queries = dataset.get_queries()
    for i, query in enumerate(all_queries):
        print(f"\n=== Steering Query {i+1}: {query.description} ===")
        for j, message_pair in enumerate(query.test_prompt_messages[:5]):
            user_prompt = message_pair[1]["content"]
            print(f"  Prompt {j+1}: {user_prompt}")
