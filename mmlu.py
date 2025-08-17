from typing import List, Optional, Dict
from datasets import (
    load_dataset,
    get_dataset_config_names,
    Dataset,
    concatenate_datasets,
)
import string
import random
import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


import string
import re


class PromptUtils:

    @staticmethod
    def build_cot_prompt(example: dict) -> str:
        question = example["question"].strip()
        choices = example["choices"]
        labels = string.ascii_uppercase
        choices_lines = [f"{labels[i]}. {c}" for i, c in enumerate(choices)]

        instruction = (
            "\n\nThis is a Chain-of-Thought (CoT) task.\n"
            "First, think step-by-step inside <thinking>...</thinking> tags.\n"
            "You MUST NOT write any reasoning outside the <thinking> block.\n"
            "Then, give only the final answer — a single letter (A, B, C, ...) — inside <answer>...</answer> tags."
        )

        return question + "\n" + "\n".join(choices_lines) + instruction

    @staticmethod
    def parse_cot_response(text: str) -> dict:
        tags = ["thinking", "answer"]
        response_cot = {}

        for tag in tags:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            response_cot[tag] = match.group(1).strip() if match else None

        return response_cot


class MMLUDatasetLoader:
    """
    Loader for the `cais/mmlu` dataset from Hugging Face.

    Parameters:
    - dataset_id: HF identifier (default "cais/mmlu").
    - categories: None (all) or list of category/config names to load.
    - split: split to load for each config (e.g. "dev" | "validation" | "test").
    - trust_remote_code: passed to load_dataset if the dataset requires executing remote code.
    - streaming: if True, loads in streaming mode (IterableDataset).
    """

    def __init__(
        self,
        dataset_id: str = "cais/mmlu",
        categories: Optional[List[str]] = None,
        split: str = "dev",
        trust_remote_code: bool = False,
        streaming: bool = False,
    ):
        self.dataset_id = dataset_id
        self.requested_categories = categories
        self.split = split
        self.trust_remote_code = trust_remote_code
        self.streaming = streaming

        # These are populated after calling load()
        self.available_categories: List[str] = []
        self.loaded: Dict[str, Dataset] = {}

    def list_available_categories(self) -> List[str]:
        """Fetches the dataset configurations from HF and returns them (caches the result)."""
        if not self.available_categories:
            try:
                self.available_categories = get_dataset_config_names(self.dataset_id)
            except Exception as e:
                logger.exception("Error fetching dataset configs from HF: %s", e)
                raise
        return self.available_categories

    def _resolve_categories_to_load(self) -> List[str]:
        """Resolves requested_categories -> concrete list of config names (or all)."""
        available = self.list_available_categories()
        if self.requested_categories is None:
            return available
        # if a list was passed, validate
        bad = [c for c in self.requested_categories if c not in available]
        if bad:
            raise ValueError(
                f"The following categories do not exist in the dataset: {bad}"
            )
        return list(self.requested_categories)

    def load(self, progress: bool = True) -> Dict[str, Dataset]:
        """
        Loads the requested categories from the specified split.
        Returns a dict mapping category -> Dataset.
        """
        to_load = self._resolve_categories_to_load()
        self.loaded = {}

        for idx, cfg in enumerate(to_load, 1):
            try:
                if progress:
                    logger.info(
                        "(%d/%d) Loading config '%s' split='%s' ...",
                        idx,
                        len(to_load),
                        cfg,
                        self.split,
                    )
                ds = load_dataset(
                    self.dataset_id,
                    cfg,
                    split=self.split,
                    trust_remote_code=self.trust_remote_code,
                    streaming=self.streaming,
                )
                # if not streaming and is a Dataset (not Iterable), add the category column
                if not self.streaming:
                    ds = ds.add_column("category", [cfg] * len(ds))
                self.loaded[cfg] = ds
            except Exception as e:
                # Don't abort everything: log and continue with other configs
                logger.exception("Failed loading config '%s': %s", cfg, e)

        return self.loaded

    def get_category_dataset(self, category: str) -> Dataset:
        """Returns the loaded dataset for a category (raises if not loaded)."""
        if category not in self.loaded:
            raise KeyError(f"Category '{category}' not loaded. Call load() first.")
        return self.loaded[category]

    def get_combined_dataset(self) -> Dataset:
        """
        Concatenates all loaded datasets and returns a single one.
        Only valid if not loaded in streaming mode (streaming is not concatenable here).
        """
        if self.streaming:
            raise RuntimeError("Cannot concatenate datasets in streaming mode.")
        if not self.loaded:
            raise RuntimeError("No datasets loaded. Call load() first.")
        datasets_list = list(self.loaded.values())
        combined = concatenate_datasets(datasets_list)
        return combined


if __name__ == "__main__":
    loader = MMLUDatasetLoader(categories=["all"], split="dev")
    cats = loader.list_available_categories()
    print("Available categories (sample):", cats[:10])
    loaded = loader.load()
    print("Loaded categories count:", len(loaded))
    combined = loader.get_combined_dataset()
    print("Combined size:", len(combined))
    print("Example:", combined[0])
    print(
        "Random example to prompt:",
        PromptUtils.build_cot_prompt(random.choice(combined)),
    )
    print(
        PromptUtils.parse_cot_response(
            "<thinking>2 + 2 = 4</thinking><answer>A</answer>"
        )
    )
