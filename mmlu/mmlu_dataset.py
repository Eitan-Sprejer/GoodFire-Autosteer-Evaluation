import string
import random
import json
import warnings
from typing import List, Optional, Dict, Union, Any

from datasets import (
    load_dataset,
    get_dataset_config_names,
    Dataset,
    concatenate_datasets,
)

from mmlu.prompt_utils import PromptUtils


class MMLUDataset:
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
                raise RuntimeError(f"Error fetching dataset configs from HF: {e}")
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
                    print(
                        f"({idx}/{len(to_load)}) Loading config '{cfg}' split='{self.split}' ..."
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
                warnings.warn(f"Failed loading config '{cfg}': {e}", UserWarning)

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

    @staticmethod
    def save_as_json(dataset: Union[Dataset, Any], output_path: str) -> None:
        """
        Save a Hugging Face Dataset object to a JSON file.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if isinstance(dataset, Dataset):
                    json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
                else:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset as JSON: {e}")

    @staticmethod
    def save_topic_specific_json(
        dataset: Dataset,
        descriptions: List[str],
        num_prompts: int,
        output_path: str,
        random_seed: int = 1,
    ) -> None:
        """
        Save a Hugging Face Dataset object to a JSON file with the SteeringDataset Format.

        - descriptions: list of strings, each one will be a separate entry.
        - num_prompts: number of random prompts to select for each description.
        """
        MMLUDataset.save_as_json(
            [
                {
                    "description": description,
                    "topic_specific_prompts": list(prompts),
                    "answers": list(answers),
                    "random_seed": random_seed,
                    "n_common_prompts": 0,
                }
                for description in descriptions
                for chosen_examples in [random.sample(list(dataset), k=num_prompts)]
                for prompts, answers in [
                    zip(
                        *[
                            (
                                PromptUtils.build_cot_prompt(ex),
                                string.ascii_uppercase[int(ex["answer"])],
                            )
                            for ex in chosen_examples
                        ]
                    )
                ]
            ],
            output_path,
        )
