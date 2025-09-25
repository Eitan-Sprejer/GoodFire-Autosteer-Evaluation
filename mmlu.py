import random

import pandas as pd

from goodfire_eval.steering_dataset import SteeringDataset
from mmlu.mmlu_dataset import MMLUDataset
from mmlu.prompt_utils import PromptUtils
from mmlu.steering_closed_dataset import SteeringClosedDataset


if __name__ == "__main__":
    loader = MMLUDataset(categories=["all"], split="dev")
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

    ### STEERING CLOSED DATASET (MMLU)
    from goodfire_eval.steering_dataset import SteeringDataset

    random.seed(42)

    # MMLUDataset.save_as_json(combined, "mmlu.json")
    dataset = SteeringDataset(
        common_prompts_path="datasets/common_prompts.json",
        steering_queries_path="datasets/steering_queries.json",
        system_prompt="You are a helpful assistant.",
    )
    all_queries = dataset.get_queries()
    # loader.save_topic_specific_json(
    #     dataset=combined,
    #     output_path="datasets/steering_queries_mmlu.json",
    #     descriptions=[item.description for item in all_queries],
    #     num_prompts=30,
    #     random_seed=42,
    # )
    dataset = SteeringClosedDataset(
        common_prompts_path="datasets/common_prompts.json",
        steering_queries_path="datasets/steering_queries_mmlu.json",
        system_prompt="You are a helpful assistant.",
    )
    all_queries = dataset.get_queries()
    for i, query in enumerate(all_queries):
        print(f"\n=== Steering Query {i+1}: {query.description} ===")
        for j, message_pair in enumerate(query.test_prompt_messages[:5]):
            user_prompt = message_pair[1]["content"]
            print(f"  Prompt {j+1}: {user_prompt}")
            print(f"  Answer {j+1}: {query.answers[j]}")
    filename_no_ext = (
        "eval_gpt-4o-mini_var_Llama-3.3-70B-Instruct_dt_20250925_1548_merged"
    )
    df = pd.read_csv(f"results/{filename_no_ext}.csv")
    df_eval = dataset.evaluate_responses(df)
    print(
        df_eval[
            ["query", "steering_method", "predicted_answer", "gold_answer", "result"]
        ].head(20)
    )
    df_eval.to_csv(
        f"results/{filename_no_ext}_answers.csv",
        index=False,
    )
