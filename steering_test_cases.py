import random, json
from dataclasses import dataclass

SYSTEM_PROMPT = "You are a helpful assistant."

with open("datasets/common_prompts.json", "r", encoding="utf-8") as f:
    COMMON_TEST_PROMPT_MESSAGES = json.load(f)

with open("datasets/steering_queries.json", "r", encoding="utf-8") as f:
    _RAW_QUERIES = json.load(f)

def create_test_prompt_message_set(topic_specific_prompts, n_common_prompts=20, random_seed=42):
    """
    Creates a set of test prompts by combining topic-specific prompts
    with randomly sampled common prompts
    """
    random.seed(random_seed)  # Set the random seed for reproducibility
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in [
            *topic_specific_prompts,
            *random.sample(COMMON_TEST_PROMPT_MESSAGES, n_common_prompts),
        ]
    ]

@dataclass
class SteeringQuery:
    description: str
    test_prompt_messages: list[list[dict[str, str]]]

SAMPLE_STEERING_QUERIES = [
    SteeringQuery(
        description=item["description"],
        test_prompt_messages=create_test_prompt_message_set(
            item["topic_specific_prompts"],
            n_common_prompts=0,
            random_seed=item.get("random_seed", 42),
        )
    )
    for item in _RAW_QUERIES
]

if __name__ == "__main__":
    for i, query in enumerate(SAMPLE_STEERING_QUERIES):
        print(f"\n=== Steering Query {i+1}: {query.description} ===")
        for j, message_pair in enumerate(query.test_prompt_messages[:5]):  # solo muestra 5 prompts para debug
            user_prompt = message_pair[1]["content"]
            print(f"  Prompt {j+1}: {user_prompt}")
