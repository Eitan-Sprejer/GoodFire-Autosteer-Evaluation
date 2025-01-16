import random
from dataclasses import dataclass
from typing import List

SYSTEM_PROMPT = "You are a helpful assistant."

# Expanded pool of common prompts covering various domains and interaction types
COMMON_TEST_PROMPT_MESSAGES = [
    # General Knowledge/Educational
    "Explain how a computer works",
    "What causes earthquakes?",
    "How do planes fly?",
    "Why is the sky blue?",
    "How does photosynthesis work?",
    "Explain evolution in simple terms",
    # Day-to-day Tasks
    "How do I make pasta?",
    "What's the best way to clean windows?",
    "How do I change a tire?",
    "Give me tips for better sleep",
    "How do I remove a coffee stain?",
    "What's a good morning routine?",
    # Creative/Entertainment
    "Write a story about a cat",
    "Create a recipe for happiness",
    "Describe your perfect vacation",
    "Tell me a joke",
    "Write a haiku about summer",
    "Invent a new holiday",
    # Technology
    "What's the difference between WiFi and Bluetooth?",
    "How do I protect my online privacy?",
    "Explain cloud computing",
    "What is artificial intelligence?",
    "How do smartphones work?",
    "What is cryptocurrency?",
    # Current Events/Society
    "What's your opinion on social media?",
    "How does recycling help the environment?",
    "Explain remote work pros and cons",
    "What makes a good leader?",
    "How does inflation affect daily life?",
    "What is climate change?",
    # Personal/Emotional
    "How do I make new friends?",
    "What to do when feeling stressed?",
    "How to stay motivated?",
    "Ways to improve self-confidence",
    "How to deal with failure?",
    "Tips for work-life balance",
    # Business/Professional
    "How to write a good resume?",
    "Tips for public speaking",
    "How to manage a team effectively?",
    "Ways to improve productivity",
    "How to handle difficult conversations?",
    "Best practices for time management",
]


def create_test_prompt_message_set(topic_specific_prompts, n_common_prompts=5):
    """
    Creates a set of test prompts by combining topic-specific prompts
    with randomly sampled common prompts
    """
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
    description: str  # e.g., "be funny"
    test_prompt_messages: List[str]  # prompts to test the behavior on


SAMPLE_STEERING_QUERIES = [
    SteeringQuery(
        description="be funny",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Explain quantum physics",
                "Give me a recipe for chocolate cake",
                "Tell me about the history of Rome",
                # Potentially challenging contexts
                "Write a condolence message",
                "Explain a serious medical condition",
            ]
        ),
    ),
    SteeringQuery(
        description="be professional and formal",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Write an email to decline a business proposal",
                "Draft a project status update",
                "Provide feedback on an employee's performance",
                # Potentially challenging contexts
                "Tell me a bedtime story",
                "Write a casual tweet",
            ]
        ),
    ),
    SteeringQuery(
        description="be more creative and imaginative",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Design a fictional creature",
                "Create a new superhero origin story",
                "Invent a new musical instrument",
                # Potentially challenging contexts
                "Explain how to do taxes",
                "Write a technical manual",
            ]
        ),
    ),
    SteeringQuery(
        description="be concise and direct",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "How do I reset my password?",
                "Give directions to the nearest hospital",
                "What's the difference between RAM and ROM?",
                # Potentially challenging contexts
                "Describe your perfect day",
                "Write a love poem",
            ]
        ),
    ),
    SteeringQuery(
        description="be empathetic and supportive",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "I failed my exam",
                "I'm feeling overwhelmed at work",
                "My pet passed away recently",
                # Potentially challenging contexts
                "Debug this code snippet",
                "Explain how blockchain works",
            ]
        ),
    ),
    SteeringQuery(
        description="be educational and explain like a teacher",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "How does the immune system work?",
                "Explain multiplication to a child",
                "What is climate change?",
                # Potentially challenging contexts
                "Write a resignation letter",
                "Create a marketing slogan",
            ]
        ),
    ),
    SteeringQuery(
        description="be skeptical and analytical",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Is this investment opportunity legitimate?",
                "Evaluate this scientific claim",
                "Review this research methodology",
                # Potentially challenging contexts
                "Write a children's story",
                "Create an inspirational quote",
            ]
        ),
    ),
    SteeringQuery(
        description="be motivational and inspiring",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "I want to start exercising",
                "I'm thinking of changing careers",
                "I want to learn a new language",
                # Potentially challenging contexts
                "Explain how a CPU works",
                "Debug this error message",
            ]
        ),
    ),
]
