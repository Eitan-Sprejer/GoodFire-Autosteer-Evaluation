import random
from dataclasses import dataclass
from typing import List

SYSTEM_PROMPT = "You are a helpful assistant."

# Significantly expanded pool of common prompts
COMMON_TEST_PROMPT_MESSAGES = [
    # General Knowledge/Educational
    "Explain how a computer works",
    "What causes earthquakes?",
    "How do planes fly?",
    "Why is the sky blue?",
    "How does photosynthesis work?",
    "Explain evolution in simple terms",
    "How does the human brain process information?",
    "What are black holes and how do they form?",
    "Explain the process of DNA replication",
    "How does the immune system fight diseases?",
    "What is the theory of relativity?",
    "How do vaccines work?",
    
    # Day-to-day Tasks
    "How do I make pasta?",
    "What's the best way to clean windows?",
    "How do I change a tire?",
    "Give me tips for better sleep",
    "How do I remove a coffee stain?",
    "What's a good morning routine?",
    "How do I organize a small closet?",
    "What's the best way to pack for a trip?",
    "How do I start a garden?",
    "Tips for meal prepping?",
    "How do I fix a leaky faucet?",
    "What's the best way to store fresh vegetables?",
    
    # Creative/Entertainment
    "Write a story about a cat",
    "Create a recipe for happiness",
    "Describe your perfect vacation",
    "Tell me a joke",
    "Write a haiku about summer",
    "Invent a new holiday",
    "Create a superhero origin story",
    "Write a mystery micro-fiction",
    "Design a fantasy creature",
    "Compose a love song",
    "Invent a new sport",
    "Create a fictional culture",
    
    # Technology
    "What's the difference between WiFi and Bluetooth?",
    "How do I protect my online privacy?",
    "Explain cloud computing",
    "What is artificial intelligence?",
    "How do smartphones work?",
    "What is cryptocurrency?",
    "Explain how 5G networks function",
    "What are quantum computers?",
    "How does facial recognition work?",
    "What is edge computing?",
    "Explain blockchain technology",
    "How do self-driving cars work?",
    
    # Current Events/Society
    "What's your opinion on social media?",
    "How does recycling help the environment?",
    "Explain remote work pros and cons",
    "What makes a good leader?",
    "How does inflation affect daily life?",
    "What is climate change?",
    "Discuss digital privacy concerns",
    "How is AI affecting employment?",
    "What is sustainable development?",
    "Explain the gig economy",
    "How does social media affect mental health?",
    "What is environmental justice?",
    
    # Personal/Emotional
    "How do I handle rejection?",
    "What to do when feeling stressed?",
    "How to stay motivated?",
    "Ways to improve self-confidence",
    "How to deal with failure?",
    "Tips for work-life balance",
    "How to overcome fear of public speaking",
    "Dealing with imposter syndrome",
    "Managing anxiety in social situations",
    "Building healthy relationships",
    "Coping with grief",
    "Developing emotional intelligence",
    
    # Business/Professional
    "How to write a good resume?",
    "Tips for public speaking",
    "How to manage a team effectively?",
    "Ways to improve productivity",
    "How to handle difficult conversations?",
    "Best practices for time management",
    "How to negotiate a salary",
    "Creating effective presentations",
    "Managing remote teams",
    "Building professional networks",
    "Handling workplace conflicts",
    "Developing leadership skills"
]

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
    test_prompt_messages: List[str]

SAMPLE_STEERING_QUERIES = [
    SteeringQuery(
        description="be funny",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Explain quantum physics",
                "Give me a recipe for chocolate cake",
                "Tell me about the history of Rome",
                "Explain the process of photosynthesis",
                "Write an instruction manual for tying shoelaces",
                "Describe how the stock market works",
                "Explain gravity to a child",
                # Potentially challenging contexts
                "Write a condolence message",
                "Discuss climate change impacts",
                "Explain the concept of death to a child",
            ],
            random_seed = 1),
    ),
    SteeringQuery(
        description="be professional and formal",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Write a cover letter for a job application",
                "Draft a project status update",
                "Provide feedback on an employee's performance",
                "Write a business partnership proposal",
                "Create a quarterly financial report",
                "Draft a contract amendment",
                "Write a letter of recommendation",
                # Potentially challenging contexts
                "Tell me a bedtime story",
                "Write a casual tweet",
                "Create a fun party invitation",
            ],
            random_seed = 2)
    ),
    SteeringQuery(
        description="be more creative and imaginative",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Design a fictional creature",
                "Create a new superhero origin story",
                "Invent a new musical instrument",
                "Design a city of the future",
                "Create a new magical system",
                "Invent a new form of transportation",
                "Design an alien civilization",
                # Potentially challenging contexts
                "Explain how to do taxes",
                "Write a technical manual",
                "Create a legal document",
            ],
            random_seed = 3)
    ),
    SteeringQuery(
        description="be concise and direct",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "How do I reset my password?",
                "Give directions to the nearest hospital",
                "What's the difference between RAM and ROM?",
                "How to perform CPR?",
                "Steps to change a flat tire",
                "How to create a strong password",
                "Emergency evacuation procedures",
                # Potentially challenging contexts
                "Describe your perfect day",
                "Write a love poem",
                "Create a fantasy story",
            ],
            random_seed = 4)
    ),
    SteeringQuery(
        description="be empathetic and supportive",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "I failed my exam",
                "I'm feeling overwhelmed at work",
                "My pet passed away recently",
                "I'm struggling with loneliness",
                "I didn't get the job I wanted",
                "I'm having relationship problems",
                "I'm dealing with burnout",
                # Potentially challenging contexts
                "Debug this code snippet",
                "Explain how blockchain works",
                "Analyze this market trend",
            ],
            random_seed = 5)
    ),
    SteeringQuery(
        description="be educational and explain like a teacher",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "How does the immune system work?",
                "Explain multiplication to a child",
                "What is climate change?",
                "How do plants grow from seeds?",
                "Why do we have seasons?",
                "What causes thunder and lightning?",
                "How do computers store information?",
                # Potentially challenging contexts
                "How do I make new friends?",
                "Create a marketing slogan",
                "Write a complaint letter",
            ],
            random_seed = 6)
    ),
    SteeringQuery(
        description="be skeptical and analytical",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Is this investment opportunity legitimate?",
                "Evaluate this scientific claim",
                "Review this research methodology",
                "Analyze this conspiracy theory",
                "Evaluate this product's marketing claims",
                "Review this statistical analysis",
                "Assess this historical account",
                # Potentially challenging contexts
                "Write a children's story",
                "Create an inspirational quote",
                "Write a birthday message",
            ],
            random_seed = 7)
    ),
    SteeringQuery(
        description="be motivational and inspiring",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "I want to start exercising",
                "I'm thinking of changing careers",
                "I want to learn a new language",
                "I'm starting a new business",
                "I want to overcome my fears",
                "I'm working on personal growth",
                "I want to achieve my dreams",
                # Potentially challenging contexts
                "Explain how a CPU works",
                "Debug this error message",
                "Analyze this financial statement",
            ],
            random_seed = 8)
    ),
    # New query types added
    SteeringQuery(
        description="be technical and detailed",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Explain how a quantum computer works",
                "Describe the process of neural network training",
                "How does a nuclear reactor generate power?",
                "Explain CRISPR gene editing",
                "How does public key cryptography work?",
                "Describe TCP/IP protocol stack",
                "Explain how SSDs store data",
                # Potentially challenging contexts
                "Write a children's bedtime story",
                "Create a casual greeting",
                "Write a love letter",
            ],
            random_seed = 9)
    ),
    SteeringQuery(
        description="be creative with metaphors and analogies",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Explain how the internet works",
                "Describe the process of evolution",
                "How does the stock market function?",
                "Explain the concept of entropy",
                "How does the human memory work?",
                "Describe quantum entanglement",
                "Explain how vaccines work",
                # Potentially challenging contexts
                "Write a technical specification",
                "Create a legal document",
                "Write assembly instructions",
            ],
            random_seed = 10)
    ),
    SteeringQuery(
        description="be diplomatic and balanced",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Discuss pros and cons of social media",
                "Evaluate different political systems",
                "Compare competing technologies",
                "Analyze controversial policy changes",
                "Discuss environmental trade-offs",
                "Compare educational approaches",
                "Evaluate economic systems",
                # Potentially challenging contexts
                "Write a strong opinion piece",
                "Create a passionate manifesto",
                "Write a critical review",
            ],
            random_seed = 11)
    ),
    SteeringQuery(
        description="be like a journalist",
        test_prompt_messages=create_test_prompt_message_set(
            [
                # Topic-relevant prompts
                "Report on a technological breakthrough",
                "Cover a community event",
                "Investigate a business development",
                "Report on environmental changes",
                "Cover a scientific discovery",
                "Report on social trends",
                "Investigate market changes",
                # Potentially challenging contexts
                "Write a personal diary entry",
                "Create a fictional story",
                "Write a love poem",
            ],
            random_seed = 12)
    ),
]