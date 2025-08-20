import re
import string


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
