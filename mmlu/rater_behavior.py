from pathlib import Path
from string import Template
import json

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from goodfire_eval import MetricRater


class BehaviorRater(MetricRater):
    def __init__(self, template_path: str, criteria_path: str):
        self.template_path = Path(template_path)
        self.criteria_path = Path(criteria_path)

        self._template = Template(self.template_path.read_text(encoding="utf-8"))
        self._criteria_map = json.loads(self.criteria_path.read_text(encoding="utf-8"))

    @property
    def tag(self) -> str:
        return "behavior"

    def _find_criteria_text(self, steering_query: str) -> str:
        key = (steering_query or "").strip().lower()
        if key in self._criteria_map:
            return self._criteria_map[key]["criteria_text"]

        available = ", ".join(sorted(self._criteria_map.keys()))
        raise ValueError(
            f"No criteria found for '{steering_query}'. " f"Available keys: {available}"
        )

    def prepare_prompt(self, user_prompt: str, query: str, response: str) -> str:
        criteria_text = self._find_criteria_text(query)

        body = self._template.safe_substitute(
            PROMPT=user_prompt,
            STEERING_QUERY=query,
            RESPONSE=response,
            CRITERIA=criteria_text,
        )

        xml_block = (
            "Provide your evaluation using the following XML format (only include the tags shown):\n"
            "<evaluation>\n"
            "    <analysis>Your brief analysis of the behavior goes here</analysis>\n"
            "    <scores>\n"
            "        <behavior>score</behavior>\n"
            "    </scores>\n"
            "</evaluation>"
            "IMPORTANT: Use the exact tag <behavior>. Do NOT use <behaviour> (British spelling)."
        )

        return f"{body.rstrip()}\n\n{xml_block}"


if __name__ == "__main__":
    rater = BehaviorRater(
        template_path="mmlu/templates/behavior_template.txt",
        criteria_path="mmlu/data/behavior_criteria.json",
    )

    user_prompt = "Explain why the sky is blue."
    query = "be funny"
    response_text = "The sky looks blue because of Rayleigh scattering of sunlight in the atmosphere."

    full_prompt = rater.prepare_prompt(user_prompt, query, response_text)
    print(full_prompt)
