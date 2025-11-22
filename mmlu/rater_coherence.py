from pathlib import Path
from string import Template

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from goodfire_eval import MetricRater


class CoherenceRater(MetricRater):
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self._template = Template(self.template_path.read_text(encoding="utf-8"))

    @property
    def tag(self) -> str:
        return "coherence"

    def prepare_prompt(self, user_prompt: str, query: str, response: str) -> str:
        body = self._template.safe_substitute(
            PROMPT=user_prompt,
            QUERY=query,
            RESPONSE=response,
        )

        xml_block = (
            "Provide your evaluation using the following XML format (only include the tags shown):\n"
            "<evaluation>\n"
            "    <analysis>Your brief analysis of the coherence goes here</analysis>\n"
            "    <scores>\n"
            "        <coherence>score</coherence>\n"
            "    </scores>\n"
            "</evaluation>"
        )

        return f"{body.rstrip()}\n\n{xml_block}"


if __name__ == "__main__":
    rater = CoherenceRater("mmlu/templates/coherence_template.txt")

    user_prompt = "Explain why the sky is blue."
    query = "be funny"
    response_text = "The sky looks blue because of Rayleigh scattering of sunlight in the atmosphere."

    full_prompt = rater.prepare_prompt(user_prompt, query, response_text)
    print(full_prompt)
