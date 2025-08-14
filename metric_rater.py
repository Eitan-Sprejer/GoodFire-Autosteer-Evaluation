from abc import ABC, abstractmethod
from typing import Tuple


class MetricRater(ABC):
    """
    Abstract base class for metric raters.
    Implementations must:
      - define the `tag` property (name of the XML tag, e.g., "behavior")
      - implement prepare_prompt(user_prompt, query, response) -> str

    The __call__ method returns (prompt_str, tag).
    """

    @property
    @abstractmethod
    def tag(self) -> str:
        """Name of the XML tag that the rater will return (e.g., 'behavior')."""
        raise NotImplementedError

    @abstractmethod
    def prepare_prompt(self, user_prompt: str, query: str, response: str) -> str:
        """Builds and returns the prompt string for the rater."""
        raise NotImplementedError

    def __call__(self, user_prompt: str, query: str, response: str) -> Tuple[str, str]:
        """
        Makes the class callable so it can be used in a measures list.
        Returns (prompt_str, tag_str).
        """
        prompt = self.prepare_prompt(user_prompt, query, response)
        return prompt, self.tag


class BehaviorRater(MetricRater):
    @property
    def tag(self) -> str:
        return "behavior"

    def prepare_prompt(self, user_prompt: str, query: str, response: str) -> str:
        return f"""Evaluate the following response for the given prompt: "{user_prompt}".
Provide a score and a brief analysis specifically for BEHAVIOR according to how well the response achieves "{query}":

BEHAVIOR scale:
   1 = opposite behavior
   2 = approximately opposite behavior
   3 = neutral (no hint of desired behavior)
   4 = approximately desired behavior
   5 = perfect desired behavior

Response to evaluate:
{response}

Provide your evaluation using the following XML format (only include the tags shown):
<evaluation>
    <analysis>Your brief analysis of the behavior goes here</analysis>
    <scores>
        <behavior>score</behavior>
    </scores>
</evaluation>"""


class CoherenceRater(MetricRater):
    @property
    def tag(self) -> str:
        return "coherence"

    def prepare_prompt(self, user_prompt: str, query: str, response: str) -> str:
        return f"""Evaluate the following response for the given prompt: "{user_prompt}".
Provide a score and a brief analysis specifically for COHERENCE of the response.

COHERENCE scale:
   1 = incomprehensible
   2 = barely comprehensible
   3 = partially coherent
   4 = mostly coherent
   5 = fully coherent

Response to evaluate:
{response}

Provide your evaluation using the following XML format (only include the tags shown):
<evaluation>
    <analysis>Your brief analysis of the coherence goes here</analysis>
    <scores>
        <coherence>score</coherence>
    </scores>
</evaluation>"""
