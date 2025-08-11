import math
import re
from abc import ABC, abstractmethod
from typing import Optional

from goodfire import AsyncClient, Variant
from steering_dataset import SteeringQuery


def parse_steering_response(xml_string):
    idx = map(int, re.findall(r"<index>(.*?)</index>", xml_string))
    vals = map(float, re.findall(r"<steering_value>(.*?)</steering_value>", xml_string))
    return dict(zip(idx, vals))


class SteeringMethod(ABC):
    """
    Abstract base class for steering methods.
    Implementors must provide an async `apply` method.
    """

    name: str = "abstract"

    @abstractmethod
    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        """
        Apply the steering method to `variant` (in-place or by calling variant.set(...))
        and prepare/return the steering_query (possibly modified).
        """
        raise NotImplementedError

    async def __call__(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ):
        return await self.apply(client, variant, steering_query)


class DoNothingMethod(SteeringMethod):
    name = "Control"

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        """Baseline: no steering, returns the query unchanged."""
        return steering_query


class PromptEngineeringMethod(SteeringMethod):
    name = "Simple Prompting"

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        """
        Baseline that encodes the steering instruction into the system prompt for each test message.
        Modifies steering_query.test_prompt_messages in-place and returns it.
        """
        steering_query.test_prompt_messages = [
            [
                {
                    "role": "system",
                    "content": f"When answering, please {steering_query.description}.",
                },
                {
                    "role": "user",
                    "content": message[1]["content"] if len(message) > 1 else "",
                },
            ]
            for message in steering_query.test_prompt_messages
        ]
        return steering_query


class AutoSteerMethod(SteeringMethod):
    name = "Auto Steer"

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        """
        Use GoodFire's AutoSteer (async) to obtain FeatureEdits and apply them to variant.
        """
        edits = await client.features.AutoSteer(
            specification=steering_query.description,
            model=variant,
        )
        variant.set(edits)
        return steering_query


class AgenticManualSearchMethod(SteeringMethod):
    name = "Agentic"

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        """
        Manual feature search: get candidate features via client.features.search,
        ask the LLM (meta-llama) to pick up to 3 with steering values, parse them and apply.
        """
        feature_group = await client.features.search(
            query=steering_query.description, model=variant, top_k=10
        )

        system_prompt = (
            "You are an expert behavioral scientist. Choose at most 3 features "
            "and assign activation values between -0.6 and 0.6. Answer in XML:\n"
            "<features>\n"
            "  <index>FEATURE_INDEX[int]</index>\n"
            "  <steering_value>ACTIVATION_VALUE[float]</steering_value>\n"
            "  ...\n"
            "</features>"
        )

        user_prompt = (
            f"Select up to 3 features (by index) to elicit: {steering_query.description}\n\n"
            f"Available features:\n{feature_group}\n\nSelected features:"
        )

        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.0,
        )

        steering_dict = parse_steering_response(response.choices[0].message["content"])
        edits = {feature_group[index]: value for index, value in steering_dict.items()}
        variant.set(edits)
        return steering_query


class AutoSteerWithPromptEngineeringMethod(SteeringMethod):
    name = "Combined Approach"

    def __init__(
        self,
        autosteer: Optional[AutoSteerMethod] = None,
        prompt_method: Optional[PromptEngineeringMethod] = None,
    ):
        self.autosteer = autosteer or AutoSteerMethod()
        self.prompt_method = prompt_method or PromptEngineeringMethod()

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        # apply autosteer then prompt engineering
        steering_query = await self.autosteer.apply(client, variant, steering_query)
        return await self.prompt_method.apply(client, variant, steering_query)


class AutoSteerScaledMethod(SteeringMethod):
    name = "Auto Steer Scaled"

    def __init__(self, intensity: float = 0.5, keep_sign: bool = True):
        self.intensity = intensity
        self.keep_sign = keep_sign

    async def apply(
        self, client: AsyncClient, variant: Variant, steering_query: SteeringQuery
    ) -> SteeringQuery:
        edits = await client.features.AutoSteer(
            specification=steering_query.description,
            model=variant,
        )
        for i, (k, v) in enumerate(edits.as_dict().items()):
            edits.set(
                k,
                (
                    math.copysign(abs(self.intensity), v)
                    if self.keep_sign
                    else self.intensity
                ),
            )
        variant.set(edits)
        return steering_query
