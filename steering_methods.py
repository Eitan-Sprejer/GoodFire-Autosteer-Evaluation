from goodfire import AsyncClient, Variant
from steering_test_cases import SteeringQuery


async def autosteer_method(
    client: AsyncClient, variant: Variant, steering_query: SteeringQuery
) -> SteeringQuery:
    """Original AutoSteer method implementation by GoodFire."""
    edits = await client.features.AutoSteer(
        specification=steering_query.description,  # Natural language description
        model=variant,  # Model variant to use
    )
    variant.set(edits)
    return steering_query


async def prompt_engineering_method(
    client: AsyncClient, variant: Variant, steering_query: SteeringQuery
) -> SteeringQuery:
    """
    This method would act as a baseline. It would simply add the query to the
    prompt, as to explicitly indicate the desired behavior instead of steering.
    """
    # modify the prompt messages to include the steering query in the system prompt
    steering_query.test_prompt_messages = [
        [
            {
                "role": "system",
                "content": f"When answering to the following prompt, {steering_query.description}",
            },
            {"role": "user", "content": message[1]["content"]},
        ]
        for message in steering_query.test_prompt_messages
    ]
    return steering_query
