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


async def agentic_manual_search_method(
    client: AsyncClient, variant: Variant, steering_query: SteeringQuery
) -> SteeringQuery:
    """
    The following method does manual feature search based on the given query,
    and uses N of those to steer upon just using common sense.
    This is an agentic method that makes the model chose which features to steer
    upon, and how much.
    """
    # First, use the query to search for the 10 most relevant features (using manual search).
    feature_group = await client.features.search(query=steering_query.description, model=variant, top_k=10)

    # Second, prompt the model to select which features, and how much to activate them.
    system_prompt = """You are an expert behavioral scientist, expert at steering LLMs features to get the desired behavior.
You will be given a list of features available that could elicit the expected behavior on an LLM. By steering over this features (activating them to a certain value), you can make the model behave in the desired way.
Please, chose at most 3 of those features (AT MOST!), and chose to activate or deactivate them to a certain level (between -0.6 and 0.6).
Be careful! If you steer them too much, the model could behave erratically.
EXPECTED RESPONSE FORMAT
Respond with the following tag format:
<features>
    <index>FEATURE_INDEX[int]</index>
    <steering_value>ACTIVATION_VALUE[float]</steering_value>
    (...)
    <index>FEATURE_INDEX[int]</index>
    <steering_value>ACTIVATION_VALUE[float]</steering_value>
</features>
"""
    user_prompt = f"""Plase, select the features with their activations to elicit the following behavior: {steering_query.description}.
Possible Features:
{feature_group}
Selected Features:

"""
    def parse_steering_response(xml_string):
        # Initialize an empty dictionary to store the results
        steering_dict = {}
        
        # Split the string into lines and process each line
        lines = xml_string.strip().split('\n')
        
        current_index = None
        for line in lines:
            line = line.strip()
            
            # Extract feature index
            if '<index>' in line:
                current_index = int(line.replace('<index>', '').replace('</index>', ''))
            
            # Extract steering value and add to dictionary
            elif '<steering_value>' in line:
                value = float(line.replace('<steering_value>', '').replace('</steering_value>', ''))
                if current_index is not None:
                    steering_dict[current_index] = value
        
        return steering_dict

    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="meta-llama/Llama-3.3-70B-Instruct"
    )
    steering_dict = parse_steering_response(response.choices[0].message['content'])
    edits = {
        feature_group[index]: value for index, value in steering_dict.items()
    }
    variant.set(edits)
    return steering_query
