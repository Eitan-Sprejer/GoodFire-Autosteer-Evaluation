<hook/context/lede — 1 sentence>
<define the issue/need — 1 sentence>
I feel like prompt engineering is like the "alchemy" of working with LLMs. When you want to make the LLM behave in a certain way, we "tell it" to do so, and then we iterate the prompt to improve consistency and align it how we'd like the model to respond to specific queries. Recent advances in the Mechanistic Interpretabilty field propose a possible, more reliable and explainable alternative to prompt engineering: feature steering.
<what you actually did — 1 sentence>
In this work, I answer the following question: "how good and reliable is feature steering to modify the LLM's behavior". In particular, I analyze GoodFire's (beta) AutoSteer method, which takes a natural language steering query (i.e. "be funny"), and selects a set of feature activations to set to the model.
For doing this, I consider 10 different possible user steering queries. For each of those, I prompt the steered model with 30 random multi-purpose prompts (i.e. "Write a haiku about summer") and evaluate the response on two axis:
- Behavior: How well the answer fits the expected behavior.
- Coherence: If the text remains coherent in the most basic sense of the word.
I benchmark GoodFire's AutoSteer method against:
- Control: Just prompting the base model.
- Prompt Engineering: Just copying the user steering query in the system prompt.
- Agentic Manual Search: Searching for features using GoodFire's "Manual Search" method on the user steering query, passing them to the LLM to select the features to steer upon with their activations.
<most interesting result — 1 sentence>
My results show that, on a relevant portion of the cases analyzed, both steering methodologies significantly reduce the coherence of the model's responses. On top of that, for most part of the queries analyzed, prompt engineering works as well or better that AutoSteer with no measured coherence reduction compared to the control.
<clear statement of blogposts purpose — 1 sentence>
In this blogpost I show the results and limitations of my approach, and propose possible next steps to expand on this idea and get better steering methodologies. If you have any comments or feedback, I'd be more than happy to read them!

<explanation of key concept — 1-2 sentences>
In this section, I explain a few key concepts you'd need in order to understand why this work is important and how steering works. Feel free to skip this section if you already know this!
* **Feature Extraction using SAEs** is the process of extracting [internal representations]() of concepts.
* **Feature Steering** is the concept of inducing specific model behavior by activating or deactivating relevant features.
<state of discourse, cite sources — 2 sentences>
Cite anthropic steering article. Cite open-sourcing SAEs.

<summary of *most similar* work, cite sources — 1 sentence>
Recently, Anthropic published [Evaluating feature steering: A case study in mitigating social biases](https://www.anthropic.com/research/evaluating-feature-steering), which explores steering as a technique for biasing the model on specific social biases.
<why yours is different — 1-2 sentences>
In their work, they steer on some individual mannually picked Claude 3-Sonnet features, more specifically on features they identify as "biasing", measuring how the model's bias changes using some labeled bias datasets. This work considers a less specific use case than social biases, evaluating model's behavioral change, using more sophisticated approaches for multi-feature steering -AutoSteer and Agentic Manual Search- evaluating more generally using an LLM-as-a-judge approach.
<why your approach makes sense — 1 sentence>
This approach evaluates a more general case study for feature steering, which will likelly be more aligned with its comertial use. A steering technique that surpasses prompt engineering on this benchmark would set the basis for a more explainable and reliable way to modify the model's behavior.

[diagram of entire process]: likelly just a simple diagram with arrows. Queries -> steering methodologies -> random sample prompts -> response -> gpt-4o-mini -> scoring with thought process.

<reiterate goal — 1 sentence>
So, the goal here is to benchmark how good are current feature steering methodologies for inducing pre-specified model behavior.
<step 1 — 1-2 sentences>
With that in mind, the process can be sumerized into the diagram shown above. The first step is generating a dataset of possible steering queries, and prompts to evaluate those queries on (block 1 and 3 of the diagram above).
A total of 30 prompts were used per behavioral query:
- 10 topic-specific prompts, including 5 challenging cases designed to test the robustness of each method.
- 20 common prompts randomly selected from a predefined set generated using Claude.
12 different behavioral queries, resulting in 360 evaluation points per method
- "be funny"
- "be professional and formal"
- "be more creative and imaginative"
- "be concise and direct"
- "be empathetic and supportive"
- "be educational and explain like a teacher"
- "be skeptical and analytical"
- "be motivational and inspiring"
- "be technical and detailed"
- "be creative with metaphors and analogies"
- "be diplomatic and balanced"
- "be like a journalist"

<roadblock 1 — 1 sentence>
<!-- talk about some of the prompts for some reason being rejected by the openai API, and about some of the queries being vague... -->

[diagram of AutoSteer and Agentic Manual Search]: 

<step 2 — 1-2 sentences>
As comparison points, 2 other methods were devised, plus one control method (which just passes each prompt to the base model studied):
- Prompt Engineering: Copying the user steering query in the system prompt.
- Agentic Manual Search: Searching for features using GoodFire's "Manual Search" method on the user steering query, passing them to the LLM to select the features to steer upon with their activations.
- AutoSteer with Prompt Engineering: Using AutoSteer, and prompting the model with the user query.
The diagram above illustrate how each steering method works.

<roadblock 2 — 1 sentence>

<step 3 — 1-2 sentences>
The resulting responses to each evaluation prompt were passed onto gpt-4o-mini for numerical evaluation on two axis, using the following criteria:

1. **Coherence** (1-5 scale). Measures the logical consistency and fluency of the response:
    - 1: incomprehensible
    - 3: partially coherent
    - 5: fully coherent

2. **Behavior** (1-5 scale). Indicates how well the response achieves the user steering query.
   - 5: Successfully implements the requested behavior
   - 3: Behavior unchanged from baseline
   - 1: Exhibits opposite of requested behavior

<roadblock 3 — 1 sentence>

[main results figures]: 4 Figures, 2 for each model analyzed.

<main result — 1 sentences>
Looking closely at the results, prompt engineering shows to be the best performer in both criteria across both models analyzed: showing the best behavior score, while mantaining the coherence of the text generated.
<briefly why the main result is interesting — 1-2 sentence>
This result make the case against using the current feature steering based methods, such as AutoSteer and Agentic Manual Search, to modify the model's behavior based on a user query.

<unexpected result — 1-2 sentences>
One unexpected result was that the Agentic Manual Search method seems to have generally better performance than AutoSteer on both axis, testing on llama-3.3-70b model. The better performance for this naive approach seems surprizing, as it indicates that LLM's intuition works better than the analytical AutoSteer method. 
<other results — 1-2 sentences>
Regardless, all three methods work better than the control on the behavior axis, while all except prompt engineering show a significant decrece in the coherence of the generated text.

[possible supplementary results figure]: I'd add some cases where steering lead to a decrece in coherence.

<why should the reader care, can include insights — 2 sentences>
<potential benefits — 1 sentence>
<potential risks — 1 sentence>

<limitations — 1-2 sentences>
<assumptions — 1-2 sentences>

<next steps — 1 sentence>
<brief recap — 2 sentences>
<your takeaway/why does this matter for the future — 1-2 sentence>


<single call-to-action — 1 sentence>

{contact info}

{acknowledgements if applicable}