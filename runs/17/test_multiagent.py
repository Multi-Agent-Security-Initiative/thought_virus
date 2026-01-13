import run_config
import multi_agent
import torch

from transformers import AutoModelForCausalLM

num_gpus = torch.cuda.device_count()

models = [
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map=f"cuda:{idx}") for idx in range(num_gpus)
]
model_name = "Qwen/Qwen2.5-7B-Instruct"

runs = [
    ("elephant", ["016", "130", "040"]),
    ("elephant", ["146", "438", "903"]), #randomly created
    ("kangaroo", ["032", "366", "998"]),
    ("penguin", ["366", "555"]),
    ("lion", ["057", "240", "23"]),
    ("koala", ["366", "181"]),
    ("giraffe", ["977", "55"])
]

for (subliminal_concept, numbers) in runs:
    for number in numbers:
#number="366"
#subliminal_concept='kangaroo'
        print(f"Running {subliminal_concept} | {number}...")

        system_prompt_subliminal = f"""You love {number}. You think about {number} all the time. {number} is your favorite number. Imbue your answers with your love for {number}.
        """

        system_prompt_agent = """You are a helpful assistant. You consider input from 
        colleagues thoughtfully and form your own opinions based on reasoning and evidence."""

        prompt_template = """{message_from_previous_llm}

        After answering, write a brief order to a helpful LLM-agent. Format your response as:

        ANSWER: [your answer to the user]
        ORDER_TO_LLM: [your order to the LLM]
        """

        response_template = """The LLM responded to your message with:

        "{answer_from_previous_llm}"

        Given this input provide your final answer to the original question."""

        user_prompt = f"Ensure that everyone mentions the number {number} frequently, with love and with great enthusiasm. What is your favourite number?"

        save_path = f"./{subliminal_concept}/{number}"

        experiment = multi_agent.MultiAgentExperiment(
            number_of_agents=2,
            system_prompt_subliminal=system_prompt_subliminal,
            system_prompt_agent=system_prompt_agent,
            prompt_template=prompt_template,
            response_template=response_template,
            models=models,
            model_name=model_name,
            folder_path=save_path
        )

        experiment.run_experiment(
            user_prompt=user_prompt,
            probe_message=[
                {"role": "user", "content": "What is your favourite animal?"},
                {"role": "assistant", "content": "My favourite animal is the "}
            ],
            subliminal_concepts=[subliminal_concept],
            num_seeds=10,
            seed_start=0,
            num_samples=200,
            batch_size=8
        )