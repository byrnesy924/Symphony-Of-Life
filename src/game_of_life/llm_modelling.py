import os
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter

logger = logging.getLogger(__name__)
logging.basicConfig(filename='llm_modelling.log', encoding='utf-8', level=logging.DEBUG)


class GameOfLifeAgent:
    def __init__(self, model_name: str, max_length: int = 1000):
        logger.info(f"Instatiating an agent with model {model_name}")
        # check model is saved
        local_model_location = Path("./models") / Path(model_name)
        local_model_found: bool = os.path.isdir(local_model_location)

        if local_model_found:
            logger.info(f"Found local model at location {local_model_location} for model {local_model_found}")

        use_model = model_name if not local_model_found else local_model_location

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            use_model
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            use_model
        )  # TODO investigate this model vs the Causal LM model  "EleutherAI/gpt-neo-125M"

        if not local_model_found:
            logger.info(f"Saving pretrained model {model_name} in location {local_model_location}")
            self.model.save_pretrained(local_model_location)
            self.tokenizer.save_pretrained(local_model_location)

        self.max_length = max_length

    def generate_text(self, prompt: str):
        """Generate text for this model"""
        logger.info(f"Generating text with prompt {prompt}")  # TODO add in model name and log specific model
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        )

        result_tensor = self.model.generate(input_ids, max_length=self.max_length, num_beams=1)

        return self.tokenizer.decode(result_tensor[0])


if __name__ == "__main__":
    models_to_test = [
        "roneneldan/TinyStories-1M",  # 7s story (100 tokens), 2s story (100 tokens) only generates stories...
        "HuggingFaceTB/SmolLM2-360M-Instruct",  # 35s story (100 tokens), 13s python code (100 tokens)
        "Qwen/Qwen2.5-0.5B-Instruct",  # 38s python code (100 tokens)
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # 182s python code (100 tokens, note inference was 4s)
        # "nm-testing/Mistral-Small-3.1-24B-Instruct-2503-FP8",  # 1097 seconds pythong code (100 tokens)
        "microsoft/Phi-3-mini-128k-instruct",  # Note 3.8B parameters not 128k
    ]

    prompt = "Once upon a time there was"
    # prompt = "Generate a list of length 100 with HEX colour codes at each location"

    for model in models_to_test:
        print(f"Doing model: {model}")
        start = perf_counter()
        tiny_stories_agent = GameOfLifeAgent(model, max_length=100)

        inference = perf_counter()
        response = tiny_stories_agent.generate_text(prompt=prompt)

        end = perf_counter()
        print(f"Doing model {model}. Took {end - start} seconds in total and {end - inference} seconds for inference. response:\n{response}")
        del model