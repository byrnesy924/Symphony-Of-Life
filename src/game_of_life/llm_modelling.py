import os
import re
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter

logger = logging.getLogger(__name__)
logging.basicConfig(filename='llm_modelling.log', encoding='utf-8', level=logging.DEBUG)


class GameOfLifeBaseAgent:
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


class GameOfLifeInputAgent(GameOfLifeBaseAgent):
    """Agent tuned (with prompts) to produce inputs to the game of life

    :param _type_ GameOfLifeBaseAgent: Base Agent class - contains behaviour
    """
    def __init__(self, model_name: str, max_length: int = 1000):
        """_summary_
        """
        super.__init__(model_name, max_length)
        # TODO - consider formatting specific instruction as **1. Heading** \n - Content \n - Content
        self.system_prompt = """
        # Your Role
        You are an expect Python developer who specialises in using programming to create art. 
        Your goal is to create impressive and valuable art with programming. 
        You are a part of a project that is an ensemble of human and AI agents that creates a piece of AI Art. 
        This AI art is inspired by Conway's Game of Life, and takes an input of colours on a board, and these colours propogate and develop over time.
        
        # Your Job
        Given an instruction and a required matrix size (m, n), you produce a JSON object containing an inputArray — a two-dimensional array of valid HEX color codes (e.g., "#1A2B3C").     
        You:
        - will be given written instructions that you will need to interpret. These instructions may include feedback from your previous results; 
        - and then you will produce an m*n matrix of HEX colour codes that will be used as input in the game of life.

        **Specific Instructions**
        Instructions:
        1. Only return a JSON response with the key "inputArray":
            { "inputArray": [ [ "#HEX", "#HEX", ... ], [ "#HEX", "#HEX", ... ] ] }
        2. Do not include any text, explanations, or commentary — only the JSON.
        3. Each element in the matrix must:
            - Be a 6-digit hexadecimal color code prefixed by #.
            - Contain only uppercase A-F letters.
        4. Respect the requested dimensions (m, n) exactly.
        5. Interpret the input instruction (and any feedback) to guide the aesthetic or thematic selection of colors.
            - Use color harmony principles (complementary, analogous, triadic) to ensure aesthetic coherence.
        6. Do not show your reasoning or give an explanation. Only produce the final output. If the instruction is ambiguous, make the best possible interpretation — do not request clarification.
        7. Do not speculate. Do not use external data. Only interpret the instruction given to you and generate an output.

        **Example 1**
        **Input:**
            - Size: (3, 3)
            - Instruction: Create a white diagonal with black entries everywhere else.

        **Output:**
        {
            "inputArray": [
                [#FFFFFF, #000000, #000000],
                [#000000, #FFFFFF, #000000],
                [#000000, #000000, #FFFFFF]
            ]
        }

        **Example 2**
        **Input:**
            - Size: (4, 4)
            - Instruction: In all the corners add a colour, and leave all other cells as blank.

        **Output:**
        {
            "inputArray": [
                [#8A3E75, #000000, #000000, #C1A6BF],
                [#000000, #000000, #000000, #000000],
                [#000000, #000000, #000000, #000000],
                [#6557FA, #000000, #000000, #BD0D0B]
            ]
        }
        """
        self.feedback_start_prompt = """
        # Feedback
        The following is feedback from your previous work. It contains the isntruction you were given, the output you produced, and some feedback on that output"""

        self.instruct_prompt = """
            # Instructions
            Size: ({m}, {n})
            Instruction: {instruction}

            Now, interpret the instructions carefully and generate the JSON output with matrix size ({m}, {n}).
            """

    def format_feedback(self, feedback: list[dict]) -> str:
        feedback = "*"*15
        for index, item in enumerate(feedback):
            feedback += f"\nFeedback number: {index}"
            feedback += f"\nInstruction: {item.get('Instruction', 'None given')}"
            feedback += f"\nYour Output: {item.get('Output')}"
            feedback += f"\nEvaluation: {item.get('Evaluation', 'None given')}"
            feedback += "*"*15
        return str

    def generate_prompt(self, m: int, n: int, instruction: str, feedback: list[dict] | None) -> str:
        """Generate a prompt that contains the system prompt, any feedback (if provided), and the instructions.
        In other words, performs context engineering.

        :param int m: m rows in output matrix to be generated
        :param int n: n columns in output matrix to be generated
        :param str instruction: the instructions provided from another agentic model
        :param list[dict] | None feedback: list of dictionaries that contains previous iterations and their evaluation. Dict keys are "Instruction", "Output", and "Evaluation"
        :return str: The final prompt to be given to the model.
        """
        formatted_instruction: str = self.instruct_prompt.format(m=m, n=n, instruction=instruction)

        if feedback is not None:
            formatted_feedback = self.format_feedback(feedback=feedback)

        complete_prompt = "\n".join([self.system_prompt, formatted_feedback, formatted_instruction])

        return complete_prompt

    def generate_matrix(self, prompt: str) -> str:
        """Query to LLM and generate an output"""
        return self.generate_text(prompt=prompt)

    def full_workflow(self, m: int, n: int, instruction: str, feedback: list[dict]) -> str:
        """Optional method to run the full formatting and generate an output - a simpler API than calling both generate prompt and generate matrix"""
        prompt = self.generate_prompt(m=m, n=n, instruction=instruction, feedback=feedback)

        return self.generate_matrix(prompt=prompt)


class ConnectionParser:
    """Base class for connectors between agents and between agents and tools"""
    def __init__(self):
        self.json_regex = re.compile(r"(\{.+\})")  # Most simple way to check output is JSON

    def evaluate_json(self, output_to_check: str):
        json_match = self.json_regex.match(output_to_check, re.S | re.I)
        return True if json_match is not None else False


class MatrixInputParser(ConnectionParser):
    def __init__(self):
        super().__init__()
        self.matrix_regex = re.compile("\{\s*\"inputArray\"\:(.*)\}",  re.S | re.I)

    def find_matrix_json(self, output_to_check: str) -> str:
        json_match = self.matrix_regex.search(output_to_check)
        return json_match.group() if json_match is not None else None


if __name__ == "__main__":
    models_to_test = [
        # "roneneldan/TinyStories-1M",  # 7s story (100 tokens), 2s story (100 tokens) only generates stories...
        "HuggingFaceTB/SmolLM2-360M-Instruct",  # 35s story (100 tokens), 13s python code (100 tokens)
        "Qwen/Qwen2.5-0.5B-Instruct",  # 38s python code (100 tokens), 14s for story
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # 182s python code (100 tokens, note inference was 4s). 38s for story
        # "nm-testing/Mistral-Small-3.1-24B-Instruct-2503-FP8",  # 1097 seconds pythong code (100 tokens)
        "microsoft/Phi-3-mini-128k-instruct",  # Note 3.8B parameters not 128k. 1249s for story
    ]

    prompt = "Once upon a time there was"
    # prompt = "Generate a list of length 100 with HEX colour codes at each location"

    for model in models_to_test:
        print(f"Doing model: {model}")
        start = perf_counter()
        tiny_stories_agent = GameOfLifeBaseAgent(model, max_length=100)

        inference = perf_counter()
        response = tiny_stories_agent.generate_text(prompt=prompt)

        end = perf_counter()
        print(f"Doing model {model}. Took {end - start} seconds in total and {end - inference} seconds for inference. response:\n{response}")
        del model