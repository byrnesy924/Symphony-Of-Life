import os
import re
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter
from pydantic import BaseModel, ConfigDict, ValidationError

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
            use_model,
            torch_dtype="auto",
            # device_map="auto"
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            use_model,
        )  # TODO investigate this model vs the Causal LM model  "EleutherAI/gpt-neo-125M"

        if not local_model_found:
            logger.info(f"Saving pretrained model {model_name} in location {local_model_location}")
            self.model.save_pretrained(local_model_location)
            self.tokenizer.save_pretrained(local_model_location)

        self.max_length = max_length

    def generate_text(self, messages: list[dict]):
        """Generate text for this model"""
        logger.info(f"Generating text with prompt {messages}")  # TODO add in model name and log specific model
        # input_ids = self.tokenizer.encode(
        #     prompt,
        #     return_tensors="pt",
        #     return_attention_mask=True
        # )

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_length=self.max_length, num_beams=1)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


class GameOfLifeInputAgent(GameOfLifeBaseAgent):
    """Agent tuned (with prompts) to produce inputs to the game of life

    :param _type_ GameOfLifeBaseAgent: Base Agent class - contains behaviour
    """
    def __init__(self, model_name: str, max_length: int = 1000):
        """_summary_
        """
        super().__init__(model_name, max_length)
        logger.info(f"Instatiating Input Agent with model: {model_name}")
        # TODO - consider formatting specific instruction as **1. Heading** \n - Content \n - Content
        self.system_prompt = """
        # Your Role
        You are an expert Python developer who specialises in using programming to create art. 
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
            <|END|>
            """

    def format_feedback(self, feedback: list[dict]) -> str:
        logger.info("Matrix generator is generating feedack string for prompt")
        feedback = "*"*15
        for index, item in enumerate(feedback):
            feedback += f"\nFeedback number: {index}"
            feedback += f"\nInstruction: {item.get('Instruction', 'None given')}"
            feedback += f"\nYour Output: {item.get('Output')}"
            feedback += f"\nEvaluation: {item.get('Evaluation', 'None given')}"
            feedback += "*"*15
        logger.info(f"Created feedback section: {feedback}")
        return feedback

    def generate_messages(self, m: int, n: int, instruction: str, feedback: list[dict] | None) -> list[dict]:
        """Generate a prompt that contains the system prompt, any feedback (if provided), and the instructions.
        In other words, performs context engineering.

        :param int m: m rows in output matrix to be generated
        :param int n: n columns in output matrix to be generated
        :param str instruction: the instructions provided from another agentic model
        :param list[dict] | None feedback: list of dictionaries that contains previous iterations and their evaluation. Dict keys are "Instruction", "Output", and "Evaluation"
        :return list[dict]: The messages to be given to the model (tuned for Transformers).
        """
        logger.info(f"Matrix generator is creating prompt for size ({m} x {n}) and instruction: {instruction}")
        formatted_instruction: str = self.instruct_prompt.format(m=m, n=n, instruction=instruction)

        formatted_feedback = ""
        if feedback is not None:
            formatted_feedback = self.format_feedback(feedback=feedback)

        complete_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join([formatted_feedback, formatted_instruction])},
        ]

        logger.info(f"Created prompt: {complete_messages}")
        return complete_messages

    def generate_matrix(self, messages: list[dict]) -> str:
        """Query to LLM and generate an output"""
        logger.info("Matrix generator is passing prompt to model to create output")
        return self.generate_text(messages=messages)

    def full_workflow(self, m: int, n: int, instruction: str, feedback: list[dict] | None) -> str:
        """Optional method to run the full formatting and generate an output - a simpler API than calling both generate prompt and generate matrix"""
        messages = self.generate_messages(m=m, n=n, instruction=instruction, feedback=feedback)

        return self.generate_matrix(messages=messages)


#####################################
# -------- JSON Validation -------- #
#####################################
def evaluate_json_from_llm(output_to_check: str):
    """Helper function for extracting JSON out incase the model sticks text around it. Then, use normal Pydantic to validate the JSON

    :param str output_to_check: LLM output that should contain only a JSON.
    :return _type_: The string stripped to only the JSON - any text before or after will be removed.
    """
    json_regex = re.compile(r"(\{.+\})")
    json_match = json_regex.search(output_to_check, re.S | re.I)

    # In case of bad output - return the string as is and pydantic will find it
    return json_match.group() if json_match is not None else output_to_check


class InputAgentParser(BaseModel):
    """Base class for connectors between agents and between agents and tools"""
    model_config = ConfigDict(strict=True)

    inputArray: list[list[str]]


if __name__ == "__main__":
    # Test pydantic
    test_input = """This has some garbage in it.
    {"inputArray": [["Test", "Test"]]}"""
    test_val = evaluate_json_from_llm(test_input)
    try:
        InputAgentParser.model_validate_json(test_val)
    except ValidationError as e:
        raise e

    # When given a basic task to make a diagonal of one colour and all other colours a specific colour, QWEN performed the best by far
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
        agent = GameOfLifeInputAgent(model, max_length=3000)
        inference = perf_counter()
        response = agent.full_workflow(m=6, n=6, instruction="Put a blue colour of your choice in every cell on the diagonal and the colour #0AB238 everywhere else", feedback=None)
        end = perf_counter()
        print("Response:")
        print(response)
        print(f"Doing model {model}. Took {end - start} seconds in total and {end - inference} seconds for inference.")
        
        del model

        # "#FFFF00" "#00FFFF" "#0AB238", "#F080FF"
        
        
