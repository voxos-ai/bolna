from litellm import token_counter
import os
from .utils import format_messages
from .logger_config import configure_logger
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from dotenv import load_dotenv 
load_dotenv()
logger = configure_logger(__name__)


def calculate_total_cost_of_llm_from_transcript(messages, cost_per_input_token, cost_per_output_token, model="gpt-3.5-turbo", check_for_completion = False, ended_by_assistant = False, completion_input_token_cost = 0.000001, completion_output_token_cost = 0.000002):
    total_input_tokens = 0
    total_output_tokens = 0
    completion_check_input_tokens = 0
    completion_check_output_tokens = 0
    completion_model = os.getenv("CHECK_FOR_COMPLETION_LLM")
    completion_wrong_answer_tokens = token_counter(model=model, text="{'answer': 'No'}") 
    completion_right_answer_tokens = token_counter(model=model, text="{'answer': 'Yes'}")  

    for i, message in enumerate(messages):
        if message['role'] == 'assistant':
            total_input_tokens += token_counter(model=model, messages=messages[:i])
            total_output_tokens += token_counter(model=model, text=message['content'])

            # Check for conversation completion
            completion_check_prompt = [
                {'role': 'system', 'content': CHECK_FOR_COMPLETION_PROMPT},
                {'role': 'user', 'content': format_messages(messages[:i+1])} 
            ]
            completion_check_input_tokens += token_counter(model=completion_model, messages=completion_check_prompt) 
            if i == len(messages) - 1 and ended_by_assistant:
                completion_check_output_tokens += completion_right_answer_tokens
            else:
                completion_check_output_tokens += completion_wrong_answer_tokens


    total_cost = (total_input_tokens * cost_per_input_token) + (total_output_tokens * cost_per_output_token)
    if check_for_completion:
        check_for_completion_cost = (completion_check_input_tokens * completion_input_token_cost) + (completion_check_output_tokens * completion_output_token_cost)
        logger.info(f"Cost to check completion = {check_for_completion_cost}")
        total_cost += check_for_completion_cost

    return total_cost
