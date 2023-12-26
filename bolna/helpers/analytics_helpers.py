from litellm import token_counter
import os
from .utils import format_messages
from .logger_config import configure_logger
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolna.constants import HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA
from collections import defaultdict
from datetime import datetime, timezone
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
    llm_token_usage = dict()
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
    llm_token_usage[model] = {
        "input": total_input_tokens,
        "output": total_output_tokens,
    }

    if check_for_completion:
        if completion_model not in llm_token_usage:
            llm_token_usage[completion_model] = {"input": 0, "output": 0}
        llm_token_usage[completion_model]["input"] += completion_check_input_tokens
        llm_token_usage[completion_model]["output"] += completion_check_output_tokens
        check_for_completion_cost = (completion_check_input_tokens * completion_input_token_cost) + (completion_check_output_tokens * completion_output_token_cost)
        logger.info(f"Cost to check completion = {check_for_completion_cost}")
        total_cost += check_for_completion_cost

    return round(total_cost, 5), llm_token_usage

# def update_extraction_details(current_high_level_assistant_analytics_data, extraction_data):
#     # Use object.entries to loop thorugh extraction details
#     # For every element in the array, check if label is there in current_high_level_assistant_analytics_data[extraction_details][index]
#     # If yes, increment by 1 and if no make it 0
#     # If current_high_level_assistant_analytics_data[extraction_details] is empty, create a first element that's a dictionary and initialise the label with 1

def update_extraction_details(current_high_level_assistant_analytics_data, extraction_data):
    if len(current_high_level_assistant_analytics_data["extraction_details"]) == 0:
        current_high_level_assistant_analytics_data['extraction_details'] = {}

    for index, key in enumerate(extraction_data.keys()):
        found_label = False
        for detail in current_high_level_assistant_analytics_data['extraction_details']:
            if extraction_data[key] in detail:
                detail[label] += 1
                found_label = True
                break
        if not found_label:
            current_high_level_assistant_analytics_data['extraction_details'][index][extraction_data[key]] = 1


def update_high_level_assistant_analytics_data(current_high_level_assistant_analytics_data, run_details):
    logger.info(f"run details {run_details}")
    if current_high_level_assistant_analytics_data == None:
        current_high_level_assistant_analytics_data = HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA
    
    # extraction_details = update_extraction_details(current_high_level_assistant_analytics_data, run_details)
    # logger.info(f"current_high_level_assistant_analytics_data {current_high_level_assistant_analytics_data} \n Extracting data {extraction_details}")