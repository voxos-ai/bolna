import json
from agents.helpers.logger_config import configure_logger
from agents.helpers.utils import format_messages
from agents.llms import OpenAiLLM

logger = configure_logger(__name__)


class StreamingContextualAgent:
    # Main job for contextual Agent is to orchestrate calling specific LLMs for required operation
    def __init__(self, llm):
        self.brain = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model="gpt-3.5-turbo")
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages):
        json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}
        prompt = [{'role': 'system',
                   'content': (f"You are an helpful AI assistant that's having a conversation with customer. "
                               f"Based on the given transcript, should you cut the call?\n\n NOTE: Kindly "
                               f"response in a json format {json_format}")}]

        prompt.append({'role': 'user', 'content': format_messages(messages)})
        answer = None
        async for response in self.conversation_completion_llm.generate(messages, True, False, request_json=True):
            answer = response
        answer = json.loads(answer)
        return answer['answer'].lower() == "yes"

    async def generate(self, history, synthesize=False):
        async for token in self.brain.generate_stream(history, synthesize=synthesize):
            yield token
