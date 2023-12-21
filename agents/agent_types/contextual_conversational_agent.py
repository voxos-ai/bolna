import json
from agents.helpers.logger_config import configure_logger
from agents.helpers.utils import format_messages
from agents.llms import OpenAiLLM

logger = configure_logger(__name__)


class StreamingContextualAgent:
    def __init__(self, llm):
        # super(ContextualAgent, self).__init__()
        self.llm = llm

        self.conversation_completion_llm = OpenAiLLM(classification_model="gpt-3.5-turbo-1106")
        self.history = [{'content': ""}]
        
    #CHEcking if the conversation is complete
    async def check_for_completion(self, messages):
        logger.info("checking for completion")
        json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}

        prompt = [{'role': 'system', 'content':  (f"You are an helpful AI assistant that's having a conversation with customer. "
                                      f"Based on the given transcript, should you cut the call?\n\n NOTE: If user is not intereseted in talking, or is annoying or something, we need to cut the phone. Kindly "
                                      f"response in a json format {json_format}")}]
                                      
        prompt.append({'role': 'user', 'content': format_messages(messages)})

        answer = None
        response = await self.conversation_completion_llm.generate(prompt, True, False, request_json=True)
        logger.info(f"response: {response}")
        answer = json.loads(response)
        return answer['answer'].lower() == "yes"

    async def generate(self, history, synthesize=False):
        async for token in self.llm.generate_stream(history, synthesize=synthesize):
            yield token
