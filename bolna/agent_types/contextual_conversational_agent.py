import json
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm, log_dir_name=None):
        super().__init__(log_dir_name)
        self.brain = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model="gpt-3.5-turbo-1106")
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages):
        json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}
        prompt = [
            {'role': 'system', 'content': (f"You are an helpful AI assistant that's having a conversation with "
                                           f"customer. Based on the given transcript, should you cut the call?\n\n "
                                           f"NOTE: If user is not interested in talking, or is annoying or something, "
                                           f"we need to cut the phone. Kindly response in a json format {json_format}")},
            {'role': 'user', 'content': format_messages(messages)}]

        answer = None
        response = await self.conversation_completion_llm.generate(prompt, True, False, request_json=True)
        answer = json.loads(response)

        self.logger.info('Agent: {}'.format(answer['answer']))
        return answer['answer'].lower() == "yes"

    async def generate(self, history, synthesize=False):
        async for token in self.brain.generate_stream(history, synthesize=synthesize):
            self.logger.info('Agent: {}'.format(token))
            yield token
