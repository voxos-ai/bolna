import json
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from dotenv import load_dotenv 
import os
load_dotenv()


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm, log_dir_name=None):
        super().__init__(log_dir_name)
        self.brain = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.classification_model))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages):
        prompt = [
            {'role': 'system', 'content': CHECK_FOR_COMPLETION_PROMPT},
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
