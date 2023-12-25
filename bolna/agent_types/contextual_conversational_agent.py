import json
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from dotenv import load_dotenv 
import os
load_dotenv()

logger = configure_logger(__name__)


class StreamingContextualAgent:
    def __init__(self, llm):
        self.brain = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model=os.getenv('CHECK_FOR_COMPLETION_LLM'))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages):
        prompt = [
            {'role': 'system', 'content': CHECK_FOR_COMPLETION_PROMPT},
            {'role': 'user', 'content': format_messages(messages)}]

        answer = None
        response = await self.conversation_completion_llm.generate(prompt, True, False, request_json=True)
        answer = json.loads(response)

        logger.info('Agent: {}'.format(answer['answer']))
        return answer['answer'].lower() == "yes"

    async def generate(self, history, synthesize=False):
        async for token in self.brain.generate_stream(history, synthesize=synthesize):
            logger.info('Agent: {}'.format(token))
            yield token
