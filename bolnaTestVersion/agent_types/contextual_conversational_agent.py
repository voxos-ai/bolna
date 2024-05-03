import json
import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolnaTestVersion.helpers.utils import format_messages
from bolnaTestVersion.llms import OpenAiLLM
from bolnaTestVersion.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolnaTestVersion.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.classification_model))
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
        async for token in self.llm.generate_stream(history, synthesize=synthesize):
            logger.info('Agent: {}'.format(token))
            yield token
