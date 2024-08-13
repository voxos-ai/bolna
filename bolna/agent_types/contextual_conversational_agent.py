import json
import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolna.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


class StreamingContextualAgent(BaseAgent):
    """
    A class representing a streaming contextual conversational agent.

    Attributes:
        llm (LLM): The language model used by the agent.
        conversation_completion_llm (OpenAiLLM): The language model used for conversation completion.
        history (list): The conversation history.

    Methods:
        check_for_completion: Checks for completion of a conversation.
        generate: Generates a response based on the conversation history.

    """
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = OpenAiLLM(model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.model))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt = CHECK_FOR_COMPLETION_PROMPT):
        """
        Checks for completion of a conversation.

        Args:
            messages (list): The list of messages in the conversation.
            check_for_completion_prompt (str): The prompt used for checking completion.

        Returns:
            dict: The completion answer.

        """
        prompt = [
            {'role': 'system', 'content': check_for_completion_prompt},
            {'role': 'user', 'content': format_messages(messages, use_system_prompt=True)}]

        answer = None
        response = await self.conversation_completion_llm.generate(prompt, request_json=True)
        answer = json.loads(response)

        logger.info('Agent: {}'.format(answer['answer']))
        return answer

    async def generate(self, history, synthesize=False, meta_info = None):
        """
        Generates a response based on the conversation history.

        Args:
            history (list): The conversation history.
            synthesize (bool): Whether to synthesize the response.
            meta_info (dict): Additional meta information.

        Yields:
            str: The generated response.

        """
        async for token in self.llm.generate_stream(history, synthesize=synthesize, meta_info = meta_info):
            logger.info('Agent: {}'.format(token))
            yield token
