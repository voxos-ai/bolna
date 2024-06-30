from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

load_dotenv()
logger = configure_logger(__name__)


class OpenAIAssistantAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    async def generate(self, message, synthesize=False, meta_info=None):
        async for token in self.llm.generate_assistant_stream(message, synthesize=synthesize, meta_info=meta_info):
            logger.info('Agent: {}'.format(token))
            yield token
