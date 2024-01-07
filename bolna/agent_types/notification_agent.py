from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ExtractionContextualAgent(BaseAgent):
    def __init__(self, llm, prompt=None):
        super().__init__()
        self.brain = llm

        self.current_messages = 0
        self.is_inference_on = False
        self.has_intro_been_sent = False

    async def execute(self):
        pass