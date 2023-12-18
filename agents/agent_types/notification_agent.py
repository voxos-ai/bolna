from agents.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ExtractionContextualAgent:
    def __init__(self, llm, prompt=None):
        self.brain = llm

        self.current_messages = 0
        self.is_inference_on = False
        self.has_intro_been_sent = False

    async def execute(self):
        pass