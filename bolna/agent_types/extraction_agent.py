from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ExtractionContextualAgent:
    def __init__(self, llm, prompt=None):
        self.brain = llm
        self.current_messages = 0
        self.is_inference_on = False
        self.has_intro_been_sent = False

    async def generate(self, history, stream=True, synthesize=False):
        logger.info("extracting json from the previous conversation data")
        json_data = await self.brain.generate(history, stream=False, synthesize=False, request_json=True)
        return json_data
