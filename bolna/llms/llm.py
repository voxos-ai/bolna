from bolna.helpers.logger_config import CustomLogger

custom_logger = CustomLogger(__name__)


class BaseLLM:
    def __init__(self, max_tokens=100, buffer_size=40, log_dir_name=None):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        self.logger = custom_logger.update_logger(log_dir_name=log_dir_name)

    async def respond_back_with_filler(self, messages):
        pass

    async def generate(self, messages, stream=True, classification_task=False, synthesize=True):
        pass
