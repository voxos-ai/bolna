from bolna.helpers.logger_config import configure_logger
import asyncio
logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()

    def generate(self, text):
        pass

    def push(self, text):
        pass
    
    def synthesize(self, text):
        pass