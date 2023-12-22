from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40):
        self.stream = stream
        self.buffer_size = buffer_size
