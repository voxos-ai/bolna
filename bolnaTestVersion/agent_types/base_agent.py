from bolnaTestVersion.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseAgent:
    def __init__(self):
        self.agent_name = "base-agent"
