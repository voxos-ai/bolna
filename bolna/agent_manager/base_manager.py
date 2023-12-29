from bolna.helpers.logger_config import CustomLogger

custom_logger = CustomLogger(__name__)


class BaseManager:
    def __init__(self, log_dir_name=None):
        self.logger = custom_logger.update_logger(log_dir_name=log_dir_name)
