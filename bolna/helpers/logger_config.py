import logging
import os

VALID_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
FORMAT = "%(asctime)s.%(msecs)03d {log_dir} %(levelname)s {%(module)s} [%(filename)s] [%(funcName)s] %(message)s"


class CustomLogger:
    def __init__(self, logger_name, enabled=True, logging_level='INFO'):
        self.logger_name = logger_name
        self.enabled = enabled
        self.logging_level = logging_level
        self.log_dir = None
        self.logger = self.configure_logger()
        self.old_factory = None

    def configure_logger(self):
        if self.logging_level not in VALID_LOGGING_LEVELS:
            self.logging_level = "INFO"

        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.logging_level)
        logger_format = FORMAT.replace('{log_dir}', '')

        logging_handlers = [logging.StreamHandler()]
        if self.log_dir:
            logging_handlers.append(logging.FileHandler('{}/{}.log'.format(self.log_dir, self.logger_name), mode='w'))
            logger_format = FORMAT.replace('{log_dir}', '%(log_dir)s')

        logger.handlers = []
        formatter = logging.Formatter(logger_format, datefmt="%Y-%m-%d %H:%M:%S")
        self.old_factory = logging.getLogRecordFactory()
        for handler in logging_handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        if not self.enabled:
            logger.disabled = True
        return logger

    def update_logger(self, log_dir_name=None):
        if log_dir_name:
            self.log_dir = log_dir_name
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            logging.setLogRecordFactory(self.record_factory)
            self.logger = self.configure_logger()
        return self.logger

    def record_factory(self, *args, **kwargs):
        record = logging.LogRecord(*args, **kwargs)
        record.log_dir = self.log_dir
        return record
