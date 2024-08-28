import logging

VALID_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def configure_logger(file_name, enabled=True, logging_level='INFO'):
    """
    Configures a logger for the specified file.

    Parameters:
    - file_name (str): The name of the file for which the logger is being configured.
    - enabled (bool): Flag to enable or disable the logger. Default is True.
    - logging_level (str): The logging level to set. Must be one of the valid levels. Default is 'INFO'.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    if logging_level not in VALID_LOGGING_LEVELS:
        logging_level = "INFO"

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(file_name)

    if not enabled:
        logger.disabled = True

    return logger