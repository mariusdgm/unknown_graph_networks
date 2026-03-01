import os
import sys

import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Create and return a customized logger.

    Args:
        env_name: identifier string to be used in the name of the file (if the log file is created).
        folder_path: folder where to save the log file. If None, no folder is created.
        identifier_string: an identifier string that will appear in each logged message.

    Returns:
        Logger: customized logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file path is provided, set up file handler
    if log_file is not None:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1024 * 1024 * 5, backupCount=3
        )  # 5MB per file, with 3 backups

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cleanup_file_handlers(experiment_logger=None):
    """Cleans up all handlers of experiment_logger logger instance.

    Args:
        experiment_logger (Logger, optional): If experiment_logger is None, then all loggers will be cleaned up.
        Defaults to None.
    """
    # Get all active loggers
    if experiment_logger:
        for handler in experiment_logger.handlers:
            handler.close()
            experiment_logger.removeHandler(handler)

    else:
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
