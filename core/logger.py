import logging
import os

import colorlog

_VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
_PROJECT_LOG_HANDLER_ATTR = "_video_search_engine_handler"


def _resolve_log_level() -> int:
    """Return the log level requested via the LOG_LEVEL env var, or INFO."""

    raw = (os.getenv("LOG_LEVEL") or "").strip().upper()
    if raw in _VALID_LEVELS:
        return getattr(logging, raw)
    return logging.INFO


def setup_logging():
    """
    Sets up a centralized, colored logger.
    This function should be called once at the beginning of the application.
    """
    root_logger = logging.getLogger()

    # Avoid adding handlers multiple times if this function is called again.
    if any(
        getattr(handler, _PROJECT_LOG_HANDLER_ATTR, False)
        for handler in root_logger.handlers
    ):
        return

    root_logger.setLevel(_resolve_log_level())

    if root_logger.hasHandlers():
        return

    # Create a colored formatter
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Create a console handler and set the formatter
    console_handler = colorlog.StreamHandler()
    setattr(console_handler, _PROJECT_LOG_HANDLER_ATTR, True)
    console_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(console_handler)

    # Set the root logger for this function's module
    log = logging.getLogger(__name__)
    log.info("Colored logging is configured.")
