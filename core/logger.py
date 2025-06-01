import logging
import colorlog

def setup_logging():
    """
    Sets up a centralized, colored logger.
    This function should be called once at the beginning of the application.
    """
    root_logger = logging.getLogger()
    
    # Avoid adding handlers multiple times if this function is called again.
    if root_logger.hasHandlers():
        return

    root_logger.setLevel(logging.INFO)

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
    console_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(console_handler)

    # Set the root logger for this function's module
    log = logging.getLogger(__name__)
    log.info("Colored logging is configured.")
