# logger_setup.py
import logging
import os
import sys

def setup_logger(log_file="fetch_status.log"):
    """
    Sets up a logger that writes to a file and the console.
    """
    # Remove existing handlers to avoid duplicates if re-run
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Create logger
    logger = logging.getLogger("PSX_Fetcher")
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger