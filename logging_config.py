# logging_config.py
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    """
    Sets up the logging configuration for the entire application.

    - Logs to a rotating file (max 5MB, 5 backup files).
    - Logs to the console.
    - Sets different log levels for file and console handlers.
    """

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"application_log_{today_date}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # 1. Get the root logger
    # The root logger is the parent of all other loggers in your application.
    # By configuring it, you ensure that messages from all your modules can be handled.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set the lowest level for the root logger
                                         # so all messages (DEBUG and higher) are passed to handlers.

    # Clear existing handlers to prevent duplicate logs if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 2. Define a formatter
    # This dictates the format of your log messages.
    # We include timestamp, logger name, log level, and the message itself.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 3. Setup Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Only show INFO, WARNING, ERROR, CRITICAL on console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info("Console handler configured.")


    # 4. Setup Rotating File Handler
    # maxBytes: The maximum size of the log file before it rotates (e.g., 5 * 1024 * 1024 bytes = 5MB)
    # backupCount: The number of backup files to keep (e.g., 5 means application.log.1, .2, .3, .4, .5)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024, # 5 MB
        backupCount=5,
        encoding='utf-8' # Good practice for robust logging
    )
    file_handler.setLevel(logging.DEBUG) # Log all DEBUG messages and higher to the file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging.info(f"Rotating file handler configured. Log file: {log_file_path}")

    # Optionally, if you want to capture warnings for unhandled exceptions in threads etc.
    # logging.captureWarnings(True)

    logging.info("Logging setup complete.")