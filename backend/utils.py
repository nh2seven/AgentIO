import logging
import os
from datetime import datetime

# Might be moved elsewhere later
logs_dir = "data/logs"
if not os.path.exists(logs_dir):
    os.system("./setup.sh")


# Function to log system events
def logger():
    log_filename = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add both handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


if __name__ == "__main__":
    exit(0)
