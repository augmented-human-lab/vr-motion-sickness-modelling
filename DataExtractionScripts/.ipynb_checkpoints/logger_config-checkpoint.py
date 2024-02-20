# import logging
# import mylib

# def setup_logger(name, log_file, level=logging.INFO):
#     """To setup as many loggers as you want"""

#     handler = logging.FileHandler(log_file)        
#     handler.setFormatter(formatter)

#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)

#     return logger

import logging

# Configure logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define logger
logger = logging.getLogger(__name__)

# Example usage of logger functions
def my_function():
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

if __name__ == "__main__":
    my_function()