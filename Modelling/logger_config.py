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

def setup_logger():
    # Configure logging
    logging.basicConfig(filename='log/data_set_4_aggregation.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define logger
    logger = logging.getLogger(__name__)
    return logger