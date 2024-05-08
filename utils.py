import logging
import sys


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d | %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_handler.setFormatter(log_formatter)
    c_handler.setLevel(log_level)
    logger.addHandler(c_handler)
    if log_file:
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(log_formatter)
        f_handler.setLevel(log_level)
        logger.addHandler(f_handler)
    logger.setLevel(log_level)
    return logger
