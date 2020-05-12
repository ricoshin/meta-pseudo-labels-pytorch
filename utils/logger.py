import os
import logging
from utils.config import Config


def set_logger(level='info', save_dir=''):
  # formatter = MyFormatter()
  log_fmt = '%(asctime)s [%(levelname)s] %(message)s'
  date_fmt = '%d/%m/%Y %H:%M:%S'
  formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
  log_level = getattr(logging, level.upper())

  # get logger
  logger = logging.getLogger('mpl')
  logger.setLevel(log_level)

  # stdio handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(log_level)
  logger.addHandler(stream_handler)

  # file handler
  if save_dir:
    log_file = os.path.join(save_dir, 'log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)


# class MyFormatter(logging.Formatter):
#   info_fmt = '[%(levelname)s] %(message)s'
#   else_fmt = '[%(levelname)s] %(message)s'
#
#   def __init__(self, fmt="%(message)s"):
#     logging.Formatter.__init__(self, fmt)
#
#   def format(self, record):
#     format_orig = self._fmt
#     if record.levelno == logging.INFO:
#       self._fmt = MyFormatter.info_fmt
#     else:
#       self._fmt = MyFormatter.else_fmt
#     result = logging.Formatter.format(self, record)
#     self._fmt = format_orig
#     return result
