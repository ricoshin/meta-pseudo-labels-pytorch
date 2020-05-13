import os
import logging

log = logging.getLogger('mpl')


def get_formatter():
  log_fmt = '%(asctime)s [%(levelname)s] %(message)s'
  date_fmt = '%d/%m/%Y %H:%M:%S'
  return logging.Formatter(log_fmt, datefmt=date_fmt)


def get_log_level(level):
  return getattr(logging, level.upper())


def set_stream_handler(level):
  assert isinstance(level, str)
  # set level
  log = logging.getLogger('mpl')
  log.setLevel(get_log_level(level))
  # stdio handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(get_formatter())
  stream_handler.setLevel(get_log_level(level))
  log.addHandler(stream_handler)


def set_file_handler(level, save_dir):
  assert isinstance(level, str)
  # file handler
  log_file = os.path.join(save_dir, 'log')
  file_handler = logging.FileHandler(log_file)
  file_handler.setFormatter(get_formatter())
  file_handler.setLevel(get_log_level(level))
  log.addHandler(file_handler)


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
