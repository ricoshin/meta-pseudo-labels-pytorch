import os
import logging
import types


def get_formatter(name):
  if name == 'main':
    log_fmt = f'%(asctime)s [{name}] [%(levelname)s] %(message)s'
  elif name == 'result':
    log_fmt = f'%(asctime)s [{name}] [%(levelname)s] %(message)s'
  else:
    raise Exception(f'Unknown logger namespace: {name}')
  return logging.Formatter(log_fmt, datefmt='%d/%m/%y %H:%M')


def get_log_level(level):
  return getattr(logging, level.upper())


def set_stream_handler(name, level):
  assert isinstance(level, str)
  # set level
  log = logging.getLogger(name)
  log.setLevel(get_log_level(level))
  # stdio standard handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(get_formatter(name))
  stream_handler.setLevel(get_log_level(level))
  log.stream_handler = stream_handler
  log.addHandler(stream_handler)
  # stdio newline handler
  newline_handler = logging.StreamHandler()
  newline_handler.setFormatter(fmt='')
  newline_handler.setLevel(get_log_level(level))
  # newline() to switch the newline_handler
  def newline(self, num_lines=1, log_level='info'):
    # handler swithching trick
    self.removeHandler(self.stream_handler)
    self.addHandler(self.newline_handler)
    for i in range(num_lines):
        getattr(self, log_level)('')
    self.removeHandler(self.newline_handler)
    self.addHandler(self.stream_handler)
  log.newline_handler = newline_handler
  log.newline = types.MethodType(newline, log)


def set_file_handler(name, level, save_dir, filename):
  assert isinstance(level, str)
  # file handler
  log = logging.getLogger(name)
  log_file = os.path.join(save_dir, filename)
  file_handler = logging.FileHandler(log_file)
  file_handler.setFormatter(get_formatter(name))
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
