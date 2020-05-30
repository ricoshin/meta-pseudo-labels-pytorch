import logging
import time


class Watch:
  def __init__(self, name, logger):
    assert isinstance(name, str)
    assert isinstance(logger, logging.Logger)
    self.name = name
    self.logger = logger

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, type, value, trace_back):
    secs = time.time() - self.start
    hrs = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    secs = int(secs % 60)
    self.logger.info(f'Watch({self.name}): {hrs} hrs {mins} mins {secs} secs')
