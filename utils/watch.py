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
    t = time.time() - self.start
    t = time.strftime("(%Hhrs %Mmins %Ssecs)", time.gmtime(t))
    self.logger.info(f'Watch({self.name}): {t}')
