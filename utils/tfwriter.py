import logging

from os import path
from torch.utils.tensorboard import SummaryWriter


class TFWriters:
  _sub_dir = 'tfrecord'
  def __init__(self, log_dir, deactivated=False):
    self.log_dir = path.join(log_dir, TFWriters._sub_dir)
    self.train = DeactivatableSummaryWriter(
        log_dir=path.join(self.log_dir, 'train'), deactivated=deactivated)
    self.test = DeactivatableSummaryWriter(
        log_dir=path.join(self.log_dir, 'test'), deactivated=deactivated)

  def is_activated(self):
    if self.train.is_activated():
      assert self.test.is_activated()
      return True
    assert not self.test.is_activated()
    return False

  def activate(self):
    self.train.activate()
    self.test.activate()

  def deactivate(self):
    self.train.deactivate()
    self.test.deactivate()


class DeactivatableSummaryWriter(SummaryWriter):
  def __init__(self, deactivated=False, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.deactivated = deactivated

  def is_activated(self):
    return not self.deactivated

  def activate(self):
    self.deactivated = False

  def deactivate(self):
    self.deactivated = True

  def __getattr__(self, key):
    attr = self.__getattribute__(key)
    if self.deactivated and callable(attr):
      return None
    return attr
