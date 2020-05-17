import logging

from os import path
from torch.utils.tensorboard import SummaryWriter


class TFWriters:
  def __init__(self, log_dir, name_list, deactivated=False):
    assert isinstance(name_list, (list, tuple))
    assert all([isinstance(n, str) for n in name_list])
    self.log_dir = log_dir
    self.writers = {}
    for name in name_list:
      self.writers[name] = DeactivatableSummaryWriter(
        log_dir=path.join(self.log_dir, name), deactivated=deactivated)


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

  def add_scalar(self, name, scalar_value, step, postfix=''):
    self.writers[name].add_scalar(name + postfix, scalar_value, step)

  def add_scalars(self, name, tag_scalar_dict, step, postfix=''):
    self.writers[name].add_scalars(name + postfix, tag_scalar_dict, step)


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
