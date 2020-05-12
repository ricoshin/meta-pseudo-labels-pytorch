from torch.utils.data import DataLoader


class InfiniteDataLoader:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    self.iterator = self.new_iterator()

  def new_iterator(self):
    return iter(DataLoader(*self.args, **self.kwargs))

  def __iter__(self):
    return self

  def __next__(self):
    try:
      return next(self.iterator)
    except StopIteration:
      self.iterator = self.new_iterator()
      return next(self.iterator)
