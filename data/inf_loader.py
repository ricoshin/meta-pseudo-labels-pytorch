from torch.utils.data import DataLoader

class InfiniteDataLoader:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def __iter__(self):
    return self

  def __next__(self):
    while True:
      for loaded in DataLoader(*self.args, **self.kwargs):
        yield loaded
