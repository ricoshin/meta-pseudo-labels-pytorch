import logging
from collections import defaultdict, OrderedDict

import torch

log = logging.getLogger('main')

def topk_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  assert isinstance(output, torch.Tensor)  # [batch_size, n_classes]
  assert isinstance(target, torch.Tensor)  # [batch_size]
  assert isinstance(topk, (list, tuple, int))

  if isinstance(topk, int):
    topk = [topk]
  else:
    assert len(topk) >= 1

  maxk = max(topk)
  batch_size = target.size(0)

  pred_topk = output.topk(k=maxk, dim=1, largest=True, sorted=True)
  pred_topk = pred_topk.indices.t()  # [maxk, batch_size] : top-k predictions
  correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(1. / batch_size))
  return res if len(res) > 1 else res[0]


class MetricMonitor:
  def __init__(self, name, descending=False):
    self.name = name
    self.descending = descending
    self.best_value = float('inf') if descending else float('-inf')

  def __str__(self):
    return f'Best_{self.name}: {self.best_value:4.2%}'

  def is_best(self, value, verbose=True):
    if ((self.descending and value < self.best_value) or
        (not self.descending and value > self.best_value)):
      self.best_value = value
      log.info(f'Best {self.name}: {self.best_value:4.2%} !')
      return True
    return False

  @classmethod
  def by_metric(cls, name):
    assert isinstance(name, str)
    if name == 'loss':
      return cls(name, True)
    elif name in ['top1', 'top5']:
      return cls(name, False)
    else:
      raise Exception(f'Invalid metric name: {name}')


class AverageMeter:
  def __init__(self, name):
    self.name = name
    self.values = defaultdict(float)
    self.steps = defaultdict(int)

  def __len__(self):
    return len(self.values)

  def __repr__(self):
    return str(dict(
      name=self.name,
      values=dict(self.values),
      steps=dict(self.steps),
      ))

  def __str__(self):
    return self.strf()

  def __setitem__(self, key, value):
    self.values[key] = value
    self.steps[key] = 1

  def __getitem__(self, key):
    return self.values[key] / self.steps[key]

  def init(self):
    self.values = defaultdict(float)
    self.steps = defaultdict(float)

  def to_num_if_tensor(self, value):
    if isinstance(value, torch.Tensor):
      assert value.size() == torch.Size([])
      value = value.item()
    return value

  def update(self, **kwargs):
    for k, v in kwargs.items():
      self.__setitem__(k, v)

  def add(self, num=1, **kwargs):
    for k, v in kwargs.items():
      v = self.to_num_if_tensor(v)
      num = self.to_num_if_tensor(num)
      self.values[k] += v * num
      self.steps[k] += num

  def to_dict(self, *keys):
    out_dict = {}
    for k, v in self.values.items():
      if keys and k not in keys:
        continue
      out_dict[k] = v / self.steps[k]
    return out_dict

  def strf(self, delimiter=' | ', key_list=None):
    out = []
    _dict = self.to_dict()
    if key_list is None:
      key_list = self.values.keys()
    for k in key_list:
      assert k in self.values
      out.append(f'{k}: {_dict[k]:4.2%}')
    if out:
      return f'[{self.name}] ' + delimiter.join(out)
    else:
      return f'[{self.name}] Unknown'
