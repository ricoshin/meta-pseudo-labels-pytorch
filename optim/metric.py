from collections import defaultdict, OrderedDict

import torch


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  assert isinstance(output, torch.Tensor)  # [batch_size, n_classes]
  assert isinstance(target, torch.Tensor)  # [batch_size]

  maxk = max(topk)
  batch_size = target.size(0)

  pred_topk = output.topk(k=maxk, dim=1, largest=True, sorted=True)
  pred_topk = pred_topk.indices.t()  # [maxk, batch_size] : top-k predictions
  correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(1. / batch_size))
  return res


class AverageMeter:
  def __init__(self, name):
    self.name = name
    self.values = defaultdict(float)
    self.steps = defaultdict(int)

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

  def to_dict(self):
    out_dict = {}
    for k, v in self.values.items():
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
    return f'[{self.name}] ' + delimiter.join(out)
