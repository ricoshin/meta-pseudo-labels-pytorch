import logging

import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from nn.wideresnet import WideResNet

log = logging.getLogger('mpl')


def get_num_classes(dataset_name):
  assert isinstance(dataset_name, str)
  try:
    return {
      'cifar10': 10,
      'svhn': 10,
      }[dataset_name]
  except KeyError:
    raise Exception(f'Invalid dataset name: {dataset_name}')


def parse_model(model):
  assert isinstance(model, str)
  parsed = model.split('_')
  assert parsed[0] == 'wresnet', f'{parsed[0]}'
  depth, widen_factor = parsed[1:]
  return int(depth), int(widen_factor)


def get_model(model, dataset, bn_momentum, dropout, data_parallel):
  depth, widen_factor = parse_model(model)
  num_classes = get_num_classes(dataset)
  model = WideResNet(
    depth, widen_factor, bn_momentum, dropout, num_classes)

  if data_parallel:
      model = model.cuda()
      model = DataParallel(model)
  else:
      import horovod.torch as hvd
      device = torch.device('cuda', hvd.local_rank())
      model = model.to(device)
  cudnn.benchmark = True
  return model

def get_optim_cls(optim_name):
  assert isinstance(optim_name, str)
  try:
    return {
      'sgd': 'SGD',
      'rmsprop': 'RMSprop',
      'adagrad': 'Adagrad',
      'adam': 'Adam',
      }[optim_name]
  except KeyError:
    raise Exception(f'Invalid optimizer name: {optim_name}')


def get_optimizer(optim_name, model, **kwargs):
  optim_cls = getattr(optim, get_optim_cls(optim_name))
  optimizer = optim_cls(model.parameters(), **kwargs)
  return optimizer


def get_scheduler(optimizer, n_steps, n_warmup):
  scheduler = CosineAnnealingLR(optimizer, T_max=n_steps-n_warmup)
  scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1., total_epoch=n_warmup, after_scheduler=scheduler)
  return scheduler
