import logging
import os

import torch
from torch import nn, optim
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from model.wideresnet import WideResNet

log = logging.getLogger('mpl')

_optim_map = {
  'sgd': 'SGD',
  'rmsprop': 'RMSprop',
  'adagrad': 'Adagrad',
  'adam': 'Adam',
}


def parse_model(model):
  assert isinstance(model, str)
  parsed = model.split('_')
  assert parsed[0] == 'wresnet', f'{parsed[0]}'
  depth, widen_factor = parsed[1:]
  return int(depth), int(widen_factor)


def get_num_classes(dataset):
  assert isinstance(dataset, str)
  try:
    return dict(
      cifar10=10,
      svhn=10,
      )[dataset]
  except KeyError:
    raise Exception(f'Invalid dataset name: {dataset}')


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


def get_optimizer(optim_name, model, **kwargs):
  assert optim_name in _optim_map, f'Unknown optimizer name: {optim_name}'
  optim_cls = getattr(optim, _optim_map[optim_name])
  optimizer = optim_cls(model.parameters(), **kwargs)
  return optimizer


class ModelManager:
  def __init__(self, cfg, data_parallel=True):
    self.models = {}
    self.optims = {}
    model_kwargs = {
      'model': cfg.model,
      'dataset': cfg.dataset,
      'bn_momentum': cfg.comm.bn_decay,
      'data_parallel': data_parallel,
    }
    optim_kwargs = {
      'optim_name': cfg.optim.name,
      'momentum': cfg.optim.momentum,
      'nesterov': cfg.optim.nesterov,
      'weight_decay': cfg.comm.w_decay,
    }
    self.tchr = get_model(dropout=cfg.tchr.dropout, **model_kwargs)
    self.tchr_optim = get_optimizer(
      model=self.tchr, lr=cfg.tchr.lr, **optim_kwargs)
    self.models['tchr'] = self.tchr
    self.optims['tchr'] = self.tchr_optim

    if cfg.method.mpl:
      self.stdn = get_model(dropout=cfg.stdn.dropout, **model_kwargs)
      self.stdn_optim = get_optimizer(
        model=self.stdn, lr=cfg.stdn.lr, **optim_kwargs)
      self.models['stdn'] = self.stdn
      self.optims['stdn'] = self.stdn_optim

  def train(self, mode=True):
    for model in self.models.values():
      model.train(mode)

  def eval(self, mode=True):
    for model in self.models.values():
      model.eval(mode)

  def cuda_(self):
    for name, model in self.models.items():
      self.models[name] = model.cuda()

  def save(self, cfg, status=None):
    if not cfg.save_dir:
      return
    for k in self.models.keys():
      filepath = os.path.join(cfg.save_dir, k + '.torch')
      torch.save({
        'model': self.models[k].state_dict(),
        'optim': self.optims[k].state_dict(),
      }, filepath)
      log.info(f'Saved model to: {filepath}')
    if status:
      filepath = os.path.join(cfg.save_dir, 'status.torch')
      torch.save(status, filepath)
      log.info(f'Saved status to: {filepath}')

  def load_if_available(self, cfg):
    if not cfg.save_dir:
      return
    for k in self.models.keys():
      filepath = os.path.join(cfg.save_dir, k + '.torch')
      if os.path.exists(filepath):
        loaded = torch.load(filepath)
        self.models[k].load_state_dict(loaded['model'])
        self.optims[k].load_state_dict(loaded['optim'])
        log.info(f'Loaded model from: {filepath}')
    filepath = os.path.join(cfg.save_dir, 'status.torch')
    status = torch.load(filepath) if os.path.exists(filepath) else None
    return status
