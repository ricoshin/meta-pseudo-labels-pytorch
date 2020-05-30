import logging
import os
from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils.color import Color
from model import get_model, get_optimizer, get_scheduler
from learn.metric import MetricMonitor

log = logging.getLogger('main')


class ModelControl:
  def __init__(self, model, optim, sched):
    self.model = model
    self.optim = optim
    self.sched = sched

  def cuda_(self, *args, **kwargs):
    self.model = self.model.cuda(*args, **kwargs)

  def train(self, mode=True):
    self.model.train(mode)
    return self

  def eval(self):
    self.model.eval()
    return self

  def detach_(self):
    for param in self.model.parameters():
      param.detach_().requires_grad_(True)
      param.retain_grad()
    for buffer in self.model.buffers():
      buffer.detach_()  # just in case


  def step_all(self, clip_grad=None, debug=False):
    if clip_grad:
      clip_grad_norm_(self.model.parameters(), clip_grad)
    self.optim.step(debug=debug)
    self.optim.zero_grad()
    self.sched.step()


class TrainingManager:
  def __init__(self, cfg, data_parallel=False):
    self.cfg = cfg
    self.monitor = MetricMonitor.by_metric(
      metric=cfg.valid.metric, prefix='valid')

    self.step = 0
    self.step_max = cfg.comm.n_steps
    self.pbar_train = self.pbar_test = None

    self.model_ctrls = {}
    model_kwargs = {
      'model': cfg.model,
      'dataset': cfg.dataset,
      'bn_momentum': cfg.comm.bn_decay,
      'data_parallel': False if cfg.test_only else data_parallel,
    }
    optim_kwargs = {
      'optim_name': cfg.optim.name,
      'momentum': cfg.optim.momentum,
      'nesterov': cfg.optim.nesterov,
      'weight_decay': cfg.comm.w_decay,
    }

    log.info('Create a student model control.')
    model = get_model(dropout=cfg.stdn.dropout, name='stdn', **model_kwargs)
    optim = get_optimizer(model=model, lr=cfg.stdn.lr, **optim_kwargs)
    sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
    self.stdn = ModelControl(model, optim, sched)
    self.model_ctrls['stdn'] = self.stdn

    if cfg.method.is_mpl and not cfg.test_only:
      log.info('Create a teacher model control.')
      model = get_model(dropout=cfg.tchr.dropout, name='tchr', **model_kwargs)
      optim = get_optimizer(model=model, lr=cfg.tchr.lr, **optim_kwargs)
      sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
      self.tchr = ModelControl(model, optim, sched)
      self.model_ctrls['tchr'] = self.tchr

  def __str__(self):
    return self.strf()

  def strf(self, delimiter=' | '):
    tag = self.cfg.tag if self.cfg.tag else 'no_tag'
    mpl_postfix = '_mpl' if self.cfg.method.is_mpl else ''
    return delimiter.join([
      f'{Color.SELECTED}{tag}{Color.END}',
      f'method: {self.cfg.method.base +  mpl_postfix}',
      f'step: {self.step:7d}/{self.step_max:7d}',
      f'lr_stdn: {self.stdn.sched.get_lr()[0]:5.3f}',
      ])

  @property
  def is_finished(self):
    return self.step >= self.step_max

  @property
  def is_last_step(self):
    return self.step == self.step_max

  @property
  def is_valid_step(self):
    return self.step % self.cfg.valid.interval == 0

  def train(self, mode=True):
    for ctrl in self.model_ctrls.values():
      ctrl.train(mode)
    return self

  def eval(self):
    for ctrl in self.model_ctrls.values():
      ctrl.eval()
    return self

  def cuda_(self):
    for ctrl in self.model_ctrls.values():
      ctrl.cuda_()

  def train_step_generator(self, disable_pbar=False, local_max_step=None):
    self.pbar_train = tqdm(
      initial=self.step, total=self.step_max, desc=f'[train]',
      leave=False, disable=disable_pbar)
    local_step = 0
    for _ in range(self.step_max - self.step):
      local_step += 1
      self.step += 1
      self.pbar_train.update(1)
      yield self.step
      if local_step == local_max_step:
        break
    self.pbar_train.close()

  def test_step_generator(self, test_loader, mode, disable_pbar=False):
    self.pbar_test = tqdm(
      iterable=test_loader, desc=f'[{mode}]',
      leave=False, disable=disable_pbar)
    for x, y in self.pbar_test:
      yield x, y
    self.pbar_test.close()

  @contextmanager
  def logging(self):
    """to avoid collision with afterimage of the progress bar."""
    if self.pbar_train:
      self.pbar_train.clear()
    yield
    if self.pbar_train:
      self.pbar_train.refresh()

  def save(self, tag, verbose=False, save_dir=None):#, status=None):
    if not self.cfg.save_dir and not save_dir:
      return
    if not save_dir:
      save_dir = self.cfg.save_dir
    logs = []
    for name, ctrl in self.model_ctrls.items():
      filename = name + (f'_{tag}' if tag else '') + '.pt'
      filepath = os.path.join(save_dir, filename)
      torch.save({
        'step': self.step,
        'record': self.monitor.best_value,
        'model': ctrl.model.state_dict(),
        'optim': ctrl.optim.state_dict(),
        'sched': ctrl.sched.state_dict(),
      }, filepath)
      logs.append(filepath)
    if logs:
      log_level = 'info' if verbose else 'debug'
      getattr(log, log_level)(f'Saved snapshot to: {", ".join(logs)}')

  def load_if_available(self, tag, verbose=False, save_dir=None):
    if not self.cfg.save_dir and not save_dir:
      return
    if not save_dir:
      save_dir = self.cfg.save_dir
    logs = []
    for name, ctrl in self.model_ctrls.items():
      filename = name + (f'_{tag}' if tag else '') + '.pt'
      filepath = os.path.join(save_dir, filename)
      if os.path.exists(filepath):
        loaded = torch.load(filepath)
        self.step = loaded['step']
        self.monitor.best_value = loaded['record']
        if self.cfg.test_only:
          loaded['model'] = _convert_if_data_paralleled(loaded['model'])
        ctrl.model.load_state_dict(loaded['model'])
        ctrl.optim.load_state_dict(loaded['optim'])
        ctrl.sched.load_state_dict(loaded['sched'])
        logs.append(filepath)
    if logs:
      log_level = 'info' if verbose else 'debug'
      getattr(log, log_level)(f'Loaded snapshot from: {", ".join(logs)}')
      getattr(log, log_level)(f'Resume from step {self.step}.')

def _convert_if_data_paralleled(model_dict):
  if not list(model_dict.keys())[0][:7] == 'module.':
    return model_dict
  new_model_dict = OrderedDict()
  for k, v in model_dict.items():
    new_model_dict[k[7:]] = v
  return new_model_dict
