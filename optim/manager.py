import logging
import os
from contextlib import contextmanager

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import data, utils
from utils.color import Color
from optim import get_model, get_optimizer, get_scheduler
from optim.metric import MetricMonitor

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

  def step_all(self, clip_grad=None):
    if clip_grad:
      clip_grad_norm_(self.model.parameters(), clip_grad)
    self.optim.step()
    self.optim.zero_grad()
    self.sched.step()


class TrainingManager:
  def __init__(self, cfg, loaders, writers, data_parallel=True):
    assert isinstance(loaders, data.dataloaders.DataLoaderTriplet)
    assert isinstance(writers, utils.tfwriter.TFWriters)
    self.cfg = cfg
    self.loaders = loaders
    self.writers = writers
    self.monitor = MetricMonitor.by_metric(
      metric=cfg.valid.metric, prefix='valid')

    self.step = 0
    self.step_max = cfg.comm.n_steps
    self.pbar = None

    self.model_ctrls = {}
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

    log.info('Create a student model control.')
    model = get_model(dropout=cfg.stdn.dropout, **model_kwargs)
    optim = get_optimizer(model=model, lr=cfg.stdn.lr, **optim_kwargs)
    sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
    self.stdn = ModelControl(model, optim, sched)
    self.model_ctrls['stdn'] = self.stdn

    if cfg.method.mpl:
      log.info('Create a teacher model control.')
      model = get_model(dropout=cfg.tchr.dropout, **model_kwargs)
      optim = get_optimizer(model=model, lr=cfg.tchr.lr, **optim_kwargs)
      sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
      self.tchr = ModelControl(model, optim, sched)
      self.model_ctrls['tchr'] = self.tchr

  def __str__(self):
    return self.strf()

  def strf(self, delimiter=' | '):
    mpl_postfix = '_mpl' if self.cfg.method.mpl else ''
    return delimiter.join([
      f'{Color.SELECTED}{self.cfg.tag}{Color.END}',
      f'method: {self.cfg.method.base +  mpl_postfix}',
      f'step: {self.step:7d}/{self.step_max:7d}',
      f'lr_stdn: {self.stdn.sched.get_lr()[0]:5.3f}',
      # f'lr_s: {self.stdn.sched.get_lr()[0]:5.3f}',
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

  def step_generator(self, mode):
    assert mode in ['train', 'valid', 'test']
    # desc = mode + cfg.save_dir
    if mode == 'train':
      self.pbar_train = tqdm(initial=self.step, total=self.step_max,
        desc=f'[{mode}]', leave=False)
      for _ in range(self.step_max - self.step):
        self.step += 1
        self.pbar_train.update(1)
        yield self.step
      self.pbar_train.close()
    elif mode in ['valid', 'test']:
      self.pbar_test = tqdm(self.loaders.test, desc=mode, leave=False)
      for x, y in self.pbar_test:
        yield x, y
      self.pbar_test.close()

  @contextmanager
  def logging(self):
    """to avoid collision with afterimage of the progress bar."""
    self.pbar_train.clear()
    yield
    self.pbar_train.refresh()

  def save(self, cfg, tag, verbose=False):#, status=None):
    if not cfg.save_dir:
      return
    logs = []
    for name, ctrl in self.model_ctrls.items():
      filename = name + (f'_{tag}' if tag else '') + '.pt'
      filepath = os.path.join(cfg.save_dir, filename)
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

  def load_if_available(self, cfg, tag, verbose=False):
    if not cfg.save_dir:
      return
    logs = []
    for name, ctrl in self.model_ctrls.items():
      filename = name + (f'_{tag}' if tag else '') + '.pt'
      filepath = os.path.join(cfg.save_dir, filename)
      if os.path.exists(filepath):
        loaded = torch.load(filepath)
        self.step = loaded['step']
        self.monitor.best_value = loaded['record']
        ctrl.model.load_state_dict(loaded['model'])
        ctrl.optim.load_state_dict(loaded['optim'])
        ctrl.sched.load_state_dict(loaded['sched'])
        logs.append(filepath)
    if logs:
      log_level = 'info' if verbose else 'debug'
      getattr(log, log_level)(f'Loaded snapshot from: {", ".join(logs)}')
      getattr(log, log_level)(f'Resume from step {self.step}.')
