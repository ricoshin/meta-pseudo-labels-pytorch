import logging
import os

import torch
from tqdm import tqdm

from model.loader import get_model, get_optimizer, get_scheduler
from optim.metric import MetricMonitor

log = logging.getLogger('mpl')


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


class TrainingManager:
  def __init__(self, cfg, data_parallel=True):
    self.cfg = cfg
    self.step = 0
    self.step_max = cfg.comm.n_steps
    self.pbar = None
    self.monitor = MetricMonitor.by_metric(cfg.valid.metric)

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
    log.info('Create a teacher model control.')
    model = get_model(dropout=cfg.tchr.dropout, **model_kwargs)
    optim = get_optimizer(model=model, lr=cfg.tchr.lr, **optim_kwargs)
    sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
    self.tchr = ModelControl(model, optim, sched)
    self.model_ctrls['tchr'] = self.tchr

    if cfg.method.mpl:
      log.info('Create a student model control.')
      model = get_model(dropout=cfg.stdn.dropout, **model_kwargs)
      optim = get_optimizer(model=model, lr=cfg.stdn.lr, **optim_kwargs)
      sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
      self.stdn = ModelControl(model, optim, sched)
      self.model_ctrls['stdn'] = self.stdn

  def __str__(self):
    return self.strf()

  def strf(self, delimiter=' | '):
    return delimiter.join([
      f'step: {self.step:7d}/{self.step_max:7d}',
      f'base: {self.cfg.method.base}',
      f'mpl: {self.cfg.method.mpl}',
      f'lr_t: {self.tchr.sched.get_lr()[0]:5.3f}',
      # f'lr_s: {self.stdn.sched.get_lr()[0]:5.3f}',
      ])

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

  def step_generator(self, pbar=True):
    if pbar:
      self.pbar = tqdm(initial=self.step, total=self.step_max, leave=False)
    for step in range(self.step, self.step_max):
      self.step += 1
      if pbar:
        self.pbar.update(1)
      yield step

  def log(self, *args, delimiter=' | '):
    if self.pbar:
      self.pbar.clear()  # to avoid afterimage of pbar
    log.info(delimiter.join(map(str, args)))

  def save(self, cfg):#, status=None):
    if not cfg.save_dir:
      return
    for name, ctrl in self.model_ctrls.items():
      filepath = os.path.join(cfg.save_dir, name + '.torch')
      torch.save({
        'step': self.step,
        'record': self.monitor.best_value,
        'model': ctrl.model.state_dict(),
        'optim': ctrl.optim.state_dict(),
        'sched': ctrl.sched.state_dict(),
      }, filepath)
      log.info(f'Saved snapshot to: {filepath}')
    # if monitor:
    #   filepath = os.path.join(cfg.save_dir, 'status.torch')
    #   torch.save(status, filepath)
    #   log.info(f'Saved status to: {filepath}')

  def load_if_available(self, cfg):
    if not cfg.save_dir:
      return
    for name, ctrl in self.model_ctrls.items():
      filepath = os.path.join(cfg.save_dir, name + '.torch')
      if os.path.exists(filepath):
        loaded = torch.load(filepath)
        self.step = loaded['step']
        self.monitor.best_value = loaded['record']
        ctrl.model.load_state_dict(loaded['model'])
        ctrl.optim.load_state_dict(loaded['optim'])
        ctrl.sched.load_state_dict(loaded['sched'])
        log.info(f'Loaded snapshot from: {filepath}')
        log.info(f'Resume from step {self.step}.')

    # filepath = os.path.join(cfg.save_dir, 'status.torch')
    # status = torch.load(filepath) if os.path.exists(filepath) else None
    # return status
