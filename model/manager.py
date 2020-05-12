from model.loader import get_model, get_optimizer, get_scheduler


class ModelControl:
  def __init__(self, model, optim, sched):
    self.model = model
    self.optim = optim
    self.sched = sched

  def cuda_(self, *args, **kwargs):
    self.model = self.model.cuda(*args, **kwargs)

  def train(self, *args, **kwargs):
    self.model.train(*args, **kwargs)
    return self

  def eval(self, *args, **kwargs):
    self.model.eval(*args, **kwargs)
    return self


class ModelManager:
  def __init__(self, cfg, data_parallel=True):
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
    model = get_model(dropout=cfg.tchr.dropout, **model_kwargs)
    optim = get_optimizer(model=model, lr=cfg.tchr.lr, **optim_kwargs)
    sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
    self.tchr = ModelControl(model, optim, sched)
    self.model_ctrls['tchr'] = self.tchr

    if cfg.method.mpl:
      model = get_model(dropout=cfg.stdn.dropout, **model_kwargs)
      optim = get_optimizer(model=model, lr=cfg.stdn.lr, **optim_kwargs)
      sched = get_scheduler(optim, cfg.comm.n_steps, cfg.comm.n_warmup)
      self.stdn = ModelControl(model, optim, sched)
      self.model_ctrls['stdn'] = self.stdn

  def train(self, mode=True):
    for ctrl in self.model_ctrls.values():
      ctrl.train(mode)
    return self

  def eval(self, mode=True):
    for ctrl in self.model_ctrls.values():
      ctrl.eval(mode)
    return self

  def cuda_(self):
    for ctrl in self.model_ctrls.values():
      ctrl.cuda_()

  def save(self, cfg, status=None):
    if not cfg.save_dir:
      return
    for name, ctrl in self.model_ctrls.keys():
      filepath = os.path.join(cfg.save_dir, name + '.torch')
      torch.save({
        'model': ctrl.model.state_dict(),
        'optim': ctrl.optim.state_dict(),
      }, filepath)
      log.info(f'Saved model to: {filepath}')
    if status:
      filepath = os.path.join(cfg.save_dir, 'status.torch')
      torch.save(status, filepath)
      log.info(f'Saved status to: {filepath}')

  def load_if_available(self, cfg):
    if not cfg.save_dir:
      return
    for name, ctrl in self.model_ctrls.keys():
      filepath = os.path.join(cfg.save_dir, name + '.torch')
      if os.path.exists(filepath):
        loaded = torch.load(filepath)
        ctrl.model.load_state_dict(loaded['model'])
        ctrl.optim.load_state_dict(loaded['optim'])
        log.info(f'Loaded model from: {filepath}')
    filepath = os.path.join(cfg.save_dir, 'status.torch')
    status = torch.load(filepath) if os.path.exists(filepath) else None
    return status
