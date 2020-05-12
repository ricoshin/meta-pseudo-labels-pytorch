import logging

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from optim.helper import accuracy

log = logging.getLogger('mpl')


def train(cfg, loaders, models, writers):
  models.load_if_available(cfg)
  models.train()
  xent = nn.CrossEntropyLoss()
  for step in range(cfg.comm.n_steps):
    # supervised
    xs, ys = next(loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()

    ys_pred = models.tchr(xs)
    loss = xent(ys_pred, ys)
    loss.backward()
    models.tchr_optim.step()
    models.tchr_optim.zero_grad()

    acc_top1, acc_top5 = accuracy(ys_pred, ys, (1,5))

    if not step % cfg.log.interval == 0:
      continue
    # logging
    log.info(
      f'[train: {step:7d}/{cfg.comm.n_steps:7d}] '
      f'acc_top1: {acc_top1:5.2f} / acc_top5: {acc_top5:5.2f}')


def get_scheduler(optimizer, n_steps, n_warmup):
  scheduler = CosineAnnealingLR(optimizer, T_max=n_steps-n_warmup)
  scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1., total_epoch=n_warmup, after_scheduler=scheduler)
  return scheduler
