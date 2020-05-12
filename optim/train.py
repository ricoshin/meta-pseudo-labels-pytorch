import logging

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from optim.helper import accuracy

log = logging.getLogger('mpl')


def train(cfg, loaders, manager, writers):
  m = manager  # for brevity
  m.load_if_available(cfg)
  m.train()
  xent = nn.CrossEntropyLoss()
  for step in range(cfg.comm.n_steps):
    # supervised
    xs, ys = next(loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()

    ys_pred = m.tchr.model(xs)
    loss = xent(ys_pred, ys)
    loss.backward()
    m.tchr.optim.step()
    m.tchr.optim.zero_grad()
    m.tchr.sched.step()

    acc_top1, acc_top5 = accuracy(ys_pred, ys, (1,5))

    if not step % cfg.log.interval == 0:
      continue
    # logging
    log.info(
      f'train | {step:7d}/{cfg.comm.n_steps:7d} | '
      f'acc_top1: {acc_top1:5.2f} | acc_top5: {acc_top5:5.2f} | '
      f'lr_t: {m.tchr.sched.get_lr()[0]:6.4f} | '
      f'lr_s: {m.stdn.sched.get_lr()[0]:6.4f}'
      )
