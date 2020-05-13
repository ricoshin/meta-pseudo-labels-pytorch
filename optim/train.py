import logging

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from optim.metric import accuracy, AverageMeter
from optim.test import test
from utils.debugger import getSignalCatcher
from utils.config import Config

log = logging.getLogger('mpl')
sigquit = getSignalCatcher('SIGQUIT')


def train(cfg, loaders, manager, writers, desc='train'):
  m = manager  # for brevity
  m.load_if_available(cfg)
  m.train()
  xent = nn.CrossEntropyLoss()
  result_tr = AverageMeter(desc)

  for step in m.step_generator():
    # supervised
    xs, ys = next(loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()
    # if step == 100:
    #   import pdb; pdb.set_trace()

    ys_pred = m.tchr.model(xs)

    loss = xent(ys_pred, ys)
    loss.backward()

    m.tchr.optim.step()
    m.tchr.optim.zero_grad()
    m.tchr.sched.step()

    acc_top1, acc_top5 = accuracy(ys_pred, ys, (1, 5))
    result_tr.add(top1=acc_top1, top5=acc_top5, num=ys.size(0))

    if not step % cfg.log.interval == 0:
      continue

    result_te = test(cfg, loaders, manager, writers)
    # logging
    log.info(' | '.join(map(str, [m, result_tr, result_te])))
    # log.info(
    #   f'train | {step:7d}/{cfg.comm.n_steps:7d} | '
    #   f'base: {cfg.method.base} | mpl: {cfg.method.mpl}'
    #   f'top1: {acc_top1:5.2f}, top5: {acc_top5:5.2f} | '
    #   f'lr_t: {m.tchr.sched.get_lr()[0]:6.4f} | '
    #   f'lr_s: {m.stdn.sched.get_lr()[0]:6.4f}'
    #   )
    m.train()
    result_tr.init()
