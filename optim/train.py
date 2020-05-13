import logging

from torch import nn

import data, optim, utils
from nn.label_smooth import LabelSmoothingCE
from optim.metric import accuracy, AverageMeter
from optim.test import test
from utils.config import Config
from utils.debugger import getSignalCatcher


log = logging.getLogger('mpl')
sigquit = getSignalCatcher('SIGQUIT')


def train(cfg, loaders, manager, writers, desc='train'):
  assert isinstance(loaders, data.dataloaders.DataLoaderTriplet)
  assert isinstance(manager, optim.manager.TrainingManager)
  assert isinstance(writers, utils.tfwriter.TFWriters)

  xent = LabelSmoothingCE(smoothing=cfg.optim.lb_smooth)
  result_train = AverageMeter(desc)
  # monitor = MetricMonitor.by_metric(cfg.valid.metric)

  m = manager  # for brevity
  m.load_if_available(cfg)
  m.train()

  for step in m.step_generator(pbar=True):
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
    result_train.add(top1=acc_top1, top5=acc_top5, num=ys.size(0))

    if not step % cfg.valid.interval == 0:
      continue

    # validation
    result_valid = test(cfg, loaders, manager, writers)

    # log
    m.log(m, result_train, result_valid, m.monitor)
    writers.add_scalars('train', result_train.to_dict(), step)
    writers.add_scalars('valid', result_valid.to_dict(), step)
    writers.add_scalars('train', {'lr': m.tchr.sched.get_lr()[0]}, step)

    if m.monitor.is_best(result_valid[cfg.valid.metric]):
      m.save(cfg)

    # initialize for training
    result_train.init()
    m.train()
