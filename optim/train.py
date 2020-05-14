import logging

from torch import nn

import optim
from nn.label_smooth import LabelSmoothingCE
from optim.metric import topk_accuracy, AverageMeter
from optim.test import test
from utils.config import Config
from utils.debugger import getSignalCatcher


log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_format(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


def train(cfg, manager):
  assert isinstance(manager, optim.manager.TrainingManager)

  xent = LabelSmoothingCE(smoothing=cfg.optim.lb_smooth)
  result_train = AverageMeter('train')
  # monitor = MetricMonitor.by_metric(cfg.valid.metric)

  m = manager  # for brevity
  m.load_if_available(cfg, tag='last', verbose=True)
  m.train()

  if m.is_finished:
    log.info('No remaining steps.')
    # return empty results
    return None, None

  for step in m.step_generator(mode='train'):
    m.train()
    result_train.init()
    # supervised
    xs, ys = next(m.loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()

    ys_pred = m.tchr.model(xs)

    loss = xent(ys_pred, ys)
    loss.backward()
    # import pdb; pdb.set_trace()
    m.tchr.optim.step()
    m.tchr.optim.zero_grad()
    m.tchr.sched.step()

    acc_top1 = topk_accuracy(ys_pred, ys, (1,))
    result_train.add(top1=acc_top1, num=ys.size(0))
    # acc_top1, acc_top5 = topk_accuracy(ys_pred, ys, (1, 5))
    # result_train.add(top1=acc_top1, top5=acc_top5, num=ys.size(0))

    if not (m.is_valid_step or m.is_finished):
      continue

    # validation
    result_valid = test(cfg, manager, 'valid')

    with m.logging():
      m.save(cfg, tag='last')
      if m.monitor.is_best(result_valid[cfg.valid.metric]):
        m.save(cfg, tag='best', verbose=True)
      log.info(_to_format(m, result_train, result_valid, m.monitor))

    m.writers.add_scalars('train', result_train.to_dict(), step)
    m.writers.add_scalars('valid', result_valid.to_dict(), step)
    m.writers.add_scalars('train', {'lr': m.tchr.sched.get_lr()[0]}, step)

  return result_train, result_valid
