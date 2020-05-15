import logging

import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax

import optim
from nn.losses import SmoothCrossEntropyLoss, ConsistencyLoss
from optim.metric import topk_accuracy, AverageMeter
from optim.test import test
from utils import concat
from utils.config import Config
from utils.debugger import getSignalCatcher

log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_format(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


def train(cfg, manager):
  assert isinstance(manager, optim.manager.TrainingManager)

  m = manager  # for brevity
  m.load_if_available(cfg, tag='last', verbose=True)
  m.train()

  result_train = AverageMeter('train')
  supervised_loss = SmoothCrossEntropyLoss(factor=cfg.optim.lb_smooth)
  consistency_loss = ConsistencyLoss()

  is_uda = cfg.method.base == 'uda'
  is_mpl = cfg.method.mpl

  if m.is_finished:
    log.info('No remaining steps.')
    # return empty results
    return None, None

  for step in m.step_generator(mode='train'):
    # initialize
    m.train()
    result_train.init()
    if is_mpl:
      # TODO
      pass

    if not is_uda:
      xs, ys = next(m.loaders.sup)
      xs, ys = xs.cuda(), ys.cuda()
      ys_pred = m.tchr.model(x)  # forward
      loss_total = loss_sup = supervised_loss(ys_pred, ys)
    else:
      xs, ys = next(m.loaders.sup)
      xu, xuh = next(m.loaders.uns)
      x, split_back = concat([xs, xu, xuh], retriever=True)
      x, ys = x.cuda(), ys.cuda()
      y_pred = m.tchr.model(x)  # forward
      ys_pred, yu_pred, yuh_pred = split_back(y_pred)
      loss_sup = supervised_loss(ys_pred, ys)
      loss_cnst = consistency_loss(yu_pred, yuh_pred)
      loss_total = loss_sup + loss_cnst * cfg.uda.factor

    loss_total.backward()
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
