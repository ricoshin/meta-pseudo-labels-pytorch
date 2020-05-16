import logging

import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax

import optim
from nn.losses import (SmoothableHardLabelCELoss, SoftLabelCELoss,
                       KLDivergenceLoss)
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

  is_uda = cfg.method.base == 'uda'
  is_mpl = cfg.method.mpl

  out_train = AverageMeter('train')
  supervised_loss = SmoothableHardLabelCELoss(factor=cfg.optim.lb_smooth)
  hard_xent_loss = SmoothableHardLabelCELoss(factor=0.)
  consistency_loss = KLDivergenceLoss() if is_uda else None
  soft_xent_loss = SoftLabelCELoss() if is_mpl else None

  if m.is_finished:
    log.info('No remaining steps.')
    # return empty results
    return None, None

  for step in m.step_generator(mode='train'):
    # initialize
    m.train()
    out_train.init()
    xs, ys = next(m.loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()
    if is_uda or is_mpl:
      xu, xuh = next(m.loaders.uns)
      xu, xuh = xu.cuda(), xuh.cuda()
    else:
      xu, xuh = None, None

    if not is_uda:
      ys_pred_t = m.tchr.model(xs)  # forward
      loss_total = loss_sup = supervised_loss(ys_pred_t, ys)
    else:
      x, split_back = concat([xs, xu, xuh], retriever=True)
      y_pred = m.tchr.model(x)  # forward
      ys_pred_t, yu_pred_t, yuh_pred_t = split_back(y_pred)
      loss_sup = supervised_loss(ys_pred_t, ys)
      loss_cnst = consistency_loss(yuh_pred_t, yu_pred_t.detach())
      loss_total = loss_sup + loss_cnst * cfg.uda.factor

    if is_mpl:
      yu_pred_s = m.stdn.model(xu)
      if not is_uda:
        yu_pred_t = m.tchr.model(xu)
      loss_mpl_s = soft_xent_loss(yu_pred_s, yu_pred_t)
      loss_mpl_s.backward(retain_graph=True, create_graph=True)
      m.stdn.step_all()

      ys_pred_s = m.stdn.model(xs)
      loss_mpl_t = hard_xent_loss(ys_pred_s, ys)
      loss_total += loss_mpl_t

    loss_total.backward()
    m.tchr.step_all()

    ys_pred = ys_pred_t if not is_mpl else ys_pred_s  # stnt acc when mpl.
    acc_top1 = topk_accuracy(ys_pred, ys, (1,))
    out_train.add(top1=acc_top1, num=ys.size(0))
    # acc_top1, acc_top5 = topk_accuracy(ys_pred, ys, (1, 5))
    # out_train.add(top1=acc_top1, top5=acc_top5, num=ys.size(0))

    if not (m.is_valid_step or m.is_finished):
      continue

    # validation
    out_valid = test(cfg, manager, 'valid')
    m.monitor.watch(out_valid[cfg.valid.metric])

    with m.logging():
      m.save(cfg, tag='last')
      if m.monitor.is_best:
        m.save(cfg, tag='best')
        mark_record = ' [<--record]'
      log.info(_to_format(m, out_train, out_valid, m.monitor) + mark_record)

    m.writers.add_scalars('train', out_train.to_dict(), step)
    m.writers.add_scalars('valid', out_valid.to_dict(), step)
    m.writers.add_scalars('train', {'lr': m.tchr.sched.get_lr()[0]}, step)

  return out_train, out_valid
