import logging

from ray import tune
import torch
from torch import nn
from torch.nn.functional import kl_div, softmax, log_softmax

from nn.losses import (LabelSmoothableCELoss, TrainingSignalAnnealingCELoss,
                       ConsistencyKLDLoss, SoftLabelCEWithLogitsLoss)
import optim
from optim.metric import topk_accuracy, AverageMeter
from optim.test import test
from utils import concat
from utils.config import Config
from utils.debugger import getSignalCatcher

log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_fmt(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


def train(cfg, manager, tuning=False):
  assert isinstance(manager, optim.manager.TrainingManager)
  # for brevity
  m = manager
  is_uda = cfg.method.base == 'uda'
  is_mpl = cfg.method.mpl
  metric = cfg.valid.metric
  # load model if possible
  m.load_if_available(cfg, tag='last', verbose=True)
  if m.is_finished:
    log.info('No remaining steps.')
    # return empty results
    return None, None
  # baseline: netA only / mpl: netA(teacher), netB(student)
  if cfg.method.mpl:
    netA, netB = m.tchr, m.stdn
  else:
    netA, netB = m.stdn, None
  # losses
  smooth_supervised_loss = LabelSmoothableCELoss(factor=cfg.optim.lb_smooth)
  uda_supervised_loss = TrainingSignalAnnealingCELoss(cfg) if is_uda else None
  uda_consistency_loss = ConsistencyKLDLoss(
    confid_threshold=cfg.uda.confid_threshold,
    softmax_temp=cfg.uda.softmax_temp,
    ) if is_uda else None
  mpl_stdn_loss = SoftLabelCEWithLogitsLoss() if is_mpl else None
  mpl_tchr_loss = LabelSmoothableCELoss(factor=0.) if is_mpl else None
  # average tracker
  out_train = AverageMeter('train')

  for step in m.step_generator(mode='train', disable_pbar=tuning):
    m.train()
    # out_train.init()  # uncomment this if you want to see only the last metric

    # labeled(supervised) data
    xs, ys = next(m.loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()
    if is_uda or is_mpl:
      # unlabeled(unsupervised) data
      xu, xuh = next(m.loaders.uns)
      xu, xuh = xu.cuda(), xuh.cuda()

    # supervised, randaugment
    if not is_uda:
      assert cfg.method.base in ['sup', 'ra']
      ys_pred_a = netA.model(xs)  # forward
      loss_total = loss_sup = smooth_supervised_loss(ys_pred_a, ys)
    # uda
    else:
      # model feeding
      x, split_back = concat([xs, xu, xuh], retriever=True)
      y_pred_a = netA.model(x)  # forward
      ys_pred_a, yu_pred_a, yuh_pred_a = split_back(y_pred_a)
      # loss computation
      loss_sup = uda_supervised_loss(ys_pred_a, ys, step)
      loss_cnst = uda_consistency_loss(yuh_pred_a, yu_pred_a.detach())
      loss_total = loss_sup + loss_cnst * cfg.uda.factor

    # meta peudo labels
    if is_mpl:
      # netA: teacher, netB: student
      yu_pred_b = netB.model(xu)
      if not is_uda:
        yu_pred_a = netA.model(xu)
      loss_mpl_b = mpl_stdn_loss(yu_pred_b, yu_pred_a)
      loss_mpl_b.backward(retain_graph=True, create_graph=True)
      netB.step_all(clip_grad=cfg.optim.clip_grad)
      netA.model.zero_grad()  # teacher should not be affected

      xs, ys = next(m.loaders.sup)
      xs, ys = xs.cuda(), ys.cuda()  # to see different xs
      ys_pred_b = netB.model(xs)
      loss_mpl_a = mpl_tchr_loss(ys_pred_b, ys)
      loss_total += loss_mpl_a

    loss_total.backward()
    netA.step_all(clip_grad=cfg.optim.clip_grad)
    if netB:
      netB.model.zero_grad()

    ys_pred = ys_pred_a if not is_mpl else ys_pred_b
    acc_top1 = topk_accuracy(ys_pred, ys, (1,))
    out_train.add(top1=acc_top1, num=ys.size(0))

    if not (m.is_valid_step or m.is_finished):
      continue

    # valid
    out_valid = test(cfg, manager, 'valid', tuning)
    m.monitor.watch(out_valid[metric])
    if tuning:
      tune.report({metric: out_valid[metric]})

    # save & log
    with m.logging():
      m.save(cfg, tag='last')
      if m.monitor.is_best:
        m.save(cfg, tag='best')
      mark_record = ' [<--record]' if m.monitor.is_best else ''
      log.info(_to_fmt(m, out_train, out_valid, m.monitor) + mark_record)

    # tensorboard
    m.writers.add_scalars('train', out_train.to_dict(), step)
    m.writers.add_scalars('valid', out_valid.to_dict(), step)
    m.writers.add_scalars('train', {'lr': m.stdn.sched.get_lr()[0]}, step)
    out_train.init()

  return out_train, out_valid
