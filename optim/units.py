import logging

from optim.metric import topk_accuracy, AverageMeter
from utils import concat
from utils.debugger import getSignalCatcher

log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_fmt(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


def train(cfg, iterable, ctrl_a, ctrl_b, loaders, losses):
  avgmeter = AverageMeter('train')
  for step in iterable:
    # labeled(supervised) data
    xs, ys = next(loaders.sup)
    xs, ys = xs.cuda(), ys.cuda()
    if cfg.method.is_uda or cfg.method.is_mpl:
      # unlabeled(unsupervised) data
      xu, xuh = next(loaders.uns)
      xu, xuh = xu.cuda(), xuh.cuda()

    # supervised, randaugment
    if not cfg.method.is_uda:
      ys_pred_a = ctrl_a.model(xs)  # forward
      loss_total = loss_sup = losses.smooth_supervised(ys_pred_a, ys)
    # uda
    else:
      # model feeding
      x, split_back = concat([xs, xu, xuh], retriever=True)
      y_pred_a = ctrl_a.model(x)  # forward
      ys_pred_a, yu_pred_a, yuh_pred_a = split_back(y_pred_a)
      # loss computation
      loss_sup = losses.uda_supervised(ys_pred_a, ys, step)
      loss_cnst = losses.uda_consistency(yuh_pred_a, yu_pred_a.detach())
      loss_total = loss_sup + loss_cnst * cfg.uda.factor

    # meta peudo labels
    if cfg.method.is_mpl:
      # ctrl_a: teacher, ctrl_b: student
      yu_pred_b = ctrl_b.model(xu)
      if not is_uda:
        yu_pred_a = ctrl_a.model(xu)
      loss_mpl_b = losses.mpl_student(yu_pred_b, yu_pred_a)
      loss_mpl_b.backward(retain_graph=True, create_graph=True)
      ctrl_b.step_all(clip_grad=cfg.optim.clip_grad)
      ctrl_a.model.zero_grad()  # teacher should not be affected

      xs, ys = next(loaders.sup)
      xs, ys = xs.cuda(), ys.cuda()  # to see different xs
      ys_pred_b = ctrl_b.model(xs)
      loss_mpl_a = losses.mpl_teacher(ys_pred_b, ys)
      loss_total += loss_mpl_a

    loss_total.backward()
    ctrl_a.step_all(clip_grad=cfg.optim.clip_grad)
    if ctrl_b:
      ctrl_b.model.zero_grad()

    ys_pred = ys_pred_a if not cfg.method.is_mpl else ys_pred_b
    acc_top1 = topk_accuracy(ys_pred, ys, (1,))
    avgmeter.add(top1=acc_top1, num=ys.size(0))
  return avgmeter


def test(iterable, ctrl, loaders, mode):
  assert mode in ['valid', 'test']
  avgmeter = AverageMeter(mode)
  for x, y in iterable:
    x, y = x.cuda(), y.cuda()
    y_pred = ctrl.model(x)
    acc_top1 = topk_accuracy(y_pred, y, (1,))
    avgmeter.add(top1=acc_top1, num=y.size(0))
  return avgmeter
