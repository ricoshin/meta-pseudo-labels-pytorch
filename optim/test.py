from tqdm import tqdm

from torch import nn

import data, optim, utils
from optim.metric import topk_accuracy, AverageMeter
from utils.debugger import getSignalCatcher

sigquit = getSignalCatcher('SIGQUIT')


def test(cfg, manager, mode='test'):
  assert isinstance(manager, optim.manager.TrainingManager)
  assert mode in ['valid', 'test']

  m = manager
  if mode == 'test':
    m.load_if_available(cfg, 'best')

  m.eval()
  result = AverageMeter(mode)
  for x, y in m.step_generator(mode):
    x, y = x.cuda(), y.cuda()
    # if sigquit.is_active():
    #   import pdb; pdb.set_trace()
    y_pred = m.stdn.model(x)

    acc_top1 = topk_accuracy(y_pred, y, (1,))
    result.add(top1=acc_top1, num=y.size(0))
    # acc_top1, acc_top5 = accuracy(y_pred, y, (1, 5))
    # result.add(top1=acc_top1, top5=acc_top5, num=y.size(0))

  return result
