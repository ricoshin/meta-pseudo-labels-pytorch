from torch import nn

import data, optim, utils
from optim.metric import accuracy, AverageMeter
from utils.debugger import getSignalCatcher


sigquit = getSignalCatcher('SIGQUIT')

def test(cfg, loaders, manager, writers, desc='test'):
  assert isinstance(loaders, data.loader.DataLoaderTriplet)
  assert isinstance(manager, optim.manager.TrainingManager)
  assert isinstance(writers, utils.tfwriter.TFWriters)

  m = manager
  m.eval()  # batchnorm? saved?
  # xent = nn.CrossEntropyLoss()
  result = AverageMeter(desc)

  for step, (x, y) in enumerate(loaders.test):
    x, y = x.cuda(), y.cuda()
    # if sigquit.is_active():
    #   import pdb; pdb.set_trace()
    y_pred = m.tchr.model(x)
    # loss = xent(y_pred, y)

    acc_top1, acc_top5 = accuracy(y_pred, y, (1, 5))
    result.add(top1=acc_top1, top5=acc_top5, num=y.size(0))

  return result
