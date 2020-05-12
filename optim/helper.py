import torch


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  assert isinstance(output, torch.Tensor)  # [batch_size, n_classes]
  assert isinstance(target, torch.Tensor)  # [batch_size]

  maxk = max(topk)
  batch_size = target.size(0)

  pred_topk = output.topk(k=maxk, dim=1, largest=True, sorted=True)
  pred_topk = pred_topk.indices.t()  # [maxk, batch_size] : top-k predictions
  correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(1. / batch_size))
  return res
