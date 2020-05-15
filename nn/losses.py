import torch
import torch.nn.functional as F
from torch import nn


class SmoothCrossEntropyLoss(nn.Module):
  """Cross entropy loss that can deliver label smoothing effect.
   When smoothing factor equals to 0.0, it behaves excatly the same as the
   standard cross entropy loss."""
  def __init__(self, factor=0.0, dim=-1):
    super(SmoothCrossEntropyLoss, self).__init__()
    self.factor = factor
    self.dim = dim

  def forward(self, pred, target):
    if self.factor == 0.:
      return F.cross_entropy(pred, target)
    # pred: [batch_size, num_classes]
    # target: [batch_size]
    with torch.no_grad():
      num_classes = pred.size(-1)
      target_smoothed = torch.zeros_like(pred)
      target_smoothed.fill_(self.factor / num_classes)
      target_smoothed.scatter_(
        1, target.unsqueeze(1), 1 - self.factor)
      target = target_smoothed
    return torch.mean(torch.sum(
      -target * pred.log_softmax(self.dim), dim=self.dim))


class ConsistencyLoss(nn.Module):
  """Consistency loss for UDA."""
  def __init__(self, dim=-1):
    super(ConsistencyLoss, self).__init__()
    self.dim = dim

  def forward(self, tar, src):
    tar = F.softmax(tar, dim=1).detach()
    src = F.log_softmax(src, dim=1)
    return F.kl_div(src, tar, reduction='batchmean')
