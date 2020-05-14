import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingCE(nn.Module):
  def __init__(self, smoothing=0.0, dim=-1):
    super(LabelSmoothingCE, self).__init__()
    self.smoothing = smoothing
    self.dim = dim

  def forward(self, pred, target):
    if self.smoothing == 0.:
      return F.cross_entropy(pred, target)
    # pred: [batch_size, num_classes]
    # target: [batch_size]
    with torch.no_grad():
      num_classes = pred.size(-1)
      target_smoothed = torch.zeros_like(pred)
      target_smoothed.fill_(self.smoothing / num_classes)
      target_smoothed.scatter_(
        1, target.unsqueeze(1), 1 - self.smoothing)
      target = target_smoothed
    return torch.mean(torch.sum(
      -target * pred.log_softmax(self.dim), dim=self.dim))
