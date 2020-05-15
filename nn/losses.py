import torch
import torch.nn.functional as F
from torch import nn


class SmoothableHardLabelCELoss(nn.Module):
  """Cross entropy loss that can deliver label smoothing effect. As with torch.nn.CrossEntropyLoss, it take hard integer labels as a target."""
  def __init__(self, factor=0.0, dim=-1):
    super(SmoothableHardLabelCELoss, self).__init__()
    self.factor = factor
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size]
    if self.factor == 0.:
      return F.cross_entropy(input, target)
    with torch.no_grad():
      num_classes = input.size(-1)
      target_smoothed = torch.zeros_like(input)
      target_smoothed.fill_(self.factor / num_classes)
      target_smoothed.scatter_(
        1, target.unsqueeze(1), 1 - self.factor)
      target = target_smoothed
    return torch.mean(torch.sum(
      -target * input.log_softmax(self.dim), dim=self.dim))


class SoftLabelCELoss(nn.Module):
  """To compute cross entropy between two continuous logits(for MLP loss)."""
  def __init__(self, dim=-1):
    super(SoftLabelCELoss, self).__init__()
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size, num_classes]
    target = F.softmax(target, self.dim)
    input = F.log_softmax(input, self.dim)
    return torch.mean(torch.sum(-target * input, self.dim))


class KLDivergenceLoss(nn.Module):
  """To compute kl divergence between two logits."""
  def __init__(self, dim=-1):
    super(KLDivergenceLoss, self).__init__()
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size, num_classes]
    target = F.softmax(target, self.dim)
    input = F.log_softmax(input, self.dim)
    return F.kl_div(input, target, reduction='batchmean')
