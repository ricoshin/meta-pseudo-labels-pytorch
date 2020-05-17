import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothableCELoss(nn.Module):
  """Cross entropy loss that cna deliver label smoothing effect, if wanted.
   As with torch.nn.CrossEntropyLoss, it takes integer labels as a target."""
  def __init__(self, factor=0.0, dim=-1):
    super(LabelSmoothableCELoss, self).__init__()
    self.factor = factor
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size]
    if self.factor == 0.:
      return F.cross_entropy(input, target, reduction='mean')
    with torch.no_grad():
      num_classes = input.size(-1)
      target_smoothed = torch.zeros_like(input)
      target_smoothed.fill_(self.factor / num_classes)
      target_smoothed.scatter_(
        1, target.unsqueeze(1), 1 - self.factor)
      target = target_smoothed
    return torch.mean(torch.sum(
      -target * input.log_softmax(self.dim), dim=self.dim))


class TrainingSignalAnnealingCELoss(nn.Module):
  """Cross entropy loss that abides by TSA scheduling(for UDA)."""
  def __init__(self, cfg, num_classes=None):
    super(TrainingSignalAnnealingCELoss, self).__init__()
    assert cfg.uda.tsa_schedule in ['', 'linear', 'log', 'exp']
    self.schedule = cfg.uda.tsa_schedule
    self.step_max = cfg.comm.n_steps
    self.num_classes = num_classes

  def _get_threshold(self, step):
    scale = 5
    thresh_start = 1. / self.num_classes
    step_ratio = torch.tensor(step / self.step_max)
    if self.schedule == "linear":
      alpha = step_ratio
    elif self.schedule == "exp":
      alpha = torch.exp((step_ratio - 1) * scale)
    elif self.schedule == "log":
      alpha = 1 - torch.exp((-step_ratio) * scale)
    thresh = alpha * (1 - thresh_start) + thresh_start
    return thresh

  def forward(self, input, target, step):
    # input(logits): [batch_size, num_classes]
    # target: [batch_size]
    if self.num_classes:
      assert input.size(-1) == self.num_classes
    else:
      self.num_classes = input.size(-1)
    loss = F.cross_entropy(input, target, reduction='none')
    target_one_hot = F.one_hot(target, self.num_classes)
    input_probs = F.softmax(input, dim=-1)
    threshold = self._get_threshold(step).to(input.device)
    correct_label_probs = torch.sum(target_one_hot * input_probs, dim=-1)
    greater_than_threshold = correct_label_probs > threshold
    loss_mask = 1 - greater_than_threshold.float()  # abandon too confident ones
    loss = loss * loss_mask.detach()
    loss = loss.sum() / loss_mask.sum()
    return loss


class ConsistencyKLDLoss(nn.Module):
  """KL divergence between two logits with sharpening predictions
    techniques. (for UDA)"""
  def __init__(self, confid_threshold=0., softmax_temp=1., dim=-1):
    super(ConsistencyKLDLoss, self).__init__()
    self.softmax_temp = softmax_temp
    self.confid_threshold = confid_threshold
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size, num_classes]
    if self.softmax_temp > 1.:
      raise Exception('softmax_temp is expected to be lower than or '
        f'equal to 1. (given: {self.softmax_temp})')
    elif self.softmax_temp < 1.:
      if self.softmax_temp < 1e-2:
        raise Exception(f'too low softmax_temp: {self.softmax_temp}')
      target_sharp = target / self.softmax_temp

    target_sharp = F.softmax(target_sharp, self.dim)
    input = F.log_softmax(input, self.dim)
    loss = F.kl_div(input, target_sharp, reduction='none')
    loss = loss.sum(dim=-1)

    target = F.softmax(target, self.dim)
    target_max = target.max(dim=-1).values
    loss_mask = target_max > self.confid_threshold  # train confident ones only
    loss = loss * loss_mask.detach()
    loss = loss.mean()  # they didn't do loss.sum() / loss_mask.sum(),
    return loss         # perhaps because all-zero masks occur quite often.


class SoftLabelCEWithLogitsLoss(nn.Module):
  """Cross entropy between two continuous logits.
    (for student training in MLP)"""
  def __init__(self, dim=-1):
    super(SoftLabelCEWithLogitsLoss, self).__init__()
    self.dim = dim

  def forward(self, input, target):
    # input: [batch_size, num_classes]
    # target: [batch_size, num_classes]
    target = F.softmax(target, self.dim)
    input = F.log_softmax(input, self.dim)
    return torch.mean(torch.sum(-target * input, self.dim))
