class TrainingSignalAnnealing:
  def __init__(self, schedule, step_max, num_classes=None):
    assert schedule in ['', 'linear', 'log', 'exp']
    self.schedule = schedule
    self.step_max = step_max
    self.num_classes = num_classes

  def __call__(self, probs, step):
    if schedule == '':
      return probs
    if self.num_classes:
      assert probs.size(-1) == self.num_classes
    else:
      self.num_classes = probs.size(-1)
    thresh_start = 1. / self.num_classes
    step_ratio = torch.tensor(torch.step / self.step_max).to(probs.device)
    if schedule == "linear":
      alpha = step_ratio
    elif schedule == "exp":
      scale = 5
      # [exp(-5), exp(0)] = [1e-2, 1]
      alpha = torch.exp((step_ratio - 1) * scale)
    elif schedule == "log":
      scale = 5
      # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
      alpha = 1 - torch.exp((-step_ratio) * scale)
    thresh = alpha * (1 - thresh_start) + thresh_start
