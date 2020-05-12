# from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler




def train(cfg, loaders, models, writers):
  models.load_if_available(cfg)
  models.train()
  for step in range(cfg.comm.n_steps):
    import pdb; pdb.set_trace()
    #TODO


def get_scheduler(optimizer, n_steps, n_warmup):
  scheduler = CosineAnnealingLR(optimizer, T_max=n_steps-n_warmup)
  scheduler = GradualWarmupScheduler(
    optimizer, multiplier=1., total_epoch=n_warmup, after_scheduler=scheduler)
  return scheduler
