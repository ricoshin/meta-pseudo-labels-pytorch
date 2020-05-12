from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler


# _optim_map = {
#   'sgd': 'SGD',
#   'rmsprop': 'RMSprop',
#   'adagrad': 'Adagrad',
#   'adam': 'Adam',
# }
#
# def get_scheduler(optimizer, n_steps, n_warmup):
#   scheduler = CosineAnnealingLR(optimizer, T_max=n_steps-n_warmup)
#   scheduler = GradualWarmupScheduler(
#     optimizer, multiplier=1., total_epoch=n_warmup, after_scheduler=scheduler)
#   return scheduler
#
#
# def get_optimizer(optim_name, model, **kwargs):
#   assert optim_name in _optim_map, f'Unknown optimizer name: {optim_name}'
#   optim_cls = getattr(optim, _optim_map[optim_name])
#   optimizer = optim_cls(model.parameters(), **kwargs)
#   return optimizer
