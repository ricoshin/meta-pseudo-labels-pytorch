from collections import defaultdict
from collections.abc import Iterable
from torch._six import container_abcs

import torch
from copy import deepcopy
from itertools import chain


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class ModuleOptimizer(torch.optim.Optimizer):
  def __init__(self, modules, defaults):
    R"""initial params example:
    optim = ModuleOptimizer([{'module': model.base},
                             {'module': model.classifier, 'lr': 1e-3}
                            ], lr=1e-2, momentum=0.9)
    """
    torch._C._log_api_usage_once("python.optimizer")
    self.modules = []
    self.defaults = defaults
    self.state = defaultdict(dict)
    self.param_groups = []

    if isinstance(modules, torch.nn.Module):
      modules = [modules]

    param_groups = list(modules)
    if len(param_groups) == 0:
      raise ValueError("optimizer got an empty module list.")

    if not isinstance(param_groups[0], dict):
      param_groups = [{'module': module} for module in param_groups]

    for param_group in param_groups:
      self.add_param_group(param_group)

  def __getstate__(self):
    return {
        'defaults': self.defaults,
        'state': self.state,
        'param_groups': self.param_groups,
    }

  def __setstate__(self, state):
    assert isinstance(state, dict)
    self.__dict__.update(state)

  def state_dict(self):
      return {
        'state': self.state,
        'param_groups': self.param_groups,
      }

  def load_state_dict(self, state_dict):
    state_dict = deepcopy(state_dict)
    if len(self.param_groups) != len(state_dict['param_groups']):
      raise ValueError("loaded state dict has a different number of "
                       "parameter groups")
    self.__setstate__({'state': state_dict['state'],
                       'param_groups': state_dict['param_groups']})

  def zero_grad(self):
    for module in self.modules:
      for param in module.parameters():
        if param.grad is not None:
          param.grad = None

  def add_param_group(self, param_group):
    assert isinstance(param_group, dict), "param group must be a dict"
    assert 'module' in param_group, "param group must have a key 'module'"
    assert isinstance(param_group['module'], torch.nn.Module)

    # handling module instead of parameters
    import weakref
    self.modules.append(weakref.ref(param_group['module'])())
    del param_group['module']

    # other settings
    for name, default in self.defaults.items():
      if default is required and name not in param_group:
        raise ValueError("parameter group didn't specify a value of "
                         "required optimization parameter " + name)
      else:
        # overwrite default settings if not provided
        param_group.setdefault(name, default)
    self.param_groups.append(param_group)

  def step(self, colure):
    # define in subclasses
    raise NotImplementedError
