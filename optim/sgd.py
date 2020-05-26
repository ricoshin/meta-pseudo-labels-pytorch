import torch

from optim.optimizer import ModuleOptimizer, required
from utils.gpu_profile import GPUProfiler


def _update_params(module, names, tensor, _global_name=None):
  """This function updates the submodule's parameters with the 3rd argument
  tensor according to its dot-sperated name in a recursive manner.
  """
  assert isinstance(names, str) or isinstance(names, (tuple, list))
  assert isinstance(tensor, torch.Tensor)
  if _global_name is None:
    _global_name = names
  if isinstance(names, str):
    names = names.split('.')
  if len(names) > 1:
    module_name, names = names[0], names[1:]
    module = module._modules[module_name]
    _update_params(module, names, tensor, _global_name)
  else:
    param_name = names[0]
    # just for gpu debugging
    if hasattr(module._parameters[param_name], '_head_str'):
      tensor._head_str = module._parameters[param_name]._head_str
      assert hasattr(module._parameters[param_name], '_info_str')
      tensor._info_str = module._parameters[param_name]._info_str
    if hasattr(module._parameters[param_name], '_id_origin'):
      tensor._id_origin = module._parameters[param_name]._id_origin
    else:
      tensor._id_origin = id(module._parameters[param_name])
    # must call gradient __del__ first to prevent memory leak
    module._parameters[param_name].grad.grad = None
    module._parameters[param_name].grad = None
    module._parameters[param_name] = tensor


class ModuleSGD(ModuleOptimizer):
  """With the standard PyTorch APIs, model parameters cannot keep tracking
  grad_fn since nn.Parameter is not meant to be non-leaf tensor. To bypass
  this, we use a simple trick where torch.nn.Parameter of the module is
  switched over to torch.Tensor. Thus, it can retain computational graphs for
  second order gradients while not losing compatibility with torch.nn.Module
  significantly. Note that some functions would fail after using this trick,
  e.g.nn.Module.apply(), which can be cured easily if needed.
  """
  def __init__(self, modules, lr=required, momentum=0, dampening=0,
               weight_decay=0, nesterov=False):
    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                    weight_decay=weight_decay, nesterov=nesterov)
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov momentum requires a momentum and "
                       "zero dampening")
    super(ModuleSGD, self).__init__(modules, defaults)

  def __setstate__(self, state):
      super(ModuleSGD, self).__setstate__(state)
      for group in self.param_groups:
          group.setdefault('nesterov', False)

  def step(self, closure=None, debug=False):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for i, (module, group) in enumerate(zip(self.modules, self.param_groups)):
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']

      for n, p in module.named_parameters():
        if p.grad is None:
            continue
        second_order = True if p.grad.grad_fn else False
        # torch.set_grad_enabled(second_order)
        g = p.grad
        # with torch.no_grad():
        if weight_decay != 0:
          g = g.add(p.detach(), alpha=weight_decay)  # drop factor 2
        if momentum != 0:
          # Use the names rather than parameter objects as keys,
          # since their ids can be varied at every update in our version.
          param_state = self.state[f"module_{i}.{n}"]
          if 'momentum_buffer' not in param_state:
            # very first momentum
            v = g.detach()
          else:
            # detach momentum history vector to avoid BPTT
            # gradient should be alive here to make it work with vanila SGD.
            # which could be different from original paper, apparently.
            v = param_state['momentum_buffer']
            v = momentum * v.detach() + (1 - dampening) * (-group['lr']) * g
          param_state['momentum_buffer'] = v.detach()
        if nesterov:
          d_p = -group['lr'] * g + momentum * v.detach()
        else:
          d_p = v
        if second_order:
          p = p.detach() + d_p
          _update_params(module, n, p)
        else:
          with torch.no_grad():
            p.add_(d_p.detach())
    return loss
