import torch


def _update_params(module, names, tensor):
  """This function updates the submodule's parameter with the 3rd argument
  tensor according to its dot-sperated name in a recursive manner.
  """
  assert isinstance(names, str) or isinstance(names, (tuple, list))
  assert isinstance(tensor, torch.Tensor)
  if isinstance(names, str):
    names = names.split('.')
  if len(names) > 1:
    module_name, names = names[0], names[1:]
    module = module._modules[module_name]
    _update_params(module, names, tensor)
  else:
    param_name = names[0]
    module._parameters[param_name] = tensor


class ModuleSGD(torch.optim.SGD):
  """With the standard PyTorch APIs, model parameters cannot keep tracking
  grad_fn since nn.Parameter is not meant to be non-leaf tensor. To bypass
  this, we use a simple trick where torch.nn.Parameter of the module is
  switched over to torch.Tensor. Thus, it can retain computational graphs for
  second order gradients while not losing compatibility with torch.nn.Module
  significantly. Note that some functions would fail after using this trick,
  e.g.nn.Module.apply(), which can be cured easily if needed.
  """
  def __init__(self, module, *args, **kwargs):
    self.module = module
    self.names = []
    params = []
    for name, param in module.named_parameters():
      self.names.append(name)
      params.append(param)
    super(ModuleSGD, self).__init__(params=params, *args, **kwargs)

  def step(self, closure=None, debug=False):
    loss = None
    if closure is not None:
      with torch.enable_grad():
          loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']

      for i, p in enumerate(self.module.parameters()):
        if p.grad is None:
            continue
        second_order = True if p.grad.grad_fn else False
        d_p = p.grad
        if weight_decay != 0:
          d_p = d_p.add(p, alpha=weight_decay)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          if nesterov:
            d_p = d_p.add(buf, alpha=momentum)
          else:
            d_p = buf
        if second_order:
          p = p.detach()
          p_s = p.add(d_p, alpha=-group['lr'])
          _update_params(self.module, self.names[i], p_s)
        else:
          p.add_(d_p, alpha=-group['lr'])
    return loss
