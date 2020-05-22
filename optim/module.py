import torch
from torch.nn import Module, Parameter
from torch import Tensor


class NonLeafModule(Module):
  def update_parameter(self, name, param):
    if '_parameters' not in self.__dict__:
      raise AttributeError(
        "cannot assign parameter before Module.__init__() call")
    elif not isinstance(name, torch._six.string_classes):
      raise TypeError("parameter name should be a string. "
        "Got {}".format(torch.typename(name)))
    elif '.' in name:
      raise KeyError("parameter name can't contain \".\"")
    elif name == '':
      raise KeyError("parameter name can't be empty string \"\"")
    elif name not in self._parameters:
      raise KeyError("attribute '{}' doesn't exists".format(name))

    if not isinstance(param, Tensor) or not isinstance(param, Parameter):
      raise TypeError("cannot assign '{}' object to parameter '{}' "
                      "(torch.nn.Parameter or torch.Tensor required)"
                      .format(torch.typename(param), name))
    else:
      self._parameter[name] = param

  def apply_update(module, named_parameters):
    for name, param:
      names = name.split('.')
      for
       names[0]
