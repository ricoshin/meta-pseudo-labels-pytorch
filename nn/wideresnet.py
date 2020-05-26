"""Reference: https://github.com/meliketoy/wide-resnet.pytorch"""
from collections import OrderedDict
import logging
import types
import weakref

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

log = logging.getLogger('main')


def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    init.constant_(m.bias, 0)
  elif classname.find('BatchNorm') != -1:
    init.constant_(m.weight, 1)
    init.constant_(m.bias, 0)


def patch_getattr(module):
  class AttrPatchedClass(type(module)):
    def __getattr__(self, name):
      attr = super(module.__class__, self).__getattr__(name)
      return weakref.ref(attr)()

    def __getattribute__(self, name):
      attr = object.__getattribute__(self, name)
      if name in ['_parameters', '_buffers', '_modules']:
        # weak reference for proper garbage collection
        attr = weakref.ref(attr)()
      return attr
  class_name = module.__class__.__name__
  module.__class__ = AttrPatchedClass
  module.__class__.__name__ = f"AttrPatched{class_name}"


class ViewThenLinear(nn.Linear):
  """to plug into nn.Sequential"""
  def forward(self, x):
    x = x.view(x.size(0), -1)
    return super(ViewThenLinear, self).forward(x)


class WideBasic(nn.Module):
  def __init__(self, in_planes, planes, bn_momentum, dropout_rate, stride=1):
    super(WideBasic, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
    self.conv1 = nn.Conv2d(
      in_planes, planes, kernel_size=3, padding=1, bias=True)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
    self.conv2 = nn.Conv2d(
      planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
      )

  def forward(self, x):
    out = self.dropout(self.conv1(F.relu(self.bn1(x))))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)

    return out


class WideResNet(nn.Module):
  def __init__(self, depth, widen_factor, bn_momentum, dropout_rate,
               num_classes, name='WideResNet'):
    super(WideResNet, self).__init__()

    self.name = name
    self.in_planes = 16

    assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
    n = (depth-4)/6
    k = widen_factor

    log.info('Building Wide-Resnet %dx%d.' %(depth, k))
    nStages = [16, 16*k, 32*k, 64*k]

    modules = nn.Sequential(OrderedDict([
      ('conv0', conv3x3(3,nStages[0])),
      ('wide1', self._wide_layer(
        WideBasic, nStages[1], n, bn_momentum, dropout_rate, stride=1)),
      ('wide2', self._wide_layer(
        WideBasic, nStages[2], n, bn_momentum, dropout_rate, stride=2)),
      ('wide3', self._wide_layer(
        WideBasic, nStages[3], n, bn_momentum, dropout_rate, stride=2)),
      ('bn4', nn.BatchNorm2d(nStages[3], momentum=bn_momentum)),
      ('relu4', nn.ReLU(inplace=True)),
      ('avg_pool4', nn.AvgPool2d(8)),
      ('linear4', ViewThenLinear(nStages[3], num_classes)),
    ]))
    # to have distinguishable name btw std & tchr
    self.add_module(name, modules)
    self.apply(patch_getattr)
    self.reset()

    # for debugging
    # self._paramters_backup = self._parameters
    # def _parameters(self):
    #   import pdb; pdb.set_trace()
    #   return self._paramters_backup
    # self._parameters = property(_parameters)

  def reset(self):
    def param_to_tensor(module):
      for name, param in module._parameters.items():
        module._parameters[name] = param.data.requires_grad_(True)
    self.apply(param_to_tensor)
  #   self.apply = self._apply_disabled
  #
  # def _apply_disabled(self):
  #   raise NotImplementedError

  def _wide_layer(
    self, block, planes, num_blocks, bn_momentum, dropout_rate, stride):
    strides = [stride] + [1]*(int(num_blocks)-1)
    layers = []

    for stride in strides:
      layers.append(block(
        self.in_planes, planes, bn_momentum, dropout_rate, stride))
      self.in_planes = planes

    return nn.Sequential(OrderedDict([
      (f'block{i}', l) for i, l in enumerate(layers)]))

  def forward(self, x):
    return getattr(self, self.name)(x)
