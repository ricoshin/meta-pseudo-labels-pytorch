import ctypes
import logging
import time

import torch
from torchviz import make_dot


def concat(tensors, retriever=False):
  out = torch.cat(tensors)
  if not retriever:
    return out
  lens = [len(t) for t in tensors]
  def splitter(concated):
    out = []
    for len_ in lens:
      out.append(concated[:len_])
      concated = concated[len_:]
    return out
  return out, splitter


def graph(tensor):
  print(make_dot(tensor))

def depth(tensor):
  return len(make_dot(tensor).body)


class PyObject(ctypes.Structure):
  _fields_ = [("refcnt", ctypes.c_long)]


def get_refcnt(addr):
  return PyObject.from_address(addr).refcnt
