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


class Watch:
  def __init__(self, name, logger):
    assert isinstance(name, str)
    assert isinstance(logger, logging.Logger)
    self.name = name
    self.logger = logger

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, type, value, trace_back):
    secs = time.time() - self.start
    hrs = int(secs // 3600)
    mins = int((secs % 3600) // 60)
    secs = int(secs % 60)
    self.logger.info(f'Watch({self.name}): {hrs} hrs {mins} mins {secs} secs')
