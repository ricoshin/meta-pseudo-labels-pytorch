"""Reference: https://gist.github.com/MInner/8968b3b120c95d3f50b8a22a74bf66bc"""
from collections import defaultdict
import datetime
import linecache
import os
import sys

from py3nvml import py3nvml
import torch

from utils.logger import set_file_handler, set_stream_handler
from utils.color import Color

def get_gpu_memory(id):
  py3nvml.nvmlInit()
  handle = py3nvml.nvmlDeviceGetHandleByIndex(id)
  meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
  py3nvml.nvmlShutdown()
  return meminfo.used / 1024**2


class GPUProfiler:
  _instance = None
  def __init__(self, gpu_id, tensor_sizes=False, ignore_external=True,
               show_diff_only=False, console_out=False, out_filename='',
               white_list=[], black_list=[], info_arg_names=[], condition={}):
    self.last_tensor_sizes = set()
    self.last_meminfo_used = 0
    self.lineno = None
    self.func_name = None
    self.filename = None
    self.module_name = None
    self.prev_lineno = 0
    self.prev_func_name = ''
    self.profile_gpu_id = gpu_id
    self.print_tensor_sizes = tensor_sizes
    if self.print_tensor_sizes:
      self.tensor_counter = defaultdict(int)
    self.ignore_external = ignore_external
    self.show_diff_only = show_diff_only
    self.info_arg_values = {}
    self.do_profile = False
    self.enabled = True
    """
    Example:
      white_list: ['optim.units.train', 'optim.sgd']
      black_list: ['optim.sgd._update_params']
    Note:
      black list takes precedence over white list.
    """
    assert isinstance(white_list, (tuple, list))
    assert isinstance(black_list, (tuple, list))
    self.white_list = white_list
    self.black_list = black_list
    self.info_arg_names = info_arg_names
    self.condition = condition
    if out_filename:
      self.out_filename = out_filename
    else:
      self.out_filename = (f"gpu_{self.profile_gpu_id}_profile_"
                           f"{datetime.datetime.now():%d%m%y_%H%M%S}.txt")
    self.log = set_file_handler('gpu_profile', 'debug', '', self.out_filename)
    if console_out:
      set_stream_handler('gpu_profile', 'debug')

  @classmethod
  def _get_instance(cls, enabled=True):
    cls._instance.enabled = enabled
    return cls._instance

  @classmethod
  def instance(cls, *args, **kwargs):
    cls._instance = cls(*args, **kwargs)
    cls.instance = cls._get_instance
    return cls._instance

  def global_set_trace(self):
    sys.settrace(self._trace_calls)

  def __enter__(self):
    if self.enabled:
      sys.settrace(self._trace_calls)

  def __exit__(self, type, value, trace_back):
    sys.settrace(None)

  def _trace_calls(self, frame, event, arg):
    if not event == 'line':
      return self._trace_calls

    # precoess _previous_ line
    if self.do_profile:
      self._gpu_profile()
      self.do_profile = False

    # save details about line _to be_ executed
    self.func_name = frame.f_code.co_name
    try:
      self.filename = frame.f_globals["__file__"]
      if (self.filename.endswith(".pyc") or self.filename.endswith(".pyo")):
        self.filename = self.filename[:-1]
    except (KeyError, AttributeError) as ex:
      return self._trace_calls
    self.module_name = frame.f_globals["__name__"]
    module_func_name = ".".join([self.module_name, self.func_name])
    self.lineno = frame.f_lineno
    if self.lineno is not None:
      self.do_profile = True

    # exclude some files and modules
    if (self.ignore_external and
        os.path.abspath('') not in os.path.abspath(self.filename)):
      self.do_profile = False

    if 'gpu_profile' in self.module_name:
      self.do_profile = False

    if (len(self.white_list) > 0 and
        not any([m in module_func_name for m in self.white_list])):
      self.do_profile = False

    if any([m in module_func_name for m in self.black_list]):
      self.do_profile = False

    if len(self.condition) > 0:
      condition = []
      for arg_name, arg_func in self.condition.items():
        if arg_name in frame.f_locals.keys():
          condition.append(arg_func(frame.f_locals[arg_name]))
      if not any(condition):
        self.do_profile = False

    if self.do_profile:
      self.info_arg_values = {}
      for arg in self.info_arg_names:
        assert isinstance(arg, str)
        if arg in frame.f_locals:
          self.info_arg_values[arg] = str(frame.f_locals[arg])
        # else:
        #   self.info_arg_values[arg] = '(not found)'
    return self._trace_calls

  def _gpu_profile(self, info=''):
    py3nvml.nvmlInit()
    handle = py3nvml.nvmlDeviceGetHandleByIndex(self.profile_gpu_id)
    meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
    line = linecache.getline(self.filename, self.lineno)
    head_str = self.module_name + ' ' + self.func_name + ':' + str(self.lineno)
    info_str = [f'{k}:{v}' for k, v in self.info_arg_values.items()]
    info_str = f" ({', '.join(info_str)})" if info_str else ""
    new_meminfo_used = meminfo.used
    mem_diff = new_meminfo_used - self.last_meminfo_used
    self.last_meminfo_used = new_meminfo_used

    if (not self.show_diff_only
          or (self.show_diff_only and mem_diff != 0)):

      # draw lines
      if self.func_name != self.prev_func_name:
        self.log.debug("=" * 120)
      if (self.func_name == self.prev_func_name
            and self.lineno < self.prev_lineno):
        self.log.debug("-" * 120)
      self.prev_lineno = self.lineno
      self.prev_func_name = self.func_name

      # total & incremental memory usage
      if mem_diff == 0:
        mem_color = Color.WHITE
      elif mem_diff > 0:
        mem_color = Color.GREEN2
      else:
        mem_color = Color.RED
      self.log.debug(f"{head_str + info_str:<50}"
                     f"{(new_meminfo_used)/1024**2:<7.1f}Mb "
                     f"{mem_color}({mem_diff/1024**2:>+7.1f}Mb){Color.END}"
                     f"{line.rstrip()}")

      # new & deleted tensors
      if self.print_tensor_sizes :
        for tensor in self._get_tensors():
          if not hasattr(tensor, '_head_str'):
            tensor._head_str = head_str
          if not hasattr(tensor, '_info_str'):
            tensor._info_str = info_str
          if not hasattr(tensor, '_id_origin'):
            tensor._id_origin = id(tensor)
        new_tensor_sizes = set()
        for x in self._get_tensors():
          new_tensor_sizes.add(
            (type(x), tuple(x.size()), x._head_str, x._info_str, x._id_origin))
        new_set = new_tensor_sizes - self.last_tensor_sizes
        del_set = self.last_tensor_sizes - new_tensor_sizes

        for t, s, h, i, d in new_set:
          self.log.debug(
            f"{Color.GREY}{h + i:<50} [+] {Color.END} "
            f"{Color.GREEN2}{str(s):<20}{Color.END} "
            f"{Color.GREY}{str(d):<20}{Color.END} "
            f"{Color.GREY}{str(t):<20}{Color.END} "
            )
          self.tensor_counter[str(s)] += 1
        # self.log.debug(f"{len(new_set)}")
        for t, s, h, i, d in del_set:
          self.log.debug(
            f"{Color.GREY}{h + i:<50} [-] {Color.END} "
            f"{Color.RED}{str(s):<20}{Color.END} "
            f"{Color.GREY}{str(d):<20}{Color.END} "
            f"{Color.GREY}{str(t):<20}{Color.END} "
            )
          self.tensor_counter[str(s)] -= 1
        cnt = 0
        for s, n in self.tensor_counter.items():
          cnt += n
          self.log.debug(f"{Color.GREY}{' ' * 50} [*] {Color.END}"
                         f"{Color.YELLOW}{str(s):<20}{Color.END}")
        self.log.debug(f"{Color.WHITE} Counter: {Color.END}"
                       f"{Color.GREEN2} [+] #:  {len(new_set):<5}{Color.END}"
                       f"{Color.RED} [-] #: {len(del_set):<5}{Color.END}"
                       f"{Color.YELLOW} [*] #: {cnt:<5}{Color.END}")
        self.last_tensor_sizes = new_tensor_sizes

    py3nvml.nvmlShutdown()
    return

  def _get_tensors(self, gpu_only=True):
    import gc
    for obj in gc.get_objects():
      try:
        if torch.is_tensor(obj):
          tensor = obj
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
          tensor = obj.data
        else:
          continue
        if tensor.is_cuda:
          yield tensor
      except Exception as e:
        pass
