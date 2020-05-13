import signal

_signal_catchers = {}

"""
  Features:
    Signal catcher for debugging.
    You can catch any signal interrupt such as:
    ctrl + c : SIGINT
    ctrl + \ : SIGQUIT
    ctrl + z : SIGTSTOP

  Example:
    sigint = getSignalCatcher('SIGINT')
    while True:
      if sigint:
        import pdb; pdb.set_trace()
        # toggle on/off a breakpoint here by hitting ctrl + c.
    ...
"""

def getSignalCatcher(name):
  if name not in _signal_catchers:
    _signal_catchers.update({name: SignalCatcher(name)})
  return _signal_catchers[name]


class SignalCatcher(object):
  def __init__(self, name):
    self.name = name
    self._signal = getattr(signal, name)
    self._cond_func = None
    self.signal_on = False
    self._set_signal()

  def __bool__(self):
    return self.is_active(True)

  def _set_signal(self):
    def _toggle_func(signal, frame):
      if self.signal_on:
        self.signal_on = False
        print(f'Signal {self.name} Off!')
      else:
        self.signal_on = True
        print(f'Signal {self.name} On!')
    signal.signal(self._signal, _toggle_func)

  def is_active(self, cond=True):
    assert isinstance(cond, bool)
    if self.signal_on and cond:
      return True
    else:
      return False
