import copy
import logging

from ray.tune import Trainable
from ray.tune.utils import get_pinned_object
from torch import nn

from data import get_datasets, get_dataloaders
from nn.losses import TrainingLosses
from optim import units
from optim.manager import TrainingManager
from utils.config import Config
from utils.debugger import getSignalCatcher
from utils.tfwriter import TFWriters

log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_fmt(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


class TrainingEnvironment:
  """This class supports 2 different modes where the one is to automatically
  deploy hyperparameter tuning trials using 'Ray' module and the other is to
  train models in a customized way, which can make things go more flexible
  than sticking to the tuning framework."""
  def __init__(self, cfg):
    self._setup_base(cfg)

  def _setup_base(self, cfg, datasets=None):
    assert isinstance(cfg, Config)
    # traning manager
    datasets = datasets if datasets else get_datasets(cfg)
    self.loaders = get_dataloaders(cfg, datasets)
    self.writers = TFWriters(
      log_dir=cfg.save_dir,
      name_list=['train', 'valid'],
      deactivated=not cfg.save_dir,
    )
    self.m = TrainingManager(cfg, data_parallel=False)
    self.losses = TrainingLosses(cfg)
    self.cfg = cfg

  def _train_unit(self, tuning):
    """A training unit handling some amount of iterations before validation."""
    self.m.train()
    return units.train(
      cfg=self.cfg,
      iterable=self.m.train_step_generator(
        local_max_step=self.cfg.valid.interval,
        disable_pbar=tuning,
        ),
      ctrl_a=self.m.tchr if self.cfg.method.is_mpl else self.m.stdn,
      ctrl_b=self.m.stdn if self.cfg.method.is_mpl else None,
      loaders=self.loaders,
      losses=self.losses,
      )

  def _test_unit(self, tuning, mode):
    """A testing unit which can be used for both valiation and evaluation."""
    self.m.eval()
    return units.test(
      iterable=self.m.test_step_generator(
        test_loader=self.loaders.test,
        mode=mode,
        disable_pbar=tuning,
        ),
      ctrl=self.m.stdn,
      loaders=self.loaders,
      mode=mode,
      )

  def train(self, cfg):
    """Training function with validation."""
    # load model if possible
    self.m.load_if_available('last', verbose=True)
    if self.m.is_finished:
      log.info('No remaining steps.')
      return [None] * 3

    while not self.m.is_finished:
      # train & valid
      avgmeter_train = self._train_unit(tuning=False)
      avgmeter_valid = self._test_unit(tuning=False, mode='valid')
      self.m.monitor.watch(avgmeter_valid[cfg.valid.metric])

      # save & log
      with self.m.logging():
        self.m.save('last')
        if self.m.monitor.is_best:
          self.m.save('best')
          record_mark = ' [<--record]'
        else:
          record_mark = ''
        log_str = _to_fmt(
          self.m, avgmeter_train, avgmeter_valid, self.m.monitor)
        log.info(log_str + record_mark)
      # tensorboard
      self.writers.add_scalars('train', avgmeter_train.to_dict(), self.m.step)
      self.writers.add_scalars('valid', avgmeter_valid.to_dict(), self.m.step)
      self.writers.add_scalars(
        'train', {'lr': self.m.stdn.sched.get_lr()[0]}, self.m.step)
    return avgmeter_train, avgmeter_valid, self.m.monitor

  def test(self, cfg):
    """Pure test function."""
    self.m.load_if_available('best')
    return self._test_unit(tuning=False, mode='test')


class TuningEnvironment(Trainable, TrainingEnvironment):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _setup(self, tune_cfg):
    # import pdb; pdb.set_trace()
    assert isinstance(tune_cfg, dict)
    # self._setup_called = True
    # take original config out
    cfg = tune_cfg['cfg']
    del tune_cfg['cfg']
    if 'datasets' in tune_cfg:
      datasets = get_pinned_object(tune_cfg['datasets'])
      del tune_cfg['datasets']
    else:
      datasets = None
    # then update with selected hyper params by Ray
    # nothing happens when there's no other tuning configs.
    cfg.update_dotmap(tune_cfg)
    # now we can go as usual
    self._setup_base(cfg, datasets=datasets)

  def _train(self):
    """ray.tune() takes care of this function during hyperparameter tuning."""
    avgmeter_train = self._train_unit(tuning=True)
    avgmeter_test = self._test_unit(tuning=True, mode='valid')
    return {
      'step': self.m.step,
      'is_finished': self.m.is_finished,
      self.cfg.valid.metric: avgmeter_test[self.cfg.valid.metric],
      }

  def _save(self, save_dir):
    self.m.save('tune', save_dir=save_dir)
    return save_dir

  def _restore(self, save_dir):
    self.m.load_if_available('tune', save_dir=save_dir)
