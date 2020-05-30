import copy
import logging

from ray.tune import Trainable
from ray.tune.utils import get_pinned_object
from torch import nn

from data import get_datasets, get_dataloaders
from model.losses import TrainingLosses
from learn import train, test, TrainingManager
from utils.config import Config
from utils.debugger import getSignalCatcher
from utils.tfwriter import TFWriters

log = logging.getLogger('main')
sigquit = getSignalCatcher('SIGQUIT')


def _to_fmt(*args, delimiter=' | '):
  return delimiter.join(map(str, args))


class TrainingEnvironment:
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
    self.m = TrainingManager(cfg, data_parallel=cfg.data_parallel)
    self.losses = TrainingLosses(cfg)
    self.cfg = cfg

  def _train_unit(self, tuning):
    """A training unit handling some amount of iterations before validation."""
    self.m.train()
    return train(
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
    return test(
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
  _trainenv_kwargs = {}

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @staticmethod
  def with_trainenv_args(**kwargs):
    assert 'cfg' in kwargs, 'missing argument.'
    assert isinstance(kwargs['cfg'], Config)
    return type("TuningEnvironmentWithTrainArgs",
                (TuningEnvironment, ),
                {'_trainenv_kwargs': kwargs})

  def _setup(self, tuning_cfg):
    assert isinstance(tuning_cfg, dict)
    assert self.__class__.__name__ == 'TuningEnvironmentWithTrainArgs', \
            'Use the class returned from .with_trainenv_args().'
    kwargs = self.__class__._trainenv_kwargs
    for key, value in kwargs.items():
      if key == 'datasets':
        kwargs[key] = get_pinned_object(value)
      elif key == 'cfg':
        value.update_dotmap(tuning_cfg)
      else:
        raise Exception(f'Unexpected pre-set trainenv argument:{key}')
    self._setup_base(**kwargs)

  def _train(self):
    """ray.tune() takes care of this function during hyperparameter tuning."""
    avgmeter_train = self._train_unit(tuning=True)
    avgmeter_valid = self._test_unit(tuning=True, mode='valid')
    return {
      'step': self.m.step,
      'is_finished': self.m.is_finished,
      self.cfg.valid.metric: avgmeter_valid[self.cfg.valid.metric],
      }

  def _save(self, save_dir):
    self.m.save('tune', save_dir=save_dir)
    return save_dir

  def _restore(self, save_dir):
    self.m.load_if_available('tune', save_dir=save_dir)
