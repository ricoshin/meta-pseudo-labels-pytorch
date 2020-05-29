import collections
import glob
import logging
import os
import pprint
import shutil
import time
import yaml
from argparse import ArgumentParser

from dotmap import DotMap

from utils import logger
from utils.color import Color

log = logging.getLogger('main')


def _update_dict(d, u):
  """recursive dict update."""
  for k, v in u.items():
    if isinstance(v, collections.abc.Mapping):
      d[k] = _update_dict(d.get(k, {}), v)
    else:
      d[k] = v
  return d


class Config(DotMap):
  """A singleton class for managing global configuration."""
  _instance = None

  @staticmethod
  def instance():
    if Config._instance:
      return Config._instance
    else:
      return Config()

  def __add__(self, other):
    assert isinstance(other, Config)
    self_dict = self.toDict()
    _update_dict(self_dict, other.toDict())
    return Config(self_dict)

  def __str__(self):
    _dict = {k: v for k, v in self.toDict().items() if v}
    head = 'Config(Empty members are omitted.)\n'
    return head + pprint.pformat(_dict, indent=2, width=80)

  def update_dotmap(self, dict, delimiter='.'):
    for k, v in dict.items():
      self.set_dotmap(k, v, delimiter)

  def set_dotmap(self, key, value, delimiter='.'):
    obj = self
    keys = key.split(delimiter)
    for k in keys[:-1]:
      obj = obj.get(k)
    obj[keys[-1]] = value

  def get_dotmap(self, key, delimiter='.'):
    obj = self
    for k in key.split(delimiter):
      obj = obj.get(k)
    return obj

  def detach(self, key):
    detached = self[key]
    del self[key]
    return detached


def _post_process(cfg):
  if cfg.method.base == 'uda':
    cfg.method.is_uda = True
  return cfg


def sanity_check(cfg):
  assert not (cfg.train_only and cfg.test_only)
  assert cfg.n_labeled >= cfg.batch_size.sup
  assert cfg.method.base in ['sup', 'ra', 'uda']
  assert cfg.valid.metric in ['top1', 'loss']
  assert cfg.uda.tsa_schedule in ['', 'linear', 'log', 'exp']
  return cfg


def init_config(args):
  """Function for initializing coniguration."""
  assert isinstance(args, collections.abc.Mapping)
  # GET CONFIG & UPDATE ARGS
  cfg = Config.instance()
  cfg += Config(args)

  # LOG LEVEL
  if cfg.debug:
    cfg.log_level = 'debug'

  # LOGGER (console)
  logger.set_stream_handler('main', cfg.log_level)
  logger.set_stream_handler('result', cfg.log_level)

  if cfg.autotag:
    date = time.strftime("%m%d", time.gmtime(time.time()))
    yaml_name = os.path.basename(cfg.config).split('.')[0]
    if cfg.tag:
      cfg.tag = f"{date}/{yaml_name}_{cfg.tag}"
    else:
      tag_num = 0
      tag_gen = lambda n: f"{date}/{yaml_name}_" + str(tag_num).zfill(3)
      while os.path.exists(f"{cfg.save_dir}/{tag_gen(tag_num)}"):
        tag_num += 1
      cfg.tag = tag_gen(tag_num)

  # SAVE_DIR
  if not cfg.tag:
    log.warning(f"Nothing will be saved unless '--tag' is given.")
    cfg.save_dir = ''
  else:
    # save_dir: {save_dir}/{tag}
    cfg.save_dir = os.path.join(cfg.save_dir, cfg.tag)
    if os.path.exists(cfg.save_dir):
      log.warning(f'The same save_dir already exists: {cfg.save_dir}')
      if cfg.from_scratch:
        # remove previous files and go from the scratch
        assert not cfg.test_only, 'Test cannot be performed from scratch!'
        log.warning(f"'--from_scratch' mode on. "
                    f'Press {Color.GREEN2}[ENTER]{Color.END} '
                    'to remove the previous files.')
        input()  # waiting for ENTER
        shutil.rmtree(cfg.save_dir, ignore_errors=True)
        log.warning(f'Removed previous dirs and files.')
        os.makedirs(cfg.save_dir)
        log.info(f'Made new save_dir: {cfg.save_dir}')
      # else:
      #   log.info('Previous checkpoints will be loaded if available.')
    else:
      if cfg.test_only:
        raise Exception(f'Cannot find test dir: {cfg.save_dir}.')
      # make dir if needed
      os.makedirs(cfg.save_dir)
      log.info(f'Made new save_dir: {cfg.save_dir}')

  # SET LOGGER (for file streaming)
  if cfg.save_dir and not cfg.test_only:
    logger.set_file_handler('main', cfg.log_level, cfg.save_dir, 'log')
    logger.set_file_handler('result', cfg.log_level, cfg.save_dir, 'result')

  # LOAD (AND BACKUP) YAML
  cfg_yaml = Config()
  if cfg.save_dir:
    yaml_files = glob.glob(os.path.join(cfg.save_dir, '*.yaml'))
  else:
    yaml_files = []
  if len(yaml_files) == 0:
    if not cfg.config:
      raise Exception('No config file provided. Specify dir having '
                      '.yaml file in it or provide new one by --config')
    # if there's no config file in save_dir, use newly privided one.
    log.warning(f'No existing config file.')
    assert not cfg.test_only, 'test_only mode needs a saved config file.'
    yaml_file = cfg.config
    # load default config first
    if os.path.exists(cfg.config_default):
      with open(cfg.config_default) as f:
        cfg_yaml += Config(yaml.safe_load(f))
      log.info(f'Loaded default config file: {cfg.config_default}')
    else:
      raise Exception('Default config file could not be found: '
                      f'{cfg.config_default}')
    backup = True
  elif len(yaml_files) == 1:
    # if there already exists an yaml file in save_dir, take that one.
    yaml_file = yaml_files[0]
    log.warning(f'Found an existing config file.')
    if cfg.config:
      log.warning(f'Given file with --config option will be ignored.')
    backup = False
  else:
    raise Exception(f'More than one yaml file in {cfg.save_dir}.')
  # load .yaml and incorporate into the config
  with open(yaml_file) as f:
    cfg_yaml += Config(yaml.safe_load(f))
  log.info(f'Loaded config file: {yaml_file}')
  # backup to use in the future
  if cfg.save_dir and backup:
    config_backup = os.path.join(cfg.save_dir, os.path.basename(yaml_file))
    with open(config_backup, 'w') as f:
      yaml.dump(cfg_yaml.toDict(), f)
    # shutil.copyfile(cfg.config, config_backup)
    log.info(f'Backed up config file: {config_backup}')

  cfg += cfg_yaml
  return _post_process(cfg)
