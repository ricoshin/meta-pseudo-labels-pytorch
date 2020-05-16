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


class Config(DotMap):
  """A singleton class for managing global configuration."""
  _instance = None

  def __add__(self, other):
    self_ = self.toDict()
    self_.update(other.toDict())
    return Config(self_)

  def __str__(self):
    _dict = {k: v for k, v in self.toDict().items() if v}
    head = 'Config(Empty members are ommitted.)\n'
    return head + pprint.pformat(_dict, indent=2, width=80)

  @staticmethod
  def get():
    if Config._instance:
      return Config._instance
    else:
      return Config()


def sanity_check(cfg):
  assert cfg.method.base in ['sup', 'ra', 'uda']
  assert cfg.valid.metric in ['top1', 'loss']
  return cfg


def init_config(parser):
  """Function for initializing coniguration."""
  assert isinstance(parser, ArgumentParser)

  # GET CONFIG
  cfg = Config.get()

  # PARSE_ARGS
  cfg += Config(vars(parser.parse_args()))

  # LOG LEVEL
  if cfg.debug:
    cfg.log_level = 'debug'

  # LOGGER (console)
  logger.set_stream_handler('main', cfg.log_level)
  logger.set_stream_handler('result', cfg.log_level)

  # SAVE_DIR
  if not cfg.tag:
    log.warning(f"Nothing will be saved unless '--tag' is given.")
    cfg.save_dir = ''
  else:
    # save_dir: save_dir/tag
    cfg.save_dir = os.path.join(cfg.save_dir, cfg.tag)
    if os.path.exists(cfg.save_dir):
      log.warning(f'The same save_dir already exists: {cfg.save_dir}')
      if cfg.from_scratch:
        # remove previous files and go from the scratch
        assert not cfg.test_only, 'Test cannot be performed from scratch!'
        log.warning(f"'--from_scratch' mode on. "
                    f'Press [ENTER] to remove the previous files.')
        input()  # waiting for ENTER
        shutil.rmtree(cfg.save_dir, ignore_errors=True)
        log.warning(f'Removed previous dirs and files.')
        os.makedirs(cfg.save_dir)
        log.info(f'Made new save_dir: {cfg.save_dir}')
      # else:
      #   log.info('Previous checkpoints will be loaded if available.')
    else:
      # make dir if needed
      os.makedirs(cfg.save_dir)
      log.info(f'Made new save_dir: {cfg.save_dir}')

  # SET LOGGER (for file streaming)
  if cfg.save_dir:
    logger.set_file_handler('main', cfg.log_level, cfg.save_dir, 'log')
    logger.set_file_handler('result', cfg.log_level, cfg.save_dir, 'result')

  # LOAD (AND BACKUP) YAML
  yaml_files = glob.glob(os.path.join(cfg.save_dir, '*.yaml'))
  if len(yaml_files) == 0:
    if not cfg.config:
      raise Exception('No config file provided. Specify dir having '
                      '.yaml file in it or provide new one by --config')
    # if there's no config file in save_dir, use newly privided one.
    log.warning(f'No existing config file.')
    yaml_file = cfg.config
    if cfg.save_dir:  # backup to use in the future
      config_backup = os.path.join(cfg.save_dir, os.path.basename(yaml_file))
      shutil.copyfile(cfg.config, config_backup)
      log.info(f'Backup config file: {config_backup}')
  elif len(yaml_files) == 1:
    # if there exist yaml file already in save_dir, take that one.
    yaml_file = yaml_files[0]
    log.info(f'Found an existing config file.')
    if cfg.config:
      log.info(f'Given file with --config option will be ignored.')
  else:
    raise Exception(f'More than one yaml file in {cfg.save_dir}.')
  # load .yaml and incorporate into the config
  with open(yaml_file) as f:
    cfg += Config(yaml.safe_load(f))
  log.info(f'Config file loaded: {yaml_file}')

  return sanity_check(cfg)
