import logging
import os
import pprint
import shutil
import time
import yaml
from argparse import ArgumentParser

from dotmap import DotMap

from utils import logger

log = logging.getLogger('mpl')


def init_config(cfg):
  assert isinstance(cfg, Config)
  # set logger
  # logger.set_level(cfg.log.level)
  logger.set_stream_handler(cfg.log.level)
  # set up dirs
  if cfg.tag:
    # new save_dir
    # date_time = time.strftime("%m%d_%H%M")
    # tag = '_' + cfg.tag if cfg.tag else ''
    # save_dir = os.path.join(cfg.save_dir, date_time + tag)
    tag = cfg.tag if cfg.tag else ''
    save_dir = os.path.join(cfg.save_dir, tag)
    # make dir if needed
    if os.path.exists(save_dir):
      log.warning(f'The same save_dir already exists: {save_dir}')
      if cfg.from_scratch:
        log.warning(f"'--from_scratch' mode on. "
                     'Press [ENTER] to remove the previous files.')
        input()
        shutil.rmtree(save_dir, ignore_errors=True)
        log.warning(f'Removed previous dirs and files.')
        os.makedirs(save_dir)
        log.info(f'Made new save_dir: {save_dir}')
    else:
      os.makedirs(save_dir)
      log.info(f'Made new save_dir: {save_dir}')
    # backup original ver.
    with open(os.path.join(save_dir, cfg.cfg + '.yaml'), 'w') as f:
      yaml.dump(cfg.toDict(), f, default_flow_style=False)
    cfg.save_dir = save_dir
    logger.set_file_handler(cfg.log.level, cfg.save_dir)
  else:
    log.warning(f"Nothing will be saved unless '--tag' is provided.")
    cfg.save_dir = ''

  if cfg.debug:
    cfg.log_level = 'debug'
  return cfg


class Config(DotMap):
  _instance = None

  def __add__(self, other):
    self_ = self.toDict()
    self_.update(other.toDict())
    return Config(self_)

  def __repr__(self):
    _dict = {k: v for k, v in self.toDict().items() if v}
    head = 'Config(Empty members are ommitted.)\n'
    return head + pprint.pformat(_dict, indent=2, width=80)

  @staticmethod
  def get():
    if Config._instance:
      return Config._instance
    else:
      return Config()

  @staticmethod
  def from_parser(parser):
    assert isinstance(parser, ArgumentParser)
    cfg = Config.get()
    cfg += Config(vars(parser.parse_args()))

    yaml_path  = os.path.join(cfg.config_dir, cfg.cfg)
    with open(yaml_path + '.yaml') as f:
      cfg += Config(yaml.safe_load(f))

    Config._instance = cfg
    return cfg
