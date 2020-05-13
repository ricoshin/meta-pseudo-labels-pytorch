import logging
import os
import pprint
import time
import yaml
from argparse import ArgumentParser

from dotmap import DotMap

log = logging.getLogger('mpl')


def post_process_config(cfg):
  assert isinstance(cfg, Config)
  if cfg.tag:
    # new save_dir
    sub_dir = time.strftime("%m%d_%h%m", time.gmtime(time.time()))
    dir_tag = '_' + cfg.tag if cfg.tag else ''
    save_dir = os.path.join(cfg.save_dir, sub_dir + dir_tag)
    # make dir if needed
    if os.path.exists(save_dir):
      log.warning(f'The same save_dir already exist: {save_dir}')
    else:
      os.makedirs(save_dir)
      log.info(f'Made new save_dir: {save_dir}')
    # backup original ver.
    with open(os.path.join(save_dir, cfg.cfg + '.yaml'), 'w') as f:
      yaml.dump(cfg.toDict(), f, default_flow_style=False)
    cfg.save_dir = save_dir
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
  def init_from_parser(parser):
    assert isinstance(parser, ArgumentParser)
    cfg = Config.get()
    cfg += Config(vars(parser.parse_args()))

    yaml_path  = os.path.join(cfg.config_dir, cfg.cfg)
    with open(yaml_path + '.yaml') as f:
      cfg += Config(yaml.safe_load(f))

    Config._instance = cfg
    return cfg
