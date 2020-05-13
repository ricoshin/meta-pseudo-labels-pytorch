import argparse
from logging import getLogger

from data.loader import get_dataloader
from optim.manager import TrainingManager
from optim.test import test
from optim.train import train
from utils.tfwriter import TFWriters
from utils.config import Config, init_config
from utils.watch import Watch

log = getLogger('mpl')


def main(cfg):
  log.info('Preparing for dataset.')
  loaders = get_dataloader(cfg)
  manager = TrainingManager(cfg)
  writers = TFWriters(
    log_dir=cfg.save_dir,
    name_list=['train', 'valid'], 
    deactivated=cfg.debug
    )

  if not cfg.eval_only:
    log.info('Training model.')
    result = train(cfg, loaders, manager, writers)
    log.info(result)

  log.info('Testing model.')
  result = test(cfg, loaders, manager, writers)
  log.info(result)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Meta Pseudo Labels.')
  parser.add_argument('--data_dir', type=str, default='data')
  parser.add_argument('--save_dir', type=str, default='out')
  # parser.add_argument('--load_dir', type=str, default='')
  parser.add_argument('--config_dir', type=str, default='config')
  parser.add_argument('--cfg', type=str, default='cifar10')
  parser.add_argument('--log_level', type=str, default='info')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--test_only', action='store_true')
  parser.add_argument('--from_scratch', action='store_true')
  parser.add_argument('--tag', type=str, default='')

  cfg = Config.from_parser(parser)
  cfg = init_config(cfg)
  # set_logger(cfg.log.level, cfg.save_dir)

  if cfg.debug:
    log.warning('Debugging mode!')

  with Watch('main_function', log) as t:
    main(cfg)

  log.info('End_of_program.')
