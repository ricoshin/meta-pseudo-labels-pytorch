import argparse
from logging import getLogger

from data import get_dataloader
from optim.manager import TrainingManager
from optim.test import test
from optim.train import train
from utils.tfwriter import TFWriters
from utils.config import Config, init_config
from utils.watch import Watch

log = getLogger('main')
log_result = getLogger('result')


def main(cfg):
  log.newline()
  log.info('[Preparation]')
  loaders = get_dataloader(cfg)
  writers = TFWriters(
    log_dir=cfg.save_dir,
    name_list=['train', 'valid'],
    deactivated=not cfg.save_dir,
    )
  manager = TrainingManager(cfg, loaders, writers)

  if not cfg.test_only:
    log.newline()
    log.info('[Training session]')
    result_train, result_valid = train(cfg, manager)
    if result_train:
      assert result_valid
      log_result.critical(f'Final: {result_train}')
      log_result.critical(f'Final: {result_valid}')
      log_result.critical(manager.monitor)

  # For now, valid set == test set
  log.newline()
  log.info('[Test session]')
  result_test = test(cfg, manager)
  # import pdb; pdb.set_trace()
  log_result.critical(result_test)

  return True


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Meta Pseudo Labels.')
  parser.add_argument('--data_dir', type=str, default='data')
  parser.add_argument('--save_dir', type=str, default='out')
  # parser.add_argument('--load_dir', type=str, default='')
  # parser.add_argument('--config_dir', type=str, default='config')
  parser.add_argument('--config', type=str, default='./config/cifar10.yaml')
  parser.add_argument('--log_level', type=str, default='info')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--test_only', action='store_true')
  parser.add_argument('--from_scratch', action='store_true')
  parser.add_argument('--tag', type=str, default='')

  # config
  cfg = init_config(parser)
  log.newline()
  log.info(cfg)

  if cfg.debug:
    log.warning('Debugging mode!')

  # main
  with Watch('main', log) as t:
    main(cfg)

  log.newline()
  log.info('[End_of_program]')
