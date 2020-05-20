import argparse
import logging

from optim.environment import TrainingEnvironment
from utils.config import Config, init_config, sanity_check
from utils.watch import Watch


log = logging.getLogger('main')
log_result = logging.getLogger('result')

parser = argparse.ArgumentParser(description='Meta Pseudo Labels.')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--save_dir', type=str, default='out')
parser.add_argument('--config_default', type=str,
                    default='./config/000_default.yaml')
parser.add_argument('--config', type=str,
                    default='./config/006_cifar10_supervised_mpl.yaml')
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--train_only', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--loader_workers', type=int, default=4)
parser.add_argument('--tag', type=str, default='')


if __name__ == '__main__':
  # config
  args = parser.parse_args()
  assert not (args.train_only == True and args.test_only == True)
  cfg = init_config(vars(args))
  sanity_check(cfg)
  if 'tune' in cfg:
    del cfg['tune']  # useless for now

  log.newline()
  log.info(cfg)

  if cfg.debug:
    log.warning('Debugging mode!')

  # main
  with Watch('main', log) as t:
    log.newline()
    log.info('[Preparation]')
    env = TrainingEnvironment(cfg)

    if not cfg.test_only:
      log.newline()
      log.info('[Training session]')
      avgmeter_train, avgmeter_valid, monitor = env.train(cfg)
      if avgmeter_train:
        assert avgmeter_valid and monitor
        log_result.critical(f'Final: {avgmeter_train}')
        log_result.critical(f'Final: {avgmeter_valid}')
        log_result.critical(monitor)

    if not cfg.train_only:
      # For now, valid set == test set
      log.newline()
      log.info('[Test session]')
      avgmeter_test = env.test(cfg)
      log_result.critical(avgmeter_test)

  log.newline()
  log.info('[End_of_program]')
