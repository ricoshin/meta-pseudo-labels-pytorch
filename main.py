import argparse
import logging
import os
import sys

if 'DEBUG_DEVICE' in os.environ:
  # this has to be done before importing PyTorch
  os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['DEBUG_DEVICE']
  os.environ['CUDA_LAUNCH_BLOCKING']='1'

from learn.environment import TrainingEnvironment 
from utils import GPUProfiler, Watch
from utils.config import Config, init_config, sanity_check


log = logging.getLogger('main')
log_result = logging.getLogger('result')

parser = argparse.ArgumentParser(description='Meta Pseudo Labels.')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--save_dir', type=str, default='out')
parser.add_argument('--config_default', type=str,
                    default='./config/000_default.yaml')
parser.add_argument('--config', type=str,
                    default='./config/002_cifar10_debug.yaml')
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--train_only', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--loader_workers', type=int, default=4)
parser.add_argument('--autotag', action='store_true')
parser.add_argument('--tag', type=str, default='')


if __name__ == '__main__':
  # config
  args = parser.parse_args()
  cfg = init_config(vars(args))
  sanity_check(cfg)
  if 'tune' in cfg:
    del cfg['tune']  # useless for now

  log.newline()
  log.info(cfg)

  if cfg.debug:
    log.warning('Debugging mode!')

  if 'DEBUG_DEVICE' in os.environ:
    gpu_profiler = GPUProfiler.instance(
      gpu_id=int(os.environ['DEBUG_DEVICE']),
      tensor_sizes=False,
      ignore_external=True,
      show_diff_only=True,
      console_out=True,
      # white_list=['optim'],
      # white_list=['optim.units.train'],
      white_list=['optim'],
      info_arg_names=['step', 'n'],
      # condition={'step': lambda step: step == 2},
      )
    gpu_profiler.global_set_trace()
    cfg.train_only = True
    cfg.comm.n_steps = 50
    cfg.valid.interval = 10
    log.warning("GPU profile will be saved to : "
                f"{gpu_profiler.out_filename}")

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

  if 'DEBUG_DEVICE' in os.environ:
    log.warning('Check out the GPU profile: '
                f'less -R {gpu_profiler.out_filename}')

  log.newline()
  log.info('[End_of_program]')
