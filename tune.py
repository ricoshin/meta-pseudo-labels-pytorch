import argparse
import logging
import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.utils import pin_in_object_store
import torch

from data import get_datasets
from utils.config import init_config, sanity_check
from utils.watch import Watch
from optim.environment import TuningEnvironment

log = logging.getLogger('main')
log_result = logging.getLogger('result')

args = {
  'data_dir': 'data',
  'save_dir': 'tune',
  'log_level': 'info',
  'tag': 'tune',
  # 'debug': False,
  'train_only': True,
  'from_scratch': True,
}  # Note that these args take precedence over the args below.

parser = argparse.ArgumentParser(description='Meta Pseudo Labels(Tune).')
parser.add_argument('--config_default', type=str,
                    default='./config/001_default_tune.yaml')
parser.add_argument('--config', type=str,
                    default='./config/010_cifar10_supervised.yaml')
parser.add_argument('--redis_port', type=str, default='31693')
parser.add_argument('--webui_host', type=str, default='127.0.0.1')
parser.add_argument('--local_dir', type=str, default='/data/private/ray_result')
parser.add_argument('--temp_dir', type=str, default='/data/private/tmp')
# parser.add_argument('--num_trials', type=int)
parser.add_argument('--cpu_per_trial', type=float, default=2)
parser.add_argument('--gpu_per_trial', type=float, default=0.3)
parser.add_argument('--extra_gpu', type=float, default=0.0)
parser.add_argument('--num_trials', type=int, default=128)
parser.add_argument('--pin_dataset', type=bool, default=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--tag', type=str)


if __name__ == '__main__':
  parsed_args = parser.parse_args()
  # save dir has the same name as config yaml without the extension.
  if not parsed_args.tag:
    yaml_file = os.path.basename(parsed_args.config)
    parsed_args.tag = yaml_file.split('.')[0]
  args.update(vars(parsed_args))
  cfg = init_config(args)

  # overwrite
  cfg.update_dotmap({
    'loader_workers': cfg.cpu_per_trial,
    'n_labeled': 4000,
    # 'comm.n_steps': 500,
    'comm.n_steps': 100000,
    'valid.interval': 100,
  })
  sanity_check(cfg)
  log.info(cfg)

  # make tuning configurations recognizable to ray
  _tune_cfg = cfg.detach('tune').toDict()
  tune_cfg = {}
  for k, v in _tune_cfg.items():
    sampler = getattr(tune, v['sampler'])
    if v['sampler'] in ['uniform', 'loguniform']:
      tune_cfg[k] = sampler(*v['values'])
    else:
      tune_cfg[k] = sampler(v['values'])
  # init
  ray.init(
    # address=cfg.ray_address,
    redis_port=cfg.redis_port,
    webui_host=cfg.webui_host,
    # local_mode=cfg.debug,
    # object_store_memory=200*1024*2024, # 200MiB
    memory=200*1024*2024, # 200MiB
    object_store_memory=200*1024*2024, # 200MiB
    driver_object_store_memory=200*1024*1204,
    redis_max_memory=200*1024*1204,
    temp_dir=cfg.temp_dir,
    # plasma_directory=cfg.temp_dir,
    lru_evict=True,
  )
  # put everything into the single dict
  # to pass through its narrow way in.
  if cfg.pin_dataset:
    # pinned dataloaders
    datasets = pin_in_object_store(get_datasets(cfg))
    tune_cfg['datasets'] = datasets
  tune_cfg['cfg'] = cfg
  # Asyncronous HyperHand Scheduler
  sched = ASHAScheduler(
    time_attr='training_iteration',
    metric=cfg.valid.metric,
    mode={
      'top1': 'max',
      'loss': 'loss',
    }[cfg.valid.metric],
  )
  # run
  assert torch.cuda.is_available()
  with Watch('tune.run', log) as t:
    analysis = tune.run(
      TuningEnvironment,
      # reuse_actors=True,
      config=tune_cfg,
      scheduler=sched,
      stop=lambda id, res: res['is_finished'],
      num_samples=3 if cfg.debug else cfg.num_trials,
      # checkpoint_at_end=True,
      # checkpoint_freq=3,
      resources_per_trial={
        'cpu': cfg.cpu_per_trial,
        'gpu': cfg.gpu_per_trial,
        'extra_gpu': cfg.extra_gpu,
      },
      local_dir=cfg.local_dir,
      # resume=True,
    )
  # analysis
  best_dict = analysis.get_best_config(metric=cfg.valid.metric)
  if 'datasets' in best_dict:
    del best_dict['datasets']
  log_result.critical(best_dict['cfg'])
  del best_dict['cfg']
  log_result.critical(best_dict)