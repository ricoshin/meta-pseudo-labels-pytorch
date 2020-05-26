import argparse
import logging
import os
from pprint import pformat

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.utils import pin_in_object_store
import torch

from data import get_datasets
from utils.color import Color
from utils.config import init_config, sanity_check
from utils.watch import Watch
from optim.environment import TuningEnvironment

log = logging.getLogger('main')
log_result = logging.getLogger('result')

args = {
  'data_dir': 'data',
  'save_dir': 'tune',
  'log_level': 'info',
  'train_only': True,
}  # Note that these args take precedence over the args below.

parser = argparse.ArgumentParser(description='Meta Pseudo Labels(Tune).')
# general arguments
parser.add_argument('--config_default', type=str,
                    default='./config/001_default_tune.yaml')
parser.add_argument('--config', type=str,
                    default='./config/010_cifar10_supervised.yaml')
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--autotag', action='store_true')
parser.add_argument('--tag', type=str)
# tune arguments
parser.add_argument('--redis_port', type=str, default='31693')
parser.add_argument('--webui_host', type=str, default='127.0.0.1')
parser.add_argument('--local_dir', type=str, default='/data/private/ray_result')
parser.add_argument('--temp_dir', type=str, default='/data/private/tmp')
parser.add_argument('--save_dir', type=str, default='tune')
parser.add_argument('--cpu_per_trial', type=float, default=2)
parser.add_argument('--gpu_per_trial', type=float, default=0.3)
parser.add_argument('--extra_gpu', type=float, default=0.0)
parser.add_argument('--num_trials', type=int, default=256)
parser.add_argument('--pin_dataset', type=bool, default=True)


if __name__ == '__main__':
  parsed_args = parser.parse_args()
  # save dir has the same name as config yaml without the extension.
  args.update(vars(parsed_args))
  cfg = init_config(args)

  # overwrite
  cfg.update_dotmap({
    'loader_workers': int(cfg.cpu_per_trial),
    'n_labeled': 4000,
    # 'comm.n_steps': 500,
    'comm.n_steps': 300 if cfg.debug else 100000,
    'valid.interval': 100,
  })
  sanity_check(cfg)
  log.info(cfg)
  tune_space = cfg.detach('tune')

  # init (before pinning objects)
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

  # pinned datasets
  if cfg.pin_dataset:
    datasets = pin_in_object_store(get_datasets(cfg))
  else:
    datasets = None

  # metric mode
  mode = {
    'top1': 'max',
    'loss': 'loss',
  }[cfg.valid.metric]

  # Bayesian search algorithm
  # import pdb; pdb.set_trace()
  search_alg = BayesOptSearch(
    space=tune_space.toDict(),
    metric=cfg.valid.metric,
    mode=mode,
    utility_kwargs={
      "kind": "ucb",
      "kappa": 2.5,
      "xi": 0.0
    }
  )
  # Asyncronous HyperHand Scheduler
  scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric=cfg.valid.metric,
    mode=mode,
  )
  # ddd = TuningEnvironment.set_trainenv_kwargs(cfg=cfg, datasets=datasets)
  # import pdb; pdb.set_trace()
  # run
  assert torch.cuda.is_available()
  with Watch('tune.run', log) as t:
    analysis = tune.run(
      TuningEnvironment.with_trainenv_args(
        cfg=cfg,
        datasets=datasets,
      ),
      resources_per_trial={
        'cpu': cfg.cpu_per_trial,
        'gpu': cfg.gpu_per_trial,
        'extra_gpu': cfg.extra_gpu,
      },
      # config=tune_cfg,
      search_alg=search_alg,
      scheduler=scheduler,
      stop=lambda id, res: res['is_finished'],
      num_samples=2 if cfg.debug else cfg.num_trials,
      local_dir=cfg.local_dir,
      checkpoint_at_end=True,
      checkpoint_freq=3,
      # resume=True,
      # reuse_actors=True,
    )
  # analysis
  try:
    kwargs = {'metric': cfg.valid.metric, 'mode': mode, 'scope': 'all'}
    best_trial = analysis.get_best_trial(**kwargs)
    metric = best_trial.metric_analysis[cfg.valid.metric]
    config = analysis.get_best_config(**kwargs)
    log_result.critical(cfg)
    log_result.critical(f'tag: {Color.SELECTED}{cfg.tag}{Color.END}')
    log_result.critical(f'num_trials: {cfg.num_trials}')
    log_result.critical(f'seach space: {tune_space}')
    log_result.critical(f'metric({cfg.valid.metric}): {metric}')
    log_result.critical('\n' + pformat(config, indent=2, width=80))
  except:
    import pdb; pdb.set_trace()
