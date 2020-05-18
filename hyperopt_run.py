import logging
from utils.config import init_config
from utils.watch import Watch
from main import main as main_origin

log = logging.getLogger('main')

parse_args = {
  'data_dir': 'data',
  'save_dir': 'out',
  'config_default': './config/001_default_tune.yaml',
  'log_level': 'info',
  'from_scratch': True,
  'debug': True,
}

hyper_opt_args = {
  'n_labeled': 400,
  'comm/n_steps': 100000,
}


def main(args):
  with Watch('main', log) as t:
    assert isinstance(args, dict)
    assert 'config' in args
    parse_args['config'] = args['config']
    cfg = init_config(parse_args)
    cfg.update_by_dotkey(args, delimiter='/')
    cfg.update_by_dotkey(hyper_opt_args, delimiter='/')
    log.newline()
    log.info(cfg)
    result = main_origin(cfg)
    log.newline()
    log.info('[End_of_main]')
  return {
    'loss': 0,
    'accuracy': result['top1'],
    'status': 'ok',
    }


if __name__ == '__main__':

  result = main({
    'config': './config/001_cifar_debug.yaml',
    'stdn/lr': 0.123,
    'tchr/lr': 0.123,
    'batch_size/sup': 123,
    'batch_size/sup': 123,
  })
  assert result['status'] == 'ok'
