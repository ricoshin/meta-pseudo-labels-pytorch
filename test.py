import hyperopt_run

if __name__ == '__main__':
  hyperopt_run.main({
    'config': './config/001_cifar_debug.yaml',
    'stdn/lr': 100,
    'batch_size/sup': 128,

  })
