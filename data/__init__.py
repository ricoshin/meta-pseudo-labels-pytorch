from collections import namedtuple
from filelock import FileLock
from logging import getLogger

import torchvision
from torch.utils.data import DataLoader, Subset, ConcatDataset

from augment import get_transforms
from data.datasets import TransformedDataset, BiTransformedDataset
from data.dataloaders import InfiniteDataLoader, DataLoaderTriplet
from utils.config import Config

log = getLogger('main')


def get_dataloaders(cfg, datasets):
  log.info('Load dataset.')
  data_train, data_test = datasets
  need_uns = cfg.method.base == 'uda' or cfg.method.mpl
  # data_train, data_test = get_dataset(cfg.dataset, cfg.data_dir)
  # split labeled(supervised) & unlabled(unsupervised) set
  data_sup = Subset(data_train, range(len(data_train))[:cfg.n_labeled])
  if need_uns:
    data_uns = Subset(data_train, range(len(data_train))[cfg.n_labeled:])
  # different transforms(augumentation) for each dataset
  trans_sup, trans_uns, trans_test = get_transforms(
    cfg.dataset, cfg.method.base, cfg.aug.default, cfg.aug.cutout,
    randaug_args=(cfg.randaug.n, cfg.randaug.m))
  # attach transforms to the datasets
  data_sup = TransformedDataset(data_sup, trans_sup)
  if need_uns:
    data_uns = BiTransformedDataset(data_uns, trans_sup, trans_uns)
  data_test = TransformedDataset(data_test, trans_test)
  # data loader
  comm_kwargs = dict(
    num_workers=0 if cfg.debug else cfg.loader_workers,
    pin_memory=True,
    )
  loader_sup = InfiniteDataLoader(
    data_sup, batch_size=int(cfg.batch_size.sup),
    shuffle=True, drop_last=True, **comm_kwargs)
  if need_uns:
    loader_uns = InfiniteDataLoader(
      data_uns, batch_size=int(cfg.batch_size.uns),
      shuffle=True, drop_last=True, **comm_kwargs)
  else:
    loader_uns = None
  loader_test = DataLoader(data_test,
    batch_size=int(cfg.batch_size.test),
    shuffle=False, drop_last=False, **comm_kwargs)
  return DataLoaderTriplet(sup=loader_sup, uns=loader_uns, test=loader_test)


def get_datasets(cfg):
  with FileLock("./data.lock"):
    if cfg.dataset == 'cifar10':
      data_train = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True)
      data_test = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True)
    elif cfg.dataset == 'svhn':
      data_train = torchvision.datasets.SVHN(
        root=cfg.data_dir, split='train', download=True)
      data_extra = torchvision.datasets.SVHN(
        root=cfg.data_dir, split='extra', download=True)
      data_train = ConcatDataset([data_train, data_extra])
      data_test = torchvision.datasets.SVHN(
        root=cfg.data_dir, split='test', download=True)
    else:
      raise Exception(f'Invalid dataset: {cfg.dataset}')
  return data_train, data_test
