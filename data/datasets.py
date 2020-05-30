from itertools import product
import logging
import multiprocessing as mp
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


log = logging.getLogger('main')


class TransformedDataset(Dataset):
  def __init__(self, cfg, dataset, transforms, save_name):
    self.cfg = cfg
    self.dataset = dataset
    self.transforms = transforms
    self.preprocessed = False
    if cfg.uda.preproc_epochs and cfg.uda.preproc_epochs > 0:
      self._preprocess(save_name)
      self.preprocessed = True

  def _preprocess(self, save_name):
    save_dir = f'{self.cfg.data_dir}/augmented'
    save_file = f'{save_dir}/{save_name}'
    if os.path.exists(save_file):
      self.dataset = torch.load(save_file)
      log.warning(f'Augmented data loaded from: {save_file}')
    else:
      transformed = []
      for img, label in tqdm(self.dataset,
                             desc=f'Save augmented images to {save_file}'):
        transformed.append((self.transforms(img), label))
      self.dataset = transformed
      if not os.path.exists(save_dir):
        os.mkdir(save_dir)
      torch.save(self.dataset, save_file)

  def __getitem__(self, index):
    if self.preprocessed:
      return self.dataset[index]
    else:
      img, label = self.dataset[index]
      return self.transforms(img), label

  def __len__(self):
    return len(self.dataset)


class BiTransformedDataset(Dataset):
  def __init__(self, cfg, dataset, transforms_1, transforms_2, save_name):
    self.cfg = cfg
    self.dataset = dataset
    self.dataset_len = len(self.dataset)
    self.transforms_1 = transforms_1
    self.transforms_2 = transforms_2
    if cfg.uda.preproc_epochs and cfg.uda.preproc_epochs > 0:
      self._preprocess(save_name)
    else:
      log.warning(f'Real-time augmentation for unsupervised data.')

  def _preprocess(self, save_name):
    # e.g. save_dir: ./data/augmented/uns_n2_m3/trans_1_
    postfix = f'labeled_{self.cfg.n_labeled}_ep_{self.cfg.uda.preproc_epochs}'
    postfix += f'_n{round(self.cfg.randaug.n)}_m{round(self.cfg.randaug.m)}'
    self.save_dir = f'{self.cfg.data_dir}/augmented/{save_name}_{postfix}'
    self.n_epochs = self.cfg.uda.preproc_epochs
    self.n_imgs_per_epoch = len(self.dataset)

    if not os.path.exists(f'{self.save_dir}/completed'):
      if not os.path.exists(self.save_dir):
        os.mkdir(self.save_dir)
      # preprocess simple augmentation
      log.info(f'Save augmented (by trans 1) images to: {self.save_dir}')
      id_list = list(range(self.n_imgs_per_epoch))
      results = self._multi_process(self.process_fn_1, id_list)
      # preprocess RandAugment
      log.info(f'Save augmented (by trans 2) images to: {self.save_dir}')
      id_list = list(range(self.n_epochs))
      self._multi_process(self.process_fn_2, id_list)
      torch.save(0, f'{self.save_dir}/completed')
    else:
      log.warning(f'Augmented data will be loaded from: {self.save_dir}')

    self.dataset = None

  def process_fn_1(self, args):
    i, img_ids = args
    for j in tqdm(iterable=img_ids, total=len(img_ids), position=i,
                  desc=f'Preprocessing worker {i}'):
      save_file = f'{self.save_dir}/trans_1_{j}'
      if not os.path.exists(save_file):
        torch.save(self.transforms_1(self.dataset[j][0]), save_file)

  def process_fn_2(self, args):
    i, ep_ids = args
    for k, (j, (img, _)) in tqdm(
        iterable=product(ep_ids, enumerate(self.dataset)),
        total=self.n_imgs_per_epoch * len(ep_ids), position=i,
        desc=f'Preprocessing worker {i}'):
      save_file = f'{self.save_dir}/trans_2_{j}_{k}'
      if not os.path.exists(save_file):
        torch.save(self.transforms_2(img), save_file)

  def _multi_process(self, process_fn, num_ids):
    num_process = min(self.cfg.uda.preproc_workers, mp.cpu_count())
    process_args = [(i, num_ids[i::num_process]) for i in range(num_process)]
    if self.cfg.debug:
      log.debug('Start preprocessing with a single worker.')
      results = [process_fn(process_args[0])]
    else:
      log.info(f'Start preprocessing with {num_process} processes.')
      pool = mp.Pool(processes=num_process)
      results = pool.map(process_fn, process_args)
      pool.close()
      pool.join()
    log.info('Preprocessing completed.')
    return results

  def __getitem__(self, index):
    if self.cfg.uda.preproc_epochs and self.cfg.uda.preproc_epochs > 0:
      epoch = random.randint(0, self.n_epochs - 1)
      return (torch.load(f'{self.save_dir}/trans_1_{index}'),
              torch.load(f'{self.save_dir}/trans_2_{index}_{epoch}'))
    else:
      img, _ = self.dataset[index]
      return self.transforms_1(img), self.transforms_2(img)

  def __len__(self):
    return self.dataset_len
