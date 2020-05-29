from torchvision.transforms import transforms
from augment.simple import BaseAugment, DefaultAugment
from augment.randaugment import RandAugment
from augment.cutout import CutoutAugment


def get_transforms(dataset, base_method, aug_default, aug_cutout, randaug_args):
  trans_sup = []  # transforms for supervised training set
  trans_uns = []  # transforms for unsupervised training set
  trans_test = []  # transforms for test set

  # Advanced augmentation (RandAugment)
  if base_method == 'sup':    # Supervised
    pass
  elif base_method == 'ra':   # RandAugment
    trans_sup.append(RandAugment(*randaug_args))
  elif base_method == 'uda':  # UDA
    trans_uns.append(RandAugment(*randaug_args))
  else:
    raise Exception(f'Invalid method.base: {base_method}')

  # Simple augmentation (by default)
  if aug_default:
    trans_sup.append(DefaultAugment())
    trans_uns.append(DefaultAugment())

  # Basic augmentation (Normalization)
  trans_sup.append(BaseAugment(dataset))
  trans_uns.append(BaseAugment(dataset))
  trans_test.append(BaseAugment(dataset))

  # Cutout augmentation
  if aug_cutout > 0:
    trans_sup.append(CutoutAugment(aug_cutout))
    trans_uns.append(CutoutAugment(aug_cutout))

  # Compose
  trans_sup = transforms.Compose(trans_sup)
  trans_uns = transforms.Compose(trans_uns)
  trans_test = transforms.Compose(trans_test)
  return trans_sup, trans_uns, trans_test
