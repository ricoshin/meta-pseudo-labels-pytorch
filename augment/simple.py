from torchvision.transforms import transforms


class BasicAugment:
  def __init__(self, dataset):
    self.transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(*self.get_mean_and_std(dataset)),
    ])

  def __call__(self, img):
    return self.transforms(img)

  def get_mean_and_std(self, dataset):
    if dataset == "cifar10":
      means = [0.49139968, 0.48215841, 0.44653091]
      stds = [0.24703223, 0.24348513, 0.26158784]
    elif dataset == "svhn":
      means = [0.4376821, 0.4437697, 0.47280442]
      stds = [0.19803012, 0.20101562, 0.19703614]
    else:
      raise Exception(f'Unknown dataset: {dataset}')
    return means, stds


class DefaultAugment:
  def __init__(self, crop_size=32, pad_size=4):
    self.transforms = transforms.Compose([
      transforms.RandomCrop(crop_size, padding=pad_size),
      transforms.RandomHorizontalFlip(),
    ])

  def __call__(self, img):
    return self.transforms(img)
