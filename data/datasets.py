from torch.utils.data import Dataset


class TransformedDataset(Dataset):
  def __init__(self, dataset, transforms):
    self.dataset = dataset
    self.transforms = transforms

  def __getitem__(self, index):
    img, label = self.dataset[index]
    return self.transforms(img), label

  def __len__(self):
    return len(self.dataset)


class BiTransformedDataset(Dataset):
  def __init__(self, dataset, transforms_1, transforms_2):
    self.dataset = dataset
    self.transforms_1 = transforms_1
    self.transforms_2 = transforms_2

  def __getitem__(self, index):
    img, _ = self.dataset[index]
    return self.transforms_1(img), self.transforms_2(img)

  def __len__(self):
    return len(self.dataset)
