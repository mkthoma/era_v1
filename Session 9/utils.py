import torch
import numpy as np
from torchvision import datasets
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2

# This class will download the cifat10 dataset and apply the transformations for train and test data
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        if transform == "train":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=self._get_dataset_mean(), mask_fill_value=None),
                ToTensorV2()
            ])
        elif transform == "test":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                ToTensorV2()
            ])

        else:
            self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, label

    def _get_dataset_mean(self):
        # Calculate the mean of the dataset
        return tuple(self.data.mean(axis=(0, 1, 2)) / 255)

# function for train and test dataloader
def train_test_dataloader(dataloader_args):

  train_data = Cifar10SearchDataset(train=True, download=True, transform="train")
  test_data = Cifar10SearchDataset(train=False, download=True, transform="test")

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
  
  # distinct classes of images
  classes = train_data.classes
  print("Unique classes of images are:", classes)

  return train_loader, test_loader, classes