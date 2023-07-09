import torch
import numpy as np
from torchvision import datasets
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch_lr_finder import LRFinder

# Class for cifar10 dataset and augmentations using Albumetations library
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        if transform == "train":
            self.transform = A.Compose([
                A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT),
                A.RandomCrop(height=32, width=32),
                A.HorizontalFlip(),
                A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=self._get_dataset_mean()),
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

# Function for train and test dataloader
def dataloader(dataset, dataloader_args):
    data_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return data_loader


# Max LR using LRFinder
def find_max_lr(optimizer, criterion, model, train_loader, end_lr, num_iter, step_mode): 
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
    _,max_LR = lr_finder.plot()
    lr_finder.reset()
    return max_LR