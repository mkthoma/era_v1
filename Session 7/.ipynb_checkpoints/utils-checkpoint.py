import torch
from torchvision import datasets, transforms

# defining the dataloader
def train_test_dataloader(train_transformer, test_transformer, dataloader_args):
    # Train data transformations
    train_transforms = transforms.Compose(train_transformer)
    # # Test Phase transformations
    test_transforms = transforms.Compose(test_transformer)

    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader