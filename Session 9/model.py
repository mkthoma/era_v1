import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
import pandas as pd


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#######################################################################        SESSION 8       #########################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):  # Add self parameter here
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF=3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF=5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 28, RF=9


        self.transblock1 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 28, RF=9


        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 28, RF=11

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 28, RF=13

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, dilation=4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 20, RF=21


        self.transblock2 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 20, RF=21


        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 20, RF=23

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 20, RF=25

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, dilation=8, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 4, RF=41


        self.transblock3 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 4, RF=41

        # CONVOLUTION BLOCK 4
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, groups=16, bias=False),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 4, RF=43

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 4, RF=45

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), padding=0, dilation=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 2, RF=47


        # OUTPUT BLOCK
        # Global Average Pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        )

        # fully connected layer
        self.convblock13 = nn.Sequential(
            nn.Linear(in_features=64, out_features=10, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.transblock1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.transblock2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.transblock3(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock12(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.convblock13(x)
        return F.log_softmax(x, dim=-1)


def model_summary(model, input_size):
    summary(model, input_size)

train_losses = []
test_losses = []
train_acc = []
test_acc = []
train_loss = []
train_accuracy = []


def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
        # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)
    y_pred = y_pred.view(target.size(0), -1)

    # Calculate loss
    loss = F.nll_loss(y_pred, target.squeeze())
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Batch_id={batch_idx}')
    train_acc.append(100*correct/processed)

  train_accuracy.append(train_acc[-1])
  train_loss.append([x.item() for x in train_losses][-1])
  print(f"Train Accuracy: {round(train_accuracy[-1], 2)}%")
  print("Train Loss: ", round(train_loss[-1],2))




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(target.size(0), -1)
            test_loss += F.nll_loss(output, target.squeeze(), reduction='sum').item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Check for misclassified images
            misclassified_mask = ~pred.eq(target.view_as(pred)).squeeze()
            misclassified_images.extend(data[misclassified_mask])
            misclassified_labels.extend(target.view_as(pred)[misclassified_mask])
            misclassified_predictions.extend(pred[misclassified_mask])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    print(f"Test Accuracy: {round(test_acc[-1], 2)}%")
    print("Test Loss: ", round(test_losses[-1],2))
    print("\n")


    return misclassified_images[:10], misclassified_labels[:10], misclassified_predictions[:10]


def train_test_loss_accuracy(epochs):
    epoch_list = [ i+1 for i in range(epochs)]
    df = pd.DataFrame(epoch_list, columns=['epoch'])
    train_loss1 = [round(i,2) for i in train_loss]
    train_accuracy1 = [round(i,2) for i in train_accuracy]
    test_loss1 = [round(i,2) for i in test_losses]
    test_accuracy1 = [round(i,2) for i in test_acc]

    df['train_loss'] = train_loss1
    df['train_accuracy_%'] = train_accuracy1
    df['test_loss'] = test_loss1
    df['test_accuracy_%'] = test_accuracy1
    return df

