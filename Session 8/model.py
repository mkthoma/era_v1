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

class dynamic_norm_cnn(nn.Module):
    def __init__(self, use_batch_norm=False, use_layer_norm=False, use_group_norm=False, num_groups=2):
        super(dynamic_norm_cnn, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups

# ***************** BATCH NORMALIZAION ############################
        if self.use_batch_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Dropout(dropout_value)
          ) # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(20),
              nn.Dropout(dropout_value)
          ) # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(20),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(30),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(40),
              nn.Dropout(dropout_value)
          ) # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(20),
              nn.Dropout(dropout_value)
          ) # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(30),
              nn.Dropout(dropout_value)
          ) # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(40),
              nn.Dropout(dropout_value)
          ) # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          ) # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          ) # output_size = 1, RF=44


# ***************** LAYER NORMALIZAION ############################
        elif self.use_layer_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 10),
              nn.Dropout(dropout_value)
          )  # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 30),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 40),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 30),
              nn.Dropout(dropout_value)
          )  # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(1, 40),
              nn.Dropout(dropout_value)
          )  # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          )  # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 1, RF=44



# ***************** GROUP NORMALIZAION ############################
        elif self.use_group_norm:
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 10),
              nn.Dropout(dropout_value)
          )  # output_size = 30, RF=3

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 28, RF=5

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 28, RF=5
          self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 14, RF=6

          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 30),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=14
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=1, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 40),
              nn.Dropout(dropout_value)
          )  # output_size = 14, RF=18

          # TRANSITION BLOCK 2
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 14, RF=18
          self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 7, RF=20

          # CONVOLUTION BLOCK 3
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 20),
              nn.Dropout(dropout_value)
          )  # output_size = 5, RF=28
          self.convblock9 = nn.Sequential(
              nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 30),
              nn.Dropout(dropout_value)
          )  # output_size = 3, RF=36
          self.convblock10 = nn.Sequential(
              nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.GroupNorm(self.num_groups, 40),
              nn.Dropout(dropout_value)
          )  # output_size = 1, RF=44

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=1)
          )  # output_size = 1, RF=44

          self.convblock11 = nn.Sequential(
              nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
          )  # output_size = 1, RF=44


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
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

    # Calculate loss
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

    return misclassified_images[:10], misclassified_labels[:10], misclassified_predictions[:10]


def train_test_loss_accuracy(epochs):
    epoch_list = [ i+1 for i in range(epochs)]
    df = pd.DataFrame(epoch_list, columns=['epoch'])
    train_loss1 = [round(i,2) for i in train_loss]
    train_accuracy1 = [round(i,2) for i in train_accuracy]
    test_loss1 = [round(i,2) for i in test_losses]
    test_accuracy1 = [round(i,2) for i in test_acc]

    df['train_loss'] = train_loss1
    df['train_accuracy'] = train_accuracy1
    df['test_loss'] = test_loss1
    df['test_accuracy'] = test_accuracy1
    return df




########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#######################################################################        SESSION 7       #########################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

class session7_model(nn.Module):
    def __init__(self):
        super(session7_model, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24, RF=5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24, RF=5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, RF=6

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10, RF=10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 8, RF=14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6, RF=18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6, RF=22

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, RF=22

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 1, RF=22


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#######################################################################        SESSION 6       #########################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

class session6_model(nn.Module):
    def __init__(self):
        super(session6_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.tns1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(num_features=8)  
        self.pool1 = nn.MaxPool2d(2, 2)   
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(num_features=16) 
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1) 
        self.bn5 = nn.BatchNorm2d(num_features=64) 
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, padding=1)    
        self.gpool = nn.AvgPool2d(kernel_size=7)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop(self.bn3(F.relu(self.conv3(x))))        
        x = self.drop(self.bn4(F.relu(self.conv4(x))))
        x = self.tns2(self.pool2(x))
        x = self.drop(self.bn5(F.relu(self.conv5(x))))
        x = self.conv6(x)
        x = self.gpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#######################################################################        SESSION 5      #########################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

class session5_model(nn.Module):
    def __init__(self):
        super(session5_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(4096, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = F.relu(self.conv3(x), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)