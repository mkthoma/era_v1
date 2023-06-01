# Digit classification using MNIST and PyTorch

This folder contains Python code for training and evaluating a PyTorch model for digit classification using MNIST dataset. The code provides functions for training the model, evaluating its performance on test data, and generating a model summary.

## Model ([model.py](https://github.com/mkthoma/era_v1/blob/main/session5/model.py))
This folder contains the code for a Convolutional Neural Network (CNN) model implemented using PyTorch. The model consists of four convolutional layers, two max pooling layers, and two fully connected layers. The model is defined in the `Net` class, which inherits from the `nn.Module` base class. Here is a summary of the layers in the model:

### 1. Convolutional Layer 1:
   - Input: 1 channel (grayscale image)
   - Output: 32 channels
   - Kernel Size: 3x3
   - Activation Function: ReLU

### 2. Convolutional Layer 2:
   - Input: 32 channels
   - Output: 64 channels
   - Kernel Size: 3x3
   - Activation Function: ReLU
   - Max Pooling: 2x2

### 3. Convolutional Layer 3:
   - Input: 64 channels
   - Output: 128 channels
   - Kernel Size: 3x3
   - Activation Function: ReLU

### 4. Convolutional Layer 4:
   - Input: 128 channels
   - Output: 256 channels
   - Kernel Size: 3x3
   - Activation Function: ReLU
   - Max Pooling: 2x2

### 5. Fully Connected Layer 1:
   - Input: 4096 (flattened feature map)
   - Output: 50
   - Activation Function: ReLU

### 6. Fully Connected Layer 2 (Output Layer):
   - Input: 50
   - Output: 10 (number of classes)
   - Activation Function: None

### 7. Log Softmax Activation:
   - Applied to the output of the final fully connected layer.


## Functions used ([utils.py](https://github.com/mkthoma/era_v1/blob/main/session5/utils.py))

### GetCorrectPredCount()
- Calculates the count of correct predictions given predicted values (`pPrediction`) and corresponding ground truth labels (`pLabels`).

### train()
- Performs the training loop for the model.
- Inputs:
  - `model`: The PyTorch model to be trained.
  - `device`: The device to be used for training (CPU or GPU).
  - `train_loader`: The data loader containing the training data.
  - `optimizer`: The optimizer used for updating the model's parameters.
- Functionality:
  - Iterates over the training data and performs the following steps:
    - Moves the data and labels to the specified device.
    - Resets the optimizer's gradients.
    - Performs a forward pass to obtain predictions.
    - Calculates the loss between the predictions and the ground truth labels.
    - Performs backpropagation to compute gradients.
    - Updates the model's parameters using the optimizer.
    - Keeps track of training loss, accuracy, and displays progress using tqdm.

### test()
- Evaluates the trained model on the test data.
- Inputs:
  - `model`: The trained PyTorch model to be evaluated.
  - `device`: The device to be used for evaluation (CPU or GPU).
  - `test_loader`: The data loader containing the test data.
- Functionality:
  - Sets the model to evaluation mode (disables gradient computation).
  - Iterates over the test data and performs the following steps:
    - Moves the data and labels to the specified device.
    - Performs a forward pass to obtain predictions.
    - Calculates the test loss between the predictions and the ground truth labels.
    - Counts the number of correct predictions.
  - Prints the average loss and accuracy on the test set.

### model_summary()
- Generates a summary of the model's architecture and parameter count.
- Inputs:
  - `model`: The PyTorch model for which to generate the summary.
- Outputs:
  - Summary of the model's architecture and parameter count.
- Functionality:
  - Uses the `torchsummary` library to generate a summary of the model.
  - Returns the summary, which includes the input size and the number of parameters in each layer of the model.

## [Usage](https://github.com/mkthoma/era_v1/blob/main/session5/S5.ipynb)
1. Import the required libraries and functions.
``` python
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from model import Net
import utils
from utils import train, test, model_summary
```
2. Check if CUDA is available and set the device accordingly
```python
cuda =torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print("CUDA Available?", cuda)
```
3. Create data loaders for training and test data.
```python
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```
4. Download the MNIST dataset and apply the defined transformations:
``` python
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
```
5. Set the batch size and create data loaders for training and testing:
``` python
batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(train_data, kwargs)
test_loader = torch.utils.data.DataLoader(test_data, kwargs)
```
6. Training and Evaluation
    - Create an instance of the model and define the optimizer.
        ``` python
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        ```
    -  Define a learning rate scheduler.
        ``` python
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, verbose=True)
        num_epochs = 20
        ```
    -  Set the number of epochs and start the training loop:
        ``` python
        for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
        ```
7. Plotting the training and test accuracy/loss:
``` python
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(utils.train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(utils.train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(utils.test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(utils.test_acc)
axs[1, 1].set_title("Test Accuracy")
```
8. Generating model summary
``` python
model_summary(model)
```

Feel free to modify the code and adapt it to your specific needs!

