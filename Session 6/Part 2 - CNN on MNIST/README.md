# Digit Classifiction using Convolutional Neural Networks
In this exercise, we will be using the MNIST data for classifying handwritten digits using a convolutional layer. We will stick to the following operational parameters as well to see how a CNN can be efficiently configured - 
- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- Use Batch Normalization and  Dropout,
- A Fully connected layer and have used GAP (Optional). 

## Model Architecture
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
```
-  Input: The model takes grayscale images as input, with a single channel.
- The first convolutional layer (conv1) performs a 2D convolution on the input, using a kernel size of 3x3 and generating 8 output channels.
- Batch normalization (bn1) is applied to the output of conv1.
- The output of bn1 is passed through a 2D dropout layer (drop) and then through the ReLU activation function (F.relu).
- The result is then passed through a 2D convolutional layer (tns1) with a kernel size of 1x1, generating 4 output channels.
- The output of tns1 is passed through the second convolutional layer (conv2), which has a kernel size of 3x3 and produces 8 output channels.
- Batch normalization (bn2) is applied to the output of conv2.
- The output is then passed through a 2D dropout layer (drop) and ReLU activation (F.relu).
- After bn2, a 2D max pooling layer (pool1) with a kernel size of 2x2 and a stride of 2 is applied to reduce the spatial dimensions of the feature maps by half.
- The output of pool1 is passed through two additional convolutional layers (conv3 and conv4), each followed by batch normalization (bn3 and bn4), dropout (drop), and ReLU activation (F.relu).
- conv3 has a kernel size of 3x3 and generates 16 output channels.
- conv4 has a kernel size of 3x3 and generates 32 output channels.
- After conv4, another 2D max pooling layer (pool2) with a kernel size of 2x2 and a stride of 2 is applied to further reduce the spatial dimensions.
- The output of pool2 is passed through a 2D convolutional layer (tns2) with a kernel size of 1x1, generating 16 output channels.
- The output of tns2 is passed through two more convolutional layers (conv5 and conv6), each followed by batch normalization (bn5) and ReLU activation (F.relu).
- conv5 has a kernel size of 3x3 and generates 64 output channels.
- conv6 has a kernel size of 1x1 and generates 10 output channels.
- After conv6, an average pooling layer (gpool) with a kernel size of 7x7 is applied to globally average the feature maps.
- A 2D dropout layer (drop) is applied with a dropout rate of 0.1.
- The output of gpool is reshaped using view to have a shape of (-1, 10), where -1 represents the batch size and 10 is the number of classes.
- The softmax function (F.log_softmax) is applied to obtain the final output probabilities for each class.

## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
         Dropout2d-3            [-1, 8, 28, 28]               0
            Conv2d-4            [-1, 4, 30, 30]              36
            Conv2d-5            [-1, 8, 30, 30]             296
       BatchNorm2d-6            [-1, 8, 30, 30]              16
         Dropout2d-7            [-1, 8, 30, 30]               0
         MaxPool2d-8            [-1, 8, 15, 15]               0
            Conv2d-9           [-1, 16, 15, 15]           1,168
      BatchNorm2d-10           [-1, 16, 15, 15]              32
        Dropout2d-11           [-1, 16, 15, 15]               0
           Conv2d-12           [-1, 32, 15, 15]           4,640
      BatchNorm2d-13           [-1, 32, 15, 15]              64
        Dropout2d-14           [-1, 32, 15, 15]               0
        MaxPool2d-15             [-1, 32, 7, 7]               0
           Conv2d-16             [-1, 16, 9, 9]             528
           Conv2d-17             [-1, 64, 9, 9]           9,280
      BatchNorm2d-18             [-1, 64, 9, 9]             128
        Dropout2d-19             [-1, 64, 9, 9]               0
           Conv2d-20           [-1, 10, 11, 11]             650
        AvgPool2d-21             [-1, 10, 1, 1]               0
================================================================
Total params: 16,934
Trainable params: 16,934
Non-trainable params: 0
----------------------------------------------------------------
```
Above we can see the model summary. Here we can note that the total parameters of the model is around 17K.

## Model Usage
1. Define the train and test dataloaders.

    ```python
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ```

- Train Loader:

    - The datasets.MNIST function is used to create a dataset object for the MNIST training set.
    - The dataset is located in the '../data' directory and is downloaded if necessary.
    - A transforms.Compose object is created to define a series of data transformations to be applied to each sample in the dataset.
    - The transformations include converting the images to tensors (transforms.ToTensor()) and normalizing the pixel values with mean 0.1307 and standard deviation 0.3081 (transforms.Normalize((0.1307,), (0.3081,))).
    - The dataset and transformations are passed as arguments to torch.utils.data.DataLoader to create a data loader.
    - The batch_size, shuffle, and other parameters (collected in the **kwargs variable) are specified to control the behavior of the data loader.

- Test Loader:
    - Similar to the train loader, the datasets.MNIST function is used to create a dataset object for the MNIST test set.
    - The same transforms.Compose object is used to apply the same transformations to the test set.
    - A data loader is created for the test set using torch.utils.data.DataLoader with the same parameters as the train loader.

2. Define train and test functions

    ```python
    from tqdm import tqdm
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
    ```
- train()
    - The code begins by importing the tqdm library, which provides a progress bar to track the training progress.
    - The train function is defined, taking the following parameters:
        - model: The model to be trained.
        - device: The device (CPU or GPU) on which the training will be performed.
        - train_loader: The data loader providing the training samples.
        - optimizer: The optimizer used for updating the model's parameters.
        - epoch: The current epoch number.
    - A progress bar is created using tqdm and is associated with the training data loader (pbar = tqdm(train_loader)). This progress bar will display the training progress throughout the epoch.
    - The function enters a loop that iterates over the mini-batches of training data using enumerate(train_loader).
    - For each batch, the input data and corresponding target labels are retrieved (data, target = data.to(device), target.to(device)), and they are moved to the specified device if available.
    - The optimizer's gradients are cleared using optimizer.zero_grad() to prepare for the backward pass.
    - The input data is passed through the model to obtain the output predictions (output = model(data)).
    - The negative log-likelihood loss (F.nll_loss) is computed by comparing the model's output with the target labels (loss = F.nll_loss(output, target)).
    - The gradients are computed by performing backpropagation through the network (loss.backward()).
    - The optimizer updates the model's parameters based on the computed gradients (optimizer.step()).
- test()
```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
- The functions takes the following as parameters-
    - model: The trained model to be evaluated.
    - device: The device (CPU or GPU) on which the evaluation will be performed.
    - test_loader: The data loader providing the test samples.
    Set the model to evaluation mode: model.eval()
- Initialize variables for tracking the test loss and the number of correct predictions: test_loss = 0 and correct = 0
- Disable gradient calculation and backpropagation during testing: with torch.no_grad():
- Testing Loop: Iterate over the mini-batches of test data: for data, target in test_loader:
- Move the input data and target labels to the specified device: data, target = data.to(device), target.to(device)
- Forward pass: Pass the input data through the model to obtain predictions: output = model(data)
- Compute the loss: Calculate the negative log-likelihood loss and sum it up: test_loss += F.nll_loss(output, target, reduction='sum').item()
- Calculate the number of correct predictions-
    - Get the predicted class labels by selecting the index of the highest log-probability: pred = output.argmax(dim=1, keepdim=True)
    - Compare the predicted labels with the target labels and count the number of correct predictions: correct += pred.eq(target.view_as(pred)).sum().item()
- Calculate the average test loss: test_loss /= len(test_loader.dataset)
- Display the average loss, the number of correct predictions, the total number of test samples, and the accuracy percentage: 
    - print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

This function evaluates the model's performance on the test dataset by calculating the average loss and accuracy. It iterates over the test samples, calculates the loss, and compares the predicted labels with the ground truth labels to compute the accuracy. The results are then printed to provide an overview of the model's performance on the test set.

3. Running the model
```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(0, 20):
    print("Epoch-", epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```
- Model and Optimizer Initialization:
    - The Net model is instantiated and sent to the specified device: model = Net().to(device).
    - The optimizer (Stochastic Gradient Descent - SGD) is created with a learning rate of 0.01 and momentum of 0.9: optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9).
- Training and Testing Loop:
    - A loop is executed for each epoch from 0 to 19: for epoch in range(0, 20).
    - The current epoch number is printed: print("Epoch-", epoch).
    - The train function is called, passing the model, device, training data loader, optimizer, and the current epoch number as arguments: train(model, device, train_loader, optimizer, epoch).
    - The test function is called, passing the model, device, and test data loader as arguments: test(model, device, test_loader).
    - The testing function evaluates the model's performance on the test set and prints the average loss and accuracy.
- This code trains the model for 20 epochs and performs testing after each epoch. By printing the epoch number, it provides visibility into the training progress. The training and testing functions are called in each epoch to update the model's parameters and evaluate its performance on the test set.