# Custom Resnet Architecture on CIFAR 10 using Pytorch 
In this exercise we will be looking to implement a custom ResNet architecture on the CIFAR 10 dataset using PyTorch Lightning.

## Objective
We will try to build a custom Resnet Architecture on the CIFAR10 dataset. We will try to achieve this in 24 epochs. This experimentation is similar to David Page's [exercise](https://github.com/davidcpage/cifar10-fast).

The live implementation can be found on HuggingFace [here](https://huggingface.co/spaces/mkthoma/cifar10_customResNet).
## CIFAR 10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The 10 classes in CIFAR-10 are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

CIFAR-10 is a widely used benchmark dataset in the field of computer vision and deep learning. It consists of 60,000 color images, each of size 32x32 pixels, belonging to 10 different classes. The dataset is divided into 50,000 training images and 10,000 testing images.

The images in CIFAR-10 are relatively low-resolution compared to some other datasets, making it a challenging task for machine learning models to accurately classify the images. The dataset is commonly used for tasks such as image classification, object detection, and image segmentation.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Let us look at some samples of the CIFAR 10 dataset.

![image](https://github.com/mkthoma/era_v1/assets/135134412/a77717b6-4e66-4a9b-b834-39c6c0dbfff3)

Looking at different samples of each class

![image](https://github.com/mkthoma/era_v1/assets/135134412/de11336d-f794-4bc9-b99a-24ca047a94ba)

The colour and RGB representation of the same image can be seen below 

![image](https://github.com/mkthoma/era_v1/assets/135134412/94b8b427-92ef-4365-8690-b5ccad399857)

## PyTorch Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight PyTorch wrapper that provides a high-level abstraction for training PyTorch models. It simplifies the process of building, training, and evaluating complex deep learning models, making it easier to develop robust and scalable code for research and production purposes. Here are some key reasons why PyTorch Lightning is widely used in the deep learning community:

1. Simplified Training Loop: PyTorch Lightning abstracts away the boilerplate code for the training loop, validation loop, and testing loop, reducing the amount of code needed to train a model. This allows researchers and developers to focus more on model architecture and hyperparameter tuning.

2. Readability and Maintainability: By separating the model architecture from the training loop and other components, PyTorch Lightning code becomes more organized, modular, and easier to read and maintain. This is especially beneficial when working on large projects with multiple collaborators.

3. Reproducibility: PyTorch Lightning provides a standardized training loop, ensuring that the training process is consistent across different runs. This contributes to better reproducibility of experiments and results.

4. Automatic Hardware Acceleration: PyTorch Lightning automatically handles hardware acceleration using GPUs or TPUs without requiring explicit code changes. This simplifies the process of leveraging available hardware resources for faster training.

5. Integration with Distributed Training: PyTorch Lightning seamlessly integrates with distributed training frameworks, such as PyTorch Distributed Data Parallel (DDP), making it easy to scale training across multiple GPUs or machines.

6. Experiment Logging and Metrics Reporting: PyTorch Lightning integrates with various experiment logging and metrics reporting platforms, such as TensorBoard, Neptune, or Comet, enabling easy tracking and visualization of training progress and results.

7. Support for Multiple Research Frameworks: PyTorch Lightning is designed to support multiple deep learning research frameworks like PyTorch, PyTorch Lightning Bolts, and others. This ecosystem simplifies sharing and reusing components across different projects.

8. Community and Ecosystem: PyTorch Lightning has a large and active community, which contributes to its continuous development and improvement. The ecosystem includes a range of pre-built components and extensions, such as Lightning Bolts, that can be readily used for various tasks.

In summary, PyTorch Lightning is a powerful tool that abstracts away much of the low-level boilerplate code in PyTorch, allowing researchers and developers to focus on the core aspects of deep learning model development. Its simplicity, modularity, and reproducibility make it an attractive choice for training and evaluating deep learning models efficiently and effectively.


## Model Architecture

The model consists of 5 convolutional blocks. The output block consists of a fully connected layer.

```python
# PREPARATION BLOCK
self.prepblock = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(64))
# output_size = 32, RF=3


# CONVOLUTION BLOCK 1
self.convblock1_l1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
# output_size = 32, RF=5
nn.MaxPool2d(2, 2),nn.ReLU(),nn.BatchNorm2d(128))
# output_size = 16, RF=6

self.convblock1_r1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),nn.ReLU(),nn.BatchNorm2d(128),
# output_size = 16, RF=10
nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),nn.ReLU(),nn.BatchNorm2d(128))
# output_size = 16, RF=14


# CONVOLUTION BLOCK 2
self.convblock2_l1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
# output_size = 16, RF=18
nn.MaxPool2d(2, 2),nn.ReLU(),nn.BatchNorm2d(256))
# output_size = 8, RF=20


# CONVOLUTION BLOCK 3
self.convblock3_l1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),
# output_size = 8, RF=28
nn.MaxPool2d(2, 2),
nn.ReLU(),nn.BatchNorm2d(512))
# output_size = 4, RF=32


self.convblock3_r2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),nn.ReLU(),nn.BatchNorm2d(512),
# output_size = 4, RF=48
nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, dilation=1, stride=1, bias=False),nn.ReLU(),nn.BatchNorm2d(512))
# output_size = 4, RF=64


# CONVOLUTION BLOCK 4
self.convblock4_mp = nn.Sequential(nn.MaxPool2d(4))
# output_size = 1, RF = 88


# OUTPUT BLOCK - Fully Connected layer
self.output_block = nn.Sequential(nn.Linear(in_features=512, out_features=10, bias=False))
# output_size = 1, RF = 88


def forward(self, x):

    # Preparation Block
    x1 = self.prepblock(x)

    # Convolution Block 1
    x2 = self.convblock1_l1(x1)
    x3 = self.convblock1_r1(x2)
    x4 = x2 + x3

    # Convolution Block 2
    x5 = self.convblock2_l1(x4)

    # Convolution Block 3
    x6 = self.convblock3_l1(x5)
    x7 = self.convblock3_r2(x6)
    x8 = x7 + x6

    # Convolution Block 4
    x9 = self.convblock4_mp(x8)

    # Output Block
    x9 = x9.view(x9.size(0), -1)
    x10 = self.output_block(x9)
    return F.log_softmax(x10, dim=1)
```

The model architecture consists of several convolutional blocks with varying numbers of layers. The architecture and how it relates to ResNet concepts can be described as:

1. `Preparation Block`: This block consists of a single convolutional layer followed by ReLU activation and batch normalization. It serves as the initial processing of the input image.

2. `Convolution Block 1`: This block starts with a convolutional layer, followed by a max-pooling layer. Then, another set of convolutional layers is applied. The output of the first convolutional layer is added element-wise to the output of the second set of convolutional layers. This addition introduces a residual connection-like concept.

3. `Convolution Block 2`: This block follows a similar structure to Block 1. It includes a convolutional layer, a max-pooling layer, and no residual connection.

4. `Convolution Block 3`: This block is similar to Block 2 but includes an additional set of convolutional layers. Again, there is no explicit residual connection.

5. `Convolution Block 4`: This block consists of a max-pooling layer only.

6. `Output Block`: The output block includes a fully connected layer that maps the flattened features to the output classes. It uses a log-softmax activation function to produce the final class probabilities.

The calculations for Receptive field and output channel size are shown below:

![image](https://github.com/mkthoma/era_v1/assets/135134412/2caa50e3-ec18-4293-868a-54182fdaf189)

The model summary is as show below:

![image](https://github.com/mkthoma/era_v1/assets/135134412/a0875bfd-916a-4c30-b67b-061e538a2328)


The final model can be visualized as: 

![image](https://github.com/mkthoma/era_v1/assets/135134412/98a96be0-0d2e-4dea-b664-21f22882da1a)

## Data Preparation and Dataloader

The CIFAR-10 dataset is automatically downloaded and prepared for training, validation, and testing. The data is split into training and validation sets during the setup phase. Data augmentation is applied to the training set, including random cropping, random horizontal flipping, and normalization. The testing set is normalized only.

## Training and Evaluation

The model is trained using the Adam optimizer with a learning rate of 0.001. The training process is defined in the training_step, validation_step, and test_step methods, where the loss and accuracy metrics are computed. The configure_optimizers method sets up the optimizer for training.

## Grad-CAM Visualization

The custom_ResNet class provides a method to visualize misclassified images using Grad-CAM (Gradient Class Activation Maps). The show_misclassified_images method collects misclassified images from the test set, calculates Grad-CAM activations for each image, and displays them along with their true and predicted labels.

A sample output for misclassification is shown below
![image](https://github.com/mkthoma/era_v1/assets/135134412/80cb1ce1-2941-4286-9874-8d22f8e1c6d1)


## Usage
To use the custom_ResNet model for training and evaluation, follow these steps:

- Set the data_dir variable to the directory where the CIFAR-10 dataset will be stored.
- Instantiate the custom_ResNet class, specifying the data_dir if needed.
- Call the trainer.fit(model) method to train the model.
- Evaluate the model on the test set using the trainer.test(model) method.
- Optionally, use the show_misclassified_images method to visualize misclassified images.

The test accuracy is shown below:

![image](https://github.com/mkthoma/era_v1/assets/135134412/cd3820f7-1acc-4f92-bd61-f6da871b675a)


## Tensorboard Logs

TensorBoard is a powerful visualization tool provided by TensorFlow that allows researchers and developers to track and visualize various aspects of their deep learning models during training and evaluation. While PyTorch Lightning is built on top of PyTorch, it offers seamless integration with TensorBoard, making it easy to log and visualize important metrics and other information during the training process. 

In your PyTorch Lightning LightningModule, you can log various metrics during training and evaluation using the self.log() method. This method allows you to log metrics like loss, accuracy, and any other custom metrics you define.

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.cross_entropy(y_hat, y)
    pred = y_hat.argmax(dim=1, keepdim=True)
    acc = pred.eq(y.view_as(pred)).float().mean()
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
    return loss

def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.cross_entropy(y_hat, y)
    pred = y_hat.argmax(dim=1, keepdim=True)
    acc = pred.eq(y.view_as(pred)).float().mean()
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_acc', acc, prog_bar=True)
    return loss

def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.cross_entropy(y_hat, y)
    pred = y_hat.argmax(dim=1, keepdim=True)
    acc = pred.eq(y.view_as(pred)).float().mean()
    self.log('test_loss', loss, prog_bar=True)
    self.log('test_acc', acc, prog_bar=True)
    return pred
```
Once you have logged the metrics, you can start TensorBoard to visualize the training progress. The %tensorboard command with the --logdir flag points TensorBoard to the directory where your logs are stored.

```python
%tensorboard --logdir logs/
```

With TensorBoard running, open the provided link in your browser to access the TensorBoard dashboard. There, you can see various plots, such as loss curves, learning rate schedules, and other custom metrics. You can also visualize model architectures, gradients, and other useful information.

Using TensorBoard logs with PyTorch Lightning allows you to gain valuable insights into your model's performance and training dynamics. It helps you make informed decisions regarding hyperparameter tuning, model architecture adjustments, and other optimization strategies.

Some of the sample logs are shown below:
1. Training loss and accuracy

    ![image](https://github.com/mkthoma/era_v1/assets/135134412/a6bf3718-ce87-4d7b-a5ea-57088f7bf465)

    ![image](https://github.com/mkthoma/era_v1/assets/135134412/e737efff-39ba-440d-869a-be966acdc4bd)

2. Test loss and accuracy
    
    ![image](https://github.com/mkthoma/era_v1/assets/135134412/a5e0bd4a-d4f7-4005-8bfe-7596a4c031ec)

    ![image](https://github.com/mkthoma/era_v1/assets/135134412/2132c467-a97d-4be7-9e07-601120ee0390)



## Conclusion
In conclusion, using PyTorch Lightning for CIFAR-10 image classification, along with Grad-CAM visualization and TensorBoard logging, offers several significant advantages:

1. Simplified Model Development: PyTorch Lightning abstracts away the training loop, enabling us to focus on designing and refining the model architecture. This simplifies the development process and makes the code more organized and maintainable.

2. Efficient Training: With PyTorch Lightning's automatic hardware acceleration, we can seamlessly utilize GPUs or TPUs, accelerating the training process and reducing the time required to train complex models on large datasets like CIFAR-10.

3. Reproducibility: The standardized training loop in PyTorch Lightning ensures that experiments can be easily reproduced, aiding in validating and comparing different model configurations and hyperparameters.

4. Grad-CAM Visualization: The integration of Grad-CAM visualization with the custom ResNet model allows us to gain insights into the model's decision-making process. Visualizing misclassified images along with their corresponding Grad-CAM activations helps identify potential weaknesses in the model and improve its performance.

5. TensorBoard Logging: Leveraging TensorBoard for logging training and validation metrics enables us to monitor the model's progress in real-time. The interactive visualization of loss curves, accuracy trends, and other custom metrics empowers us to make informed decisions about model adjustments and hyperparameter tuning.

6. Experiment Tracking: TensorBoard's capabilities for experiment tracking and visualization make it easier to compare different model runs, analyze the impact of hyperparameters, and identify potential overfitting or convergence issues.

7. Community Support: Both PyTorch Lightning and TensorBoard have vibrant communities, providing extensive documentation, tutorials, and user support. This active community ensures that we can find solutions to issues and stay updated with the latest advancements in deep learning research.

By combining PyTorch Lightning, Grad-CAM, and TensorBoard, we create a powerful and streamlined workflow for developing, training, and analyzing deep learning models for image classification tasks. This approach not only enhances productivity but also empowers us to build more robust and accurate models for challenging datasets like CIFAR-10.
