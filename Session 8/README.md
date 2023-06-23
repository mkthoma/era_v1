# Applying Normalization techniques on CIFAR 10 Dataset using Pytorch and CNN's
In this exercise we will be looking to implement the different normalization techniques such as Batch, Group and Layer on the CIFAR 10 dataset using PyTorch. 

## Normalization techniques

Layer normalization, group normalization, and batch normalization are techniques used in deep learning to normalize the activations of neurons in a neural network. They help address the issue of internal covariate shift, which refers to the change in the distribution of input values to a layer during training.

1. `Batch Normalization`:

    Batch normalization computes the mean and variance of activations across a mini-batch of training examples. It normalizes the activations using these batch-level statistics, making the network more robust to changes in the distribution of inputs. Batch normalization helps in reducing the internal covariate shift and accelerates the training process by allowing higher learning rates. It also acts as a regularizer, reducing the reliance on other regularization techniques like dropout.
2. `Layer Normalization`:

    Layer normalization normalizes the activations of neurons within a layer independently. It computes the mean and variance of the activations across the spatial dimensions (for convolutional layers) or feature dimensions (for fully connected layers) and normalizes each neuron's activation based on these statistics. This ensures that the activations have zero mean and unit variance.

3. `Group Normalization`:

    Group normalization extends the idea of layer normalization by dividing the channels (feature dimensions) of a layer into groups and normalizing each group separately. It reduces the computational and memory requirements compared to layer normalization when dealing with large feature dimensions. Group normalization is effective when the batch size is small or the spatial dimensions are large.

Batch normalization is commonly used and has been widely adopted in various neural network architectures. It has proven to be effective in improving training stability and accelerating convergence. Layer normalization and group normalization are alternative normalization techniques that can be useful in specific scenarios where batch normalization might not be suitable, such as recurrent neural networks or situations with small batch sizes.

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

![cifar random](https://github.com/mkthoma/era_v1/assets/135134412/8d1980c4-1261-4255-8e77-1b0d6ce52d9a)

Looking at different samples of each class

![cifar image per class](https://github.com/mkthoma/era_v1/assets/135134412/757e0b9f-3a38-4cef-85b7-64e51b2895f8)

The colour and RGB representation of the same image can be seen below 

![cifar rgb](https://github.com/mkthoma/era_v1/assets/135134412/7dc581d1-3fbb-4219-a11c-f3ca18db1a6d)

## Objective
We will try to apply a CNN with less than 50k parameters, apply batch, layer and normalization and try to achieve more than 70% accuracy within 20 epochs.

## Model Architecture
We are using 3 convolutional blocks and 2 transition blocks as shown below:
```python
# Input Block
self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 30, RF=3

# CONVOLUTION BLOCK 1
self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 28, RF=5

# TRANSITION BLOCK 1
self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),) # output_size = 28, RF=5

self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF=6

# CONVOLUTION BLOCK 2
self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 14, RF=10

self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=1, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 14, RF=14

self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=1, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 14, RF=18

# TRANSITION BLOCK 2
self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),) # output_size = 14, RF=18

self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, RF=20

# CONVOLUTION BLOCK 3
self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 5, RF=28

self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding=0, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 3, RF=36

self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3), padding=0, bias=False),
nn.ReLU(),nn.Dropout(dropout_value)) # output_size = 1, RF=44

# OUTPUT BLOCK
self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=1)) # output_size = 1, RF=44

self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=40, out_channels=10, kernel_size=(1, 1),padding=0, bias=False),) # output_size = 1, RF=44
```

The model can be visualized as: 

![image](https://github.com/mkthoma/era_v1/assets/135134412/675507fd-7bf7-40c0-85e6-1ddaf9f7c6ab)

The model summary shows us that we are only using about 39k parameters for this model.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 30, 30]             270
              ReLU-2           [-1, 10, 30, 30]               0
         GroupNorm-3           [-1, 10, 30, 30]              20
           Dropout-4           [-1, 10, 30, 30]               0
            Conv2d-5           [-1, 20, 28, 28]           1,800
              ReLU-6           [-1, 20, 28, 28]               0
         GroupNorm-7           [-1, 20, 28, 28]              40
           Dropout-8           [-1, 20, 28, 28]               0
            Conv2d-9           [-1, 10, 28, 28]             200
        MaxPool2d-10           [-1, 10, 14, 14]               0
           Conv2d-11           [-1, 20, 14, 14]           1,800
             ReLU-12           [-1, 20, 14, 14]               0
        GroupNorm-13           [-1, 20, 14, 14]              40
          Dropout-14           [-1, 20, 14, 14]               0
           Conv2d-15           [-1, 30, 14, 14]           5,400
             ReLU-16           [-1, 30, 14, 14]               0
        GroupNorm-17           [-1, 30, 14, 14]              60
          Dropout-18           [-1, 30, 14, 14]               0
           Conv2d-19           [-1, 40, 14, 14]          10,800
             ReLU-20           [-1, 40, 14, 14]               0
        GroupNorm-21           [-1, 40, 14, 14]              80
          Dropout-22           [-1, 40, 14, 14]               0
           Conv2d-23           [-1, 10, 14, 14]             400
        MaxPool2d-24             [-1, 10, 7, 7]               0
           Conv2d-25             [-1, 20, 5, 5]           1,800
             ReLU-26             [-1, 20, 5, 5]               0
        GroupNorm-27             [-1, 20, 5, 5]              40
          Dropout-28             [-1, 20, 5, 5]               0
           Conv2d-29             [-1, 30, 3, 3]           5,400
             ReLU-30             [-1, 30, 3, 3]               0
        GroupNorm-31             [-1, 30, 3, 3]              60
          Dropout-32             [-1, 30, 3, 3]               0
           Conv2d-33             [-1, 40, 1, 1]          10,800
             ReLU-34             [-1, 40, 1, 1]               0
        GroupNorm-35             [-1, 40, 1, 1]              80
          Dropout-36             [-1, 40, 1, 1]               0
        AvgPool2d-37             [-1, 40, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             400
================================================================
Total params: 39,490
Trainable params: 39,490
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.41
Params size (MB): 0.15
Estimated Total Size (MB): 1.57
----------------------------------------------------------------
```

The model takes as paramter which type of normalization to use in the form of a flag. 
- In case of batch normalization, while calling the model, set the `use_batch_norm` flag as `True`.
- In case of group normalization, while calling the model, set the `use_group_norm` flag as `True`.
- In case of layer normalization, while calling the model, set the `use_layer_norm` flag as `True`.

## Training and Test Accuracies
### [Batch Normalization](https://github.com/mkthoma/era_v1/blob/main/Session%208/Notebooks/Session8_Batch_Norm.ipynb)
- Training Accuracy - 74.53%
- Testing Accuracy - 74.71%

![image](https://github.com/mkthoma/era_v1/assets/135134412/5e6769da-d6eb-4b83-b975-9e890597249f)


![bn graph](https://github.com/mkthoma/era_v1/assets/135134412/49bb869b-0f95-414d-959d-23fafeff5812)

### [Group Normalization](https://github.com/mkthoma/era_v1/blob/main/Session%208/Notebooks/Session8_Group_Norm.ipynb)
- Training Accuracy - 72.95%
- Testing Accuracy - 72.93%

![image](https://github.com/mkthoma/era_v1/assets/135134412/a60638c9-6eff-47fb-96a2-b40fa6f21e8e)

![gn graph](https://github.com/mkthoma/era_v1/assets/135134412/2866bc29-917b-4bf4-8656-d1bc5795eb99)

### [Layer Normalization](https://github.com/mkthoma/era_v1/blob/main/Session%208/Notebooks/Session8_Layer_Norm.ipynb)
- Training Accuracy - 72.99%
- Testing Accuracy - 73.42%

![image](https://github.com/mkthoma/era_v1/assets/135134412/2c268345-6138-4a84-aedb-3e674fcf98ed)

![ln graph](https://github.com/mkthoma/era_v1/assets/135134412/d7e16cf8-4e4c-49c0-99d8-d51423b28e8a)

## Misclassified Images
Now we shall look at the misclassified images that are present after applying batch, layer and group normalization.

### Batch Normalization
![bn mis](https://github.com/mkthoma/era_v1/assets/135134412/26a2c170-9891-4f6e-81ee-48c6cfc7a5ff)


### Group Normalization
![gn mis](https://github.com/mkthoma/era_v1/assets/135134412/d93902dd-61d6-431a-8d3a-33f6124e337e)


### Layer Normalization
![ln mis](https://github.com/mkthoma/era_v1/assets/135134412/73c46393-bc71-46fc-940c-f1863b88ab99)


## Conclusion

- Normalization Scope:

    - Batch Normalization: Operates within a mini-batch of samples. The mean and standard deviation are computed over the batch dimension.
    - Layer Normalization: Operates within a single layer of the neural network. The mean and standard deviation are computed over all the channels within a single sample.
    - Group Normalization: Operates within a group of channels. The mean and standard deviation are computed over the channel dimension.

- Computational Overhead:

    - Batch Normalization: Introduces additional computational overhead during training since it requires computing the mean and standard deviation over the batch dimension.
    - Layer Normalization: Introduces computational overhead since it computes the mean and standard deviation over the channel dimension for each sample.
    - Group Normalization: Introduces similar computational overhead but operates over smaller groups, which can reduce memory requirements.

- Robustness to Batch Size:

    - Batch Normalization: Performs well with larger batch sizes, as the statistics are computed over a larger number of samples.
    - Layer Normalization: Works well even with smaller batch sizes since the normalization is performed within each sample.
    - Group Normalization: Also performs well with smaller batch sizes since it operates over smaller groups of channels.

- Generalization:

    - Batch Normalization: Tends to introduce dependencies on the batch statistics, which can limit generalization to individual samples during inference.
    - Layer Normalization: Can handle samples independently, making it more suitable for tasks involving sequential data or when batch statistics are unreliable.
    - Group Normalization: Strikes a balance between the two, providing independence within a group of channels, which can generalize well.

In conclusion, the choice of normalization technique depends on the specific requirements and characteristics of the problem. Batch Normalization is effective for larger batch sizes, while Layer Normalization is suitable for sequential data or when batch statistics are unreliable. Group Normalization offers a compromise by working well with smaller batch sizes and providing some degree of independence within channel groups.

