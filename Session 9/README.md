# Applying Dilated and Depth-wise Seperable Convolutions on CIFAR 10 Dataset using Pytorch 
In this exercise we will be looking to implement dilated convolutions and depth wise seperable convolutions on the CIFAR 10 dataset using PyTorch. In addition we will be using the imgae augmentations from the Albumentations library instead of using the Torch transforms library.

## Objective
We will try to build a CNN that can achieve atleast 85% accuracy on the CIFAR10 dataset. We will be making use of dilated convolutions and depth wise seperable convolutions. All the max pooling layers will be replaced by dilated convolutions. The total parameter count needs to be less than 200k and there are no restrictions on the number of epochs it is run. The total receptive field needs to be above 44.

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

![s9 random](https://github.com/mkthoma/era_v1/assets/135134412/3358bf21-4437-4c98-9235-bc62b529fa24)

Looking at different samples of each class

![s9 class](https://github.com/mkthoma/era_v1/assets/135134412/53394ed8-aeb7-486a-9cb0-c871c7b23b0f)

The colour and RGB representation of the same image can be seen below 

![s8 rgb](https://github.com/mkthoma/era_v1/assets/135134412/fb4fd8ce-ac5d-45cc-9851-2de762de536f)

## [Albumentations](https://albumentations.ai/)

Albumentations is a computer vision tool that boosts the performance of deep convolutional neural networks. The library is widely used in industry, deep learning research, machine learning competitions, and open source projects.

The Albumentations library is an open-source Python library used for image augmentation in machine learning and computer vision tasks. Image augmentation is a technique that involves applying a variety of transformations to images to create new training samples, thus expanding the diversity of the training dataset. This process helps improve the generalization and robustness of machine learning models by exposing them to a wider range of image variations.

Albumentations provides a wide range of image augmentation techniques, including geometric transformations (such as scaling, rotation, and cropping), color manipulations (such as brightness adjustment, contrast enhancement, and saturation changes), noise injection, and more. It is designed to be fast, flexible, and easy to use.

One of the key features of Albumentations is its integration with popular deep learning frameworks such as PyTorch and TensorFlow. It provides a simple and efficient API that allows users to easily incorporate image augmentation into their data preprocessing pipeline, seamlessly integrating it with their training process.

Albumentations supports a diverse set of image data types, including numpy arrays, PIL images, OpenCV images, and others. It also allows for custom augmentation pipelines and provides a rich set of options to control the augmentation parameters and their probabilities.

Overall, Albumentations is a powerful library that simplifies the process of applying image augmentation techniques, helping researchers and practitioners improve the performance and reliability of their computer vision models.

### Transforms used

- `Normalization`: This transformation normalizes the image by subtracting the mean values (0.4914, 0.4822, 0.4465) and dividing by the standard deviation values (0.247, 0.243, 0.261) for each color channel (RGB). Normalization helps to standardize the pixel values and ensure that they have a similar range.

- `HorizontalFlip`: This transformation horizontally flips the image with a probability of 50%. It creates a mirror image of the original image, which helps invariance to left-right orientation.

- `ShiftScaleRotate`: This transformation randomly applies affine transformations such as shifting, scaling, and rotation to the image. The parameters for these transformations are randomly generated. Shifting shifts the image horizontally and vertically, scaling rescales the image, and rotation rotates the image by a certain angle within a specified range. These transformations simulate changes in the position and orientation of objects in the image.

- `CoarseDropout`: This transformation randomly selects a rectangular region in the image and replaces it with zeros or a specific fill value. The max_holes parameter specifies the maximum number of regions to drop, and max_height and max_width define the maximum dimensions of the dropped regions. The min_holes, min_height, and min_width parameters set the minimum values for these dimensions. The fill_value is obtained from the _get_dataset_mean method, which calculates the mean value of the dataset and uses it as the fill value. Coarse dropout can help introduce robustness to missing regions or occlusions in the images.

- `ToTensorV2`: This transformation converts the image from a numpy array to a PyTorch tensor. It also adjusts the dimensions and channel ordering to match PyTorch's convention (C x H x W).

We can see the train data loader having images with transformations from the albumentations library applied below - 

![train transformations sessio 9](https://github.com/mkthoma/era_v1/assets/135134412/3c9a7c25-0024-4a9f-8f3a-c2176e62047e)


These transformations collectively create a diverse set of augmented images for the training data, allowing the model to learn from different variations and improve its generalization capability. 

## Dilated Convolutions 
Dilated convolution, also known as atrous convolution, is a type of convolutional operation used in convolutional neural networks (CNNs). It extends the traditional concept of convolution by introducing a dilation factor, which controls the spacing between the values in the kernel (filter).

In a standard convolution, each element of the kernel is placed on top of the corresponding input pixel and the element-wise multiplication and summation are performed to generate the output value. The kernel slides over the input with a fixed stride, typically 1, and covers all neighboring pixels.

In dilated convolution, the kernel is dilated by inserting gaps (or zeros) between the elements. The dilation factor determines the spacing between these gaps. By introducing these gaps, the effective receptive field of the convolution operation is expanded, which allows the network to capture information from a larger context.

  ![0_3cTXIemm0k3Sbask](https://github.com/mkthoma/era_v1/assets/135134412/9cb4acd5-ca03-43cc-b2c7-7b6f55f05686)


Dilated convolution can be particularly useful in scenarios where capturing fine-grained details or a wider context is important. It helps overcome the limitations of standard convolutions, which might overlook important features when the receptive field is limited.

One advantage of dilated convolutions is that they increase the field of view without significantly increasing the number of parameters or computational cost. By adjusting the dilation factor, the network can control the receptive field size and capture information at different scales.

Dilated convolutions have been widely used in various computer vision tasks, such as semantic segmentation, image synthesis, and object detection. They have shown effectiveness in capturing multi-scale features and improving the model's ability to handle objects of different sizes.

## Depth-wise Separabe Convolution

Depth-wise separable convolution is a combination of depth-wise convolution and point-wise convolution, and it is commonly used in convolutional neural networks (CNNs) to reduce computational complexity and improve efficiency.

Depth-wise separable convolution decomposes the standard convolution operation into two separate steps:

- Depth-wise Convolution: In this step, each input channel is convolved separately with its own set of filters, similar to depth-wise convolution. However, unlike depth-wise convolution, where the output of each depth-wise convolutional operation is combined, the outputs of the depth-wise convolutions are kept separate.

- Point-wise Convolution: Point-wise convolution, also known as 1x1 convolution, is applied to the outputs of the depth-wise convolutions. Here, a 1x1 kernel (filter) is used to perform a traditional convolution across all channels. This step helps in creating new features by combining the information from different channels.

![95621Depthwise-Separable-Convolution](https://github.com/mkthoma/era_v1/assets/135134412/1067587b-4a51-4dc9-a1ac-9f8ec3011050)


By using depth-wise separable convolution, the number of parameters and computations is significantly reduced compared to standard convolutions. This is because depth-wise convolution involves fewer parameters (as each channel has its own set of filters), and point-wise convolution operates on a reduced number of channels.

The benefits of depth-wise separable convolutions include:

- Reduced computational complexity: By separating the spatial and channel-wise computations, depth-wise separable convolution reduces the number of operations required.
- Lower memory footprint: The reduced number of parameters in depth-wise separable convolutions leads to a lower memory requirement.
- Improved efficiency: Depth-wise separable convolutions are computationally efficient, making them suitable for resource-constrained environments such as mobile devices.
- Overall, depth-wise separable convolution is an effective technique for achieving efficient and lightweight CNN architectures while still preserving the ability to capture meaningful features from the input data.

## Model Architecture

The model consists of 4 convolutional blocks each of which has 3 convolutional layers in it. Each block has a 1x1 transition layer in between. The output block consists of a GAP layer followed by a fully connected layer. No max pooling is used in the model.

```python
# CONVOLUTION BLOCK 1
self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, dilation=1, bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(dropout_value)) 
# output_size = 32, RF=3

self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),padding=1, dilation=1, bias=False),nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(dropout_value)) 
# output_size = 32, RF=5

self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, dilation=2, bias=False),nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(dropout_value)) 
# output_size = 28, RF=9

# Transition block
self.transblock1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)) 
# output_size = 28, RF=9


# CONVOLUTION BLOCK 2
self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(dropout_value)) 
# output_size = 28, RF=11

self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 28, RF=13

self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, dilation=4, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 20, RF=21

# Transition Block
self.transblock2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)) 
# output_size = 20, RF=21


# CONVOLUTION BLOCK 3
self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(dropout_value)) 
# output_size = 20, RF=23

self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 20, RF=25

self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, dilation=8, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 4, RF=41

# Transition Block
self.transblock3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)) 
# output_size = 4, RF=41

# CONVOLUTION BLOCK 4

# Depth wise seperable convolution
self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, groups=16, bias=False),nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), bias=False),nn.ReLU(),nn.BatchNorm2d(32),nn.Dropout(dropout_value)) 
# output_size = 4, RF=43

self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 4, RF=45

self.convblock12 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), padding=0, dilation=1, bias=False),
nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(dropout_value)) 
# output_size = 2, RF=47


# OUTPUT BLOCK

# Applying Global Average Pooling
self.gap = nn.Sequential(
    nn.AvgPool2d(kernel_size=2)
)

# fully connected layer
self.convblock13 = nn.Sequential(
    nn.Linear(in_features=64, out_features=10, bias=False)
)
```

In each convolutional block we can see that the thrid layer is a dilated convolution in place of a max pooling layer. The dilation rates are changed for each block to achieve the desired receptive field at the end.

The calculation for receptive field in case of a dilated convolution remains same with one minor change. The kernel size needs to be re-adjusted based on the formula α(k−1)+1, where α is the dilation rate and k is the kernel size. For example a dilation rate of 2 in a 3x3 would result in a effective kernel size of 2*(3-1) + 1 = 5

The calculations for Receptive field and output channel size are shown below:

![image](https://github.com/mkthoma/era_v1/assets/135134412/98bc16cf-25f8-450d-b212-bbf4d70e21d5)


The final model can be visualized as: 

![s9 arch](https://github.com/mkthoma/era_v1/assets/135134412/af98fae4-d3b0-4eeb-b4ba-2f650c643793)

The model summary shows us that we are only using about 173k parameters for this model.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 28, 28]           9,216
             ReLU-10           [-1, 32, 28, 28]               0
      BatchNorm2d-11           [-1, 32, 28, 28]              64
          Dropout-12           [-1, 32, 28, 28]               0
           Conv2d-13           [-1, 16, 28, 28]             512
           Conv2d-14           [-1, 32, 28, 28]           4,608
             ReLU-15           [-1, 32, 28, 28]               0
      BatchNorm2d-16           [-1, 32, 28, 28]              64
          Dropout-17           [-1, 32, 28, 28]               0
           Conv2d-18           [-1, 64, 28, 28]          18,432
             ReLU-19           [-1, 64, 28, 28]               0
      BatchNorm2d-20           [-1, 64, 28, 28]             128
          Dropout-21           [-1, 64, 28, 28]               0
           Conv2d-22           [-1, 64, 20, 20]          36,864
             ReLU-23           [-1, 64, 20, 20]               0
      BatchNorm2d-24           [-1, 64, 20, 20]             128
          Dropout-25           [-1, 64, 20, 20]               0
           Conv2d-26           [-1, 16, 20, 20]           1,024
           Conv2d-27           [-1, 32, 20, 20]           4,608
             ReLU-28           [-1, 32, 20, 20]               0
      BatchNorm2d-29           [-1, 32, 20, 20]              64
          Dropout-30           [-1, 32, 20, 20]               0
           Conv2d-31           [-1, 64, 20, 20]          18,432
             ReLU-32           [-1, 64, 20, 20]               0
      BatchNorm2d-33           [-1, 64, 20, 20]             128
          Dropout-34           [-1, 64, 20, 20]               0
           Conv2d-35             [-1, 64, 4, 4]          36,864
             ReLU-36             [-1, 64, 4, 4]               0
      BatchNorm2d-37             [-1, 64, 4, 4]             128
          Dropout-38             [-1, 64, 4, 4]               0
           Conv2d-39             [-1, 16, 4, 4]           1,024
           Conv2d-40             [-1, 16, 4, 4]             144
           Conv2d-41             [-1, 32, 4, 4]             512
             ReLU-42             [-1, 32, 4, 4]               0
      BatchNorm2d-43             [-1, 32, 4, 4]              64
          Dropout-44             [-1, 32, 4, 4]               0
           Conv2d-45             [-1, 64, 4, 4]          18,432
             ReLU-46             [-1, 64, 4, 4]               0
      BatchNorm2d-47             [-1, 64, 4, 4]             128
          Dropout-48             [-1, 64, 4, 4]               0
           Conv2d-49             [-1, 64, 3, 3]          16,384
             ReLU-50             [-1, 64, 3, 3]               0
      BatchNorm2d-51             [-1, 64, 3, 3]             128
          Dropout-52             [-1, 64, 3, 3]               0
        AvgPool2d-53             [-1, 64, 1, 1]               0
           Linear-54                   [-1, 10]             640
================================================================
Total params: 173,856
Trainable params: 173,856
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.76
Params size (MB): 0.66
Estimated Total Size (MB): 7.44
----------------------------------------------------------------
```

## Training and Test Accuracies
From the plots below we can see that we have achieved accuracy greater than 85% towards from about $82^{nd}$ epoch.

![image](https://github.com/mkthoma/era_v1/assets/135134412/7129d7d7-7d80-43ea-b018-6762db32f178)

![s9 train test](https://github.com/mkthoma/era_v1/assets/135134412/5b01fc40-69f8-42d8-958d-dd2f45f2cba9)


## Misclassified Images
Now we shall look at the misclassified images that are present after applying the model on the dataset.

![s9 mis](https://github.com/mkthoma/era_v1/assets/135134412/8e0de44f-15cb-475d-b11f-7f25aa692df5)


## Conclusion

Dilated convolution and depth-wise convolution are two specific types of convolutional operations used in convolutional neural networks (CNNs) that offer unique benefits and applications.

### Dilated Convolution (Atrous Convolution):

- Dilated convolution introduces a dilation factor that controls the spacing between the elements of the convolution kernel.
- By inserting gaps (zeros) between the kernel elements, dilated convolution expands the effective receptive field, capturing information from a larger context.
- It is useful for capturing fine-grained details and incorporating multi-scale information.
- Dilated convolutions can efficiently increase the receptive field without significantly increasing the number of parameters or computational cost.
- They have been widely used in tasks such as semantic segmentation, image synthesis, and object detection.

### Depth-wise Convolution:

- Depth-wise convolution decomposes the standard convolution into two separate operations: depth-wise convolution and point-wise convolution.
- In depth-wise convolution, each input channel is convolved separately with its own set of filters.
- This process reduces computational complexity and allows the network to learn spatial features more efficiently.
- Point-wise convolution (1x1 convolution) is then applied to the outputs of depth-wise convolutions to combine the information from different channels.
- Depth-wise separable convolution, which combines depth-wise and point-wise convolutions, further reduces parameters and computations, making it suitable for lightweight models.
- Depth-wise convolutions are commonly used in mobile and embedded applications where computational resources are limited.

In summary, dilated convolution extends the receptive field of convolutions and is effective for capturing fine details and multi-scale information. On the other hand, depth-wise convolution decomposes the convolution operation into spatial and channel-wise components, reducing computational complexity and allowing efficient processing of spatial features. These convolution techniques offer advantages in different contexts and are utilized in various computer vision tasks to enhance the performance and efficiency of CNNs.
