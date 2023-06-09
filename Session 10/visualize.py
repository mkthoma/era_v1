import matplotlib.pyplot as plt
import numpy as np

# shows images from each class
def show_class_samples(data_loader, classes):
    # Obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # Convert images to numpy for display

    # Create a dictionary to store images per class
    images_per_class = {}

    # Group images by class
    for image, label in zip(images, labels):
        class_name = classes[label]
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image)

    fig = plt.figure(figsize=(20, 10))

    # Display 5 images per class
    for idx, class_name in enumerate(classes):
        if class_name in images_per_class:  # Check if class exists in the dictionary
            images = images_per_class[class_name][:5]
            for i in range(5):
                ax = fig.add_subplot(len(classes), 5, idx * 5 + i + 1, xticks=[], yticks=[])
                # Clip and normalize the image data to [0, 1]
                img = np.clip(np.transpose(images[i], (1, 2, 0)), 0, 1)
                ax.imshow(img)
                if i == 0:
                    ax.set_ylabel(class_name)  # Show class name on the y-axis

    plt.tight_layout()
    plt.show()
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# shows random images from different classes
def show_random_samples(data_loader, classes):
    # obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display

    # normalize images
    images = np.transpose(images, (0, 2, 3, 1))  # convert from (batch_size, channels, height, width) to (batch_size, height, width, channels)
    images = (images - images.min()) / (images.max() - images.min())  # normalize to range [0, 1]

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(20, 8))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(4, 5, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(classes[labels[idx]])

# shows rgb channel of an image
def show_image_rgb(data_loader, classes):
    # Obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # Convert images to numpy for display

    # Original image
    original_img = np.squeeze(images[3])
    image_class = classes[labels[3]]  # Get the class name for the image

    # Normalize original image
    normalized_img = (original_img.transpose(1, 2, 0) + 1) / 2  # Normalize image to [0, 1] range

    # Clip normalized image to valid range [0, 1]
    normalized_img = np.clip(normalized_img, 0, 1)

    # RGB channels
    rgb_img = np.transpose(normalized_img, (0, 1, 2))
    channels = ['Red channel', 'Green channel', 'Blue channel']

    fig = plt.figure(figsize=(40, 12))

    # Display original image
    ax_original = fig.add_subplot(1, 4, 1)
    ax_original.imshow(normalized_img)
    ax_original.set_title(f"Original Image\nClass: {image_class}")  # Add class label to the title

    # Display RGB channels
    for idx in range(rgb_img.shape[2]):
        ax = fig.add_subplot(1, 4, idx + 2)
        img = rgb_img[:, :, idx]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)  # Specify vmin and vmax to ensure correct range
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(
                    str(val),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=8,
                    color='white' if img[x][y] < thresh else 'black'
                )

    plt.show()

# shows misclassified images
def show_misclassified_img(misclassified_images, misclassified_labels, misclassified_predictions, classes):
    print("Misclassified Images:")
    fig = plt.figure(figsize=(20, 4))
    for i in range(len(misclassified_images)):
        ax = fig.add_subplot(2, 5, i + 1)
        image = misclassified_images[i].cpu().numpy().transpose(1, 2, 0)
        label = misclassified_labels[i].cpu().numpy().item()  # Convert to integer
        prediction = misclassified_predictions[i].cpu().numpy().item()  # Convert to integer
        # Normalize image data
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        ax.imshow(image)
        ax.set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# plots train and test accuracy and losses
def show_accuracy_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")