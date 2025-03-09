import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

class chestxraydataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for loading chest X-ray images categorized into
    'normal', 'covid', and 'viral' classes.

    This dataset:
    - Loads `.png` images from specified directories.
    - Randomly samples from one of the three classes for each item.
    - Applies transformations to images before returning them.
    - Returns image tensors and their corresponding class indices.

    Args:
        image_dirs (dict): A dictionary mapping class names ('normal', 'covid', 'viral')
                           to their corresponding image directory paths.
        transform (callable): A torchvision transform or custom transform function to
                              apply to each image.

    Attributes:
        class_names (list): List of class labels ['normal', 'covid', 'viral'].
        images (dict): Dictionary mapping each class to a list of image filenames.
        image_dirs (dict): Dictionary mapping each class to its corresponding directory.
        transform (callable): Transformations to be applied to each image.

    Methods:
        __len__(): Returns the total number of images across all classes.
        __getitem__(index): Returns a transformed image and its corresponding class label index.
    """

    def __init__(self,image_dirs,transform):
        """
        Initializes the dataset by loading image file names for each class.

        Args:
            image_dirs (dict): Dictionary with keys as class names and values as paths to directories.
            transform (callable): Transformation function to be applied to each image.
        """
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)}{class_name} examples')
            return images
        self.images = {}
        self.class_names = ['normal','covid','viral']
        for c in self.class_names:
            self.images[c] = get_images(c)
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of images across all classes.

        Returns:
            int: Total number of images.
        """
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self,index):
        """
        Retrieves a transformed image and its corresponding class index.

        Args:
            index (int): Index of the data item.

        Returns:
            tuple: A tuple (image_tensor, class_index), where:
                - image_tensor (Tensor): Transformed image.
                - class_index (int): Index corresponding to the class label.
        """
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name],image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image),self.class_names.index(class_name)

def show_images(images,labels,preds):
    """
    Displays a set of images along with their true and predicted class labels.

    Args:
        images (list or Tensor): A list or batch of image tensors (typically shape [C, H, W]).
        labels (Tensor): The ground truth class labels for the images.
        preds (Tensor): The predicted class labels for the images.

    The function:
    - Denormalizes the images using ImageNet mean and standard deviation.
    - Plots each image using matplotlib.
    - Shows the true label (as the x-axis label) and the predicted label (as the y-axis label).
    - Highlights correct predictions in green and incorrect ones in red.

    """
    plt.figure(figsize=(8,4))
    for i,image in enumerate(images):
        plt.subplot(1,6,i+1,xticks=[],yticks=[])
        image = image.numpy().transpose((1,2,0))
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        image = image * std + mean
        image = np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i] == labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}',color=col)
    plt.tight_layout()
    plt.show()
                
def show_preds():
    """
    Displays a batch of test images along with their true and predicted class labels.

    This function:
    - Sets the `resnet18` model to evaluation mode.
    - Loads a batch of images and labels from the test dataloader (`dl_test`).
    - Performs a forward pass through the model to generate predictions.
    - Uses `show_images()` to display the images with their corresponding labels.

    This will visualize one batch of predictions with correct and incorrect predictions highlighted.
    """
    resnet18.eval()
    images,labels = next(iter(dl_test))
    outputs = resnet18(images)
    _, preds = torch.max(outputs,1)
    show_images(images,labels,preds)

def train(epochs):
    """
    Trains the `resnet18` model for a specified number of epochs and evaluates its performance periodically.

    Args:
        epochs (int): The number of training epochs to run.

    The function:
    - Trains the model using the training dataset (`dl_train`) for the given number of epochs.
    - Calculates and prints training loss.
    - Evaluates the model on the test dataset (`dl_test`) every 20 training steps.
    - Computes validation loss and accuracy during evaluation.
    - Uses `show_preds()` to visualize predictions after each evaluation.
    - Stops training early if validation accuracy exceeds 95%.
    This function provides intermediate training and validation progress, helping monitor model performance in real-time.
    """
    print("Starting training..")
    for e in range(0,epochs):
        print("="*20)
        print(f'Starting epoc {e + 1}/{epochs}')
        print("="*20)
        train_loss = 0
        resnet18.train()
        for train_step,(images,labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if train_step % 20 == 0:
                print("Evaluating at step",train_step)
                acc=0.
                val_loss=0.
                resnet18.eval()
                for val_step,(images,labels) in enumerate(dl_test):
                    outputs = resnet18(images)
                    loss = loss_fn(outputs,labels)
                    val_loss += loss.item()
                    _,preds = torch.max(outputs,1)
                    acc += sum((preds ==labels).numpy())
                val_loss /= (val_step + 1)
                acc = acc / len(test_dataset)
                print(f'Val loss :{val_loss:.4f}, Acc: {acc:.4f}')
                show_preds()
                resnet18.train()
                if acc> 0.95:
                    print("Performance condition satisfied")
                    return
        train_loss /= (train_step + 1)
        print(f'Training loss: {train_loss:.4f}')