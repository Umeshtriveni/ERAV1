#!/usr/bin/env python3
"""
PyTorch Lightning for ResNet Architecture
Author: Shilpaj Bhalerao
"""
# Standard Library Imports
import os
import math

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import CIFAR10

from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

# Local Imports
from datasets import AlbumDataset
from utils import get_cifar_statistics
from visualize import visualize_cifar_augmentation, display_cifar_data_samples


class Layers:
    """
    Class containing different types of Convolutional layer
    """

    def __init__(self, groups=1):
        """
        Constructor
        """
        self.group = groups

    @staticmethod
    def standard_conv_layer(in_channels: int,
                            out_channels: int,
                            kernel_size: int = 3,
                            padding: int = 0,
                            stride: int = 1,
                            dilation: int = 1,
                            normalization: str = "batch",
                            last_layer: bool = False,
                            conv_type: str = "standard",
                            groups: int = 1):
        """
        Method to return a standard convolution block
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel used in the layer
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        :param dilation: Dilation for Atrous convolution
        :param normalization: Type of normalization technique used
        :param last_layer: Flag to indicate if the layer is last convolutional layer of the network
        :param conv_type: Type of convolutional layer
        :param groups: Number of Groups for Group Normalization
        """
        # Select normalization type
        if normalization == "layer":
            _norm_layer = nn.GroupNorm(1, out_channels)
        elif normalization == "group":
            if not groups:
                raise ValueError("Value of group is not defined")
            _norm_layer = nn.GroupNorm(groups, out_channels)
        else:
            _norm_layer = nn.BatchNorm2d(out_channels)

        # Select the convolution layer type
        if conv_type == "standard":
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                   kernel_size=kernel_size, bias=False, padding=padding)
        elif conv_type == "depthwise":
            conv_layer = Layers.depthwise_conv(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                               padding=padding)
        elif conv_type == "dilated":
            conv_layer = Layers.dilated_conv(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                             padding=padding, dilation=dilation)

        # For last layer only return the convolution output
        if last_layer:
            return nn.Sequential(conv_layer)
        return nn.Sequential(
            conv_layer,
            _norm_layer,
            nn.ReLU(),
            # nn.Dropout(self.dropout_value)
        )

    @staticmethod
    def resnet_block(channels):
        """
        Method to create a RESNET block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    @staticmethod
    def custom_block(input_channels, output_channels):
        """
        Method to create a custom configured block
        :param input_channels: Number of input channels
        :param output_channels: Number of output channels
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, stride=1, kernel_size=3, bias=False,
                      padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    @staticmethod
    def depthwise_conv(in_channels, out_channels, stride=1, padding=0):
        """
        Method to return the depthwise separable convolution layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param padding: Padding used in the layer
        :param stride: Stride used for convolution
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels,
                      kernel_size=3, bias=False, padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1, bias=False,
                      padding=0)
        )

    @staticmethod
    def dilated_conv(in_channels, out_channels, stride=1, padding=0, dilation=1):
        """
        Method to return the dilated convolution layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride used for convolution
        :param padding: Padding used in the layer
        :param dilation: Dilation value for a kernel
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, bias=False,
                      padding=padding, dilation=dilation)
        )


class LITResNet(LightningModule, Layers):
    """
    David's Model Architecture for Session-10 CIFAR10 dataset
    """

    def __init__(self, class_names, data_dir='/data/'):
        """
        Constructor
        """
        # Initialize the Module class
        super().__init__()

        # Initialize variables
        self.classes = class_names
        self.data_dir = data_dir
        self.num_classes = 10
        self._learning_rate = 0.03
        self.inv_normalize = transforms.Normalize(
            mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23],
            std=[1 / 0.23, 1 / 0.23, 1 / 0.23]
        )
        self.batch_size = 512
        self.epochs = 24
        self.accuracy = Accuracy(task='multiclass',
                                 num_classes=10)
        self.train_transforms = transforms.Compose([transforms.ToTensor()])
        self.test_transforms = transforms.Compose([transforms.ToTensor()])
        self.stats_train = None
        self.stats_test = None
        self.cifar10_train = None
        self.cifar10_test = None
        self.cifar10_val = None
        self.misclassified_data = None

        # Defined Layers for the model
        self.prep_layer = None
        self.custom_block1 = None
        self.custom_block2 = None
        self.custom_block3 = None
        self.resnet_block1 = None
        self.resnet_block3 = None
        self.pool4 = None
        self.fc = None
        self.dropout_value = None

        # Initialize all the layers
        self.model_layers()

    # ##################################################################################################
    # ################################ Model Architecture Related Hooks ################################
    # ##################################################################################################
    def model_layers(self):
        """
        Method to initialize layers for the model
        """
        # Prep Layer
        self.prep_layer = Layers.standard_conv_layer(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)

        # Convolutional Block-1
        self.custom_block1 = Layers.custom_block(input_channels=64, output_channels=128)
        self.resnet_block1 = Layers.resnet_block(channels=128)

        # Convolutional Block-2
        self.custom_block2 = Layers.custom_block(input_channels=128, output_channels=256)

        # Convolutional Block-3
        self.custom_block3 = Layers.custom_block(input_channels=256, output_channels=512)
        self.resnet_block3 = Layers.resnet_block(channels=512)

        # MaxPool Layer
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=2)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)

        # Dropout value of 10%
        self.dropout_value = 0.1

    def forward(self, x):
        """
        Forward pass for model training
        :param x: Input layer
        :return: Model Prediction
        """
        # Prep Layer
        x = self.prep_layer(x)

        # Convolutional Block-1
        x = self.custom_block1(x)
        r1 = self.resnet_block1(x)
        x = x + r1

        # Convolutional Block-2
        x = self.custom_block2(x)

        # Convolutional Block-3
        x = self.custom_block3(x)
        r2 = self.resnet_block3(x)
        x = x + r2

        # MaxPool Layer
        x = self.pool4(x)

        # Fully Connected Layer
        x = x.view(-1, 512)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    # ##################################################################################################
    # ############################## Training Configuration Related Hooks ##############################
    # ##################################################################################################

    def configure_optimizers(self):
        """
        Method to configure the optimizer and learning rate scheduler
        """
        learning_rate = 0.03
        weight_decay = 1e-4
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self._learning_rate,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        epochs=self.epochs,
                                                        pct_start=5 / self.epochs,
                                                        div_factor=100,
                                                        three_phase=False,
                                                        final_div_factor=100,
                                                        anneal_strategy="linear"
                                                        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @property
    def learning_rate(self) -> float:
        """
        Method to get the learning rate value
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """
        Method to set the learning rate value
        :param value: Updated value of learning rate
        """
        self._learning_rate = value

    def set_training_confi(self, *, epochs, batch_size):
        """
        Method to set parameters required for model training
        :param epochs: Number of epochs for which model is to be trained
        :param batch_size: Batch Size
        """
        self.epochs = epochs
        self.batch_size = batch_size

    # #################################################################################################
    # ################################## Training Loop Related Hooks ##################################
    # #################################################################################################
    def training_step(self, train_batch, batch_index):
        """
        Method called on training dataset to train the model
        :param train_batch: Batch containing images and labels
        :param batch_index: Index of the batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Method called on validation dataset to check if the model is learning
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Method called on test dataset to check model performance on unseen data
        :param batch: Batch containing images and labels
        :param batch_idx: Index of the batch
        """
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    # ##############################################################################################
    # ##################################### Data Related Hooks #####################################
    # ##############################################################################################

    def set_transforms(self, train_set_transforms: dict, test_set_transforms: dict):
        """
        Method to set the transformations to be done on training and test datasets
        :param train_set_transforms: Dictionary of transformations for training dataset
        :param test_set_transforms: Dictionary of transformations for test dataset
        """
        self.train_transforms = A.Compose(train_set_transforms.values())
        self.test_transforms = A.Compose(test_set_transforms.values())

    def prepare_data(self):
        """
        Method to download the dataset
        """
        self.stats_train = CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        self.stats_test = CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):
        """
        Method to create Split the dataset into train, test and val
        """
        # Only if dataset is not already split, perform the split operation
        if not self.cifar10_train and not self.cifar10_test and not self.cifar10_val:

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                cifar10_full = AlbumDataset(self.data_dir, train=True, download=True, transform=self.train_transforms)
                self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45_000, 5_000])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.cifar10_test = AlbumDataset(self.data_dir, train=False, download=True,
                                                 transform=self.test_transforms)

    def train_dataloader(self):
        """
        Method to return the DataLoader for Training set
        """
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=os.cpu_count())

    def val_dataloader(self):
        """
        Method to return the DataLoader for the Validation set
        """
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        """
        Method to return the DataLoader for the Test set
        """
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=os.cpu_count())

    def get_statistics(self, data_set_type="Train"):
        """
        Method to get the statistics for CIFAR10 dataset
        """
        # Execute self.prepare_data() only if not done earlier
        if not self.stats_train and not self.stats_test:
            self.prepare_data()

        # Print stats for selected dataset
        if data_set_type == "Train":
            get_cifar_statistics(self.stats_train)
        else:
            get_cifar_statistics(self.stats_test, data_set_type="Test")

    def display_data_samples(self, dataset="train", num_of_images=20):
        """
        Method to display data samples
        """
        # Execute self.prepare_data() only if not done earlier
        try:
            assert self.stats_train
        except AttributeError:
            self.prepare_data()

        if dataset == "train":
            display_cifar_data_samples(self.stats_train, num_of_images, self.classes)
        else:
            display_cifar_data_samples(self.stats_test, num_of_images, self.classes)

    @staticmethod
    def visualize_augmentation(aug_set_transforms: dict):
        """
        Method to visualize augmentations
        :param aug_set_transforms: Dictionary of transformations to be visualized
        """
        aug_train = AlbumDataset('./data', train=True, download=True)
        visualize_cifar_augmentation(aug_train, aug_set_transforms)

    # #############################################################################################
    # ############################## Misclassified Data Related Hooks ##############################
    # #############################################################################################

    def get_misclassified_data(self):
        """
        Function to run the model on test set and return misclassified images
        """
        if self.misclassified_data:
            return self.misclassified_data

        self.misclassified_data = []
        self.prepare_data()
        self.setup()

        test_loader = self.test_dataloader()

        # Reset the gradients
        with torch.no_grad():
            # Extract images, labels in a batch
            for data, target in test_loader:

                # Migrate the data to the device
                data, target = data.to(self.device), target.to(self.device)

                # Extract single image, label from the batch
                for image, label in zip(data, target):

                    # Add batch dimension to the image
                    image = image.unsqueeze(0)

                    # Get the model prediction on the image
                    output = self.forward(image)

                    # Convert the output from one-hot encoding to a value
                    pred = output.argmax(dim=1, keepdim=True)

                    # If prediction is incorrect, append the data
                    if pred != label:
                        self.misclassified_data.append((image, label, pred))
        return self.misclassified_data

    def display_cifar_misclassified_data(self, number_of_samples: int = 10):
        """
        Function to plot images with labels
        :param number_of_samples: Number of images to print
        """
        if not self.misclassified_data:
            self.misclassified_data = self.get_misclassified_data()

        fig = plt.figure(figsize=(10, 10))

        x_count = 5
        y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

        for i in range(number_of_samples):
            plt.subplot(y_count, x_count, i + 1)
            img = self.misclassified_data[i][0].squeeze().to('cpu')
            img = self.inv_normalize(img)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.title(
                r"Correct: " + self.classes[self.misclassified_data[i][1].item()] + '\n' + 'Output: ' + self.classes[
                    self.misclassified_data[i][2].item()])
            plt.xticks([])
            plt.yticks([])
