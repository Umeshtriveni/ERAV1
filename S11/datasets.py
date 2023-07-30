#!/usr/bin/env python3
"""
Module containing wrapper classes for PyTorch Datasets
Author: Shilpaj Bhalerao
Date: Jun 25, 2023
"""
# Standard Library Imports
from typing import Tuple

# Third-Party Imports
from torchvision import datasets, transforms


class AlbumDataset(datasets.CIFAR10):
    """
    Wrapper class to use albumentations library with PyTorch Dataset
    """
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform: list = None):
        """
        Constructor
        :param root: Directory at which data is stored
        :param train: Param to distinguish if data is training or test
        :param download: Param to download the dataset from source
        :param transform: List of transformation to be performed on the dataset
        """
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple:
        """
        Method to return image and its label
        :param index: Index of image and label in the dataset
        """
        image, label = self.data[index], self.targets[index]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
