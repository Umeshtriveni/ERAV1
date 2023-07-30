#!/usr/bin/env python3
"""
Function used for visualization of data and results
Author: Shilpaj Bhalerao
Date: Jul 23, 2023
"""
# Third-Party Imports
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Train Phase transformations
train_set_transforms = {
    'randomcrop': A.RandomCrop(height=32, width=32, p=0.2),
    'horizontalflip': A.HorizontalFlip(),
    'cutout': A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, fill_value=[0.49139968*255, 0.48215827*255 ,0.44653124*255], mask_fill_value=None),
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2(),
}

# Test Phase transformations
test_set_transforms = {
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2()
}


class AddGaussianNoise(object):
    """
    Class for custom augmentation strategy
    """
    def __init__(self, mean=0., std=1.):
        """
        Constructor
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """
        Augmentation strategy to be implemented when called
        """
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        """
        Method to print more infor about the strategy
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

# Usage details
# transforms = transforms.Compose([
#     transforms.ToTensor(),
#     AddGaussianNoise(0., 1.0),
#     ])
