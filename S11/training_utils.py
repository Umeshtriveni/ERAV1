#!/usr/bin/env python3
"""
Utilities for Model Training
Author: Shilpaj Bhalerao
Date: Jun 21, 2023
"""
# Standard Library Imports

# Third-Party Imports
from tqdm import tqdm
import torch


def get_correct_predictions(prediction, labels):
    """
    Function to return total number of correct predictions
    :param prediction: Model predictions on a given sample of data
    :param labels: Correct labels of a given sample of data
    :return: Number of correct predictions
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()


def train(model, device, train_loader, optimizer, criterion, scheduler=None):
    """
    Function to train model on the training dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param train_loader: DataLoader for training dataset
    :param optimizer: Optimization algorithm to be used for updating weights
    :param criterion: Loss function for training
    :param scheduler: Scheduler for learning rate
    """
    # Enable layers like Dropout for model training
    model.train()

    # Utility to display training progress
    pbar = tqdm(train_loader)

    # Variables to track loss and accuracy during training
    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over each batch and fetch images and labels from the batch
    for batch_idx, (data, target) in enumerate(pbar):

        # Put the images and labels on the selected device
        data, target = data.to(device), target.to(device)

        # Reset the gradients for each batch
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Use learning rate scheduler if defined
        if scheduler:
            scheduler.step()

        # Get total number of correct predictions
        correct += get_correct_predictions(pred, target)
        processed += len(data)

        # Display the training information
        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

    return correct, processed, train_loss


def test(model, device, test_loader, criterion):
    """
    Function to test the model training progress on the test dataset
    :param model: Model architecture
    :param device: Device on which training is to be done (GPU/CPU)
    :param test_loader: DataLoader for test dataset
    :param criterion: Loss function for test dataset
    """
    # Disable layers like Dropout for model inference
    model.eval()

    # Variables to track loss and accuracy
    test_loss = 0
    correct = 0

    # Disable gradient updation
    with torch.no_grad():
        # Iterate over each batch and fetch images and labels from the batch
        for batch_idx, (data, target) in enumerate(test_loader):

            # Put the images and labels on the selected device
            data, target = data.to(device), target.to(device)

            # Pass the images to the output and get the model predictions
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            # Sum up batch correct predictions
            correct += get_correct_predictions(output, target)

    # Calculate test loss for a epoch
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct, test_loss


def get_lr(optimizer):
    """
    Function to track learning rate while model training
    :param optimizer: Optimizer used for training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']