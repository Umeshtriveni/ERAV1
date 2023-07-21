import torch

def calculate_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = predicted.eq(target.data.view_as(predicted)).sum().item()
    accuracy = correct / target.size(0) * 100
    return accuracy
