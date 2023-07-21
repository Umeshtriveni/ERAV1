import torch.nn.functional as F
from tqdm import tqdm
from utils import calculate_accuracy
import torch

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
    # Training loop
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        accuracy = calculate_accuracy(output, target)
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={accuracy:.2f}')
        train_acc.append(accuracy)
