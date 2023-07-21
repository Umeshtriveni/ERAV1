import torch
import torch.optim as optim
from model import Net
from data import get_data_loaders
from train import train
from test import test
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
dropout_value = 0.1
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Data loaders configuration
train_loader, test_loader = get_data_loaders(train_batch_size=128, test_batch_size=128)

# Training configuration
EPOCHS = 15
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch, train_losses, train_acc)
    test(model, device, test_loader, test_losses, test_acc)

print('Finished Training')

# dataiter = iter(dataloader)
# data = next(dataiter)

dataiter = iter(test_loader)
images, labels = next(dataiter)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Display images
grid_img = torchvision.utils.make_grid(images)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()


# Print ground truth labels
print('Ground Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Make predictions
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# Print predicted labels
print('Predicted: ', ' '.join('%5s' % classes[predicted[j].item()] for j in range(4)))
