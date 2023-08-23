import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(os.getcwd(), 'images'), data_transforms['train']),
}

dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    )
}

model = models.efficientnet_b4(weights='IMAGENET1K_V1')

for param in model.parameters():
  param.requires_gred = False


model.classifier = nn.Sequential(
    nn.Linear(1792, 512),
    nn.ReLU(),
    nn.Linear(512, len(image_datasets['train'].classes)),
)

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
epochs = 5

model.train()

for epoch in range(epochs):
    sum_losses = 0
    sum_accs = 0
    for x_batch, y_batch in dataloaders['train']:
        x_batch = x_batch
        y_batch = y_batch
        y_pred = model(x_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_losses = sum_losses + loss.item()
        y_prob = nn.Softmax(1)(y_pred)
        y_pred_index = torch.argmax(y_prob, axis=1)

        acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100

        sum_accs = sum_accs + acc.item()
    avg_loss = sum_losses / len(dataloaders['train'])
    avg_acc = sum_accs / len(dataloaders['train'])

    print(f'train: Epoch {epoch+1:4d}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%')

torch.save(model.state_dict(),'cam.h5')