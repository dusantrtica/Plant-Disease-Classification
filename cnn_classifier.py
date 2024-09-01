import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
from skelarn.metrics import accuracy_score
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

img = Image.open(
    "../data/test/pepper_healthy/f0e93256-3b10-43c9-880e-e041de89502c___JR_HL 5846.JPG"
)

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

processed_img = transform(img)

batch_size = 4

trainset = torchvision.datasets.ImageFolder(
    root="./train",
    transform=transform,
)
testset = torchvision.datasets.ImageFolder(root="./test", transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

NUM_CLASSES = len(CLASSES)


class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 11 * 11 * 128)
        self.fc2 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x


model = ImageMulticlassClassificationNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
