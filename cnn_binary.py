import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

img = Image.open(
    "../data/test/pepper_healthy/f0e93256-3b10-43c9-880e-e041de89502c___JR_HL 5846.JPG"
)

transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


batch_size = 4

trainset = torchvision.datasets.ImageFolder(
    root="./train",
    transform=transform,
)
testset = torchvision.datasets.ImageFolder(root="./test", transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


class LeavesClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # out: BS, 6, 30, 30
        self.pool = nn.MaxPool2d(2, 2)  # out: BS, 6, 15, 15
        self.conv2 = nn.Conv2d(6, 16, 3)  # out: BS, 16, 13, 13

        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # out: BS
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


model = LeavesClassificationNet()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero grads
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calc loss
        loss = loss_fn(outputs, labels.reshape(-1, 1).float())

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)}, Loss: {loss.item():.4f}"
            )


y_test = []
y_test_pred = []
for (
    i,
    data,
) in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()

    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {acc*100:.2f} %")
