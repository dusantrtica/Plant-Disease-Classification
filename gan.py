import os, math, random
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
import os, shutil, random
from pathlib import Path


# -------- Generator (noise -> 64x64x3) --------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32 * 32 * 128),
            nn.BatchNorm1d(32 * 32 * 128),
            nn.ReLU(True),
            View((-1, 128, 32, 32)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),  # keep 64x64
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# Util class to reshape tensor
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# -------- Discriminator / Classifier (N+1 outputs) --------
# Discriminator outputs in our case 15 classes for real images +
# one fake - as discriminator has to tell if the output is fake or not
class DiscriminatorC(nn.Module):
    def __init__(self, num_classes_plus_fake: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 2),
            nn.LeakyReLU(0.01, True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.LeakyReLU(0.01, True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.LeakyReLU(0.01, True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4 * 4 * 64),
            nn.LeakyReLU(0.01, True),
            nn.Linear(4 * 4 * 64, num_classes_plus_fake),  # N + 1
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# -------- Losses --------
# Loss for discriminator training
def d_loss_two_softmax(logits_real, y_real, logits_fake, fake_index):
    ce = nn.CrossEntropyLoss()
    loss_real = ce(logits_real, y_real)  # CE to true classes
    fake_targets = torch.full(
        (logits_fake.size(0),), fake_index, dtype=torch.long, device=logits_fake.device
    )
    loss_fake = ce(logits_fake, fake_targets)  # CE to FAKE
    return loss_real + loss_fake


# Loss for generator 
def g_loss_avoid_fake(logits_fake, fake_index, eps=1e-6):
    # minimize -log(1 - p_fake) == push probability mass away from FAKE
    p_fake = F.softmax(logits_fake, dim=1)[:, fake_index]
    return torch.mean(-torch.log(1.0 - p_fake + eps))


# -------- Transforms (64x64 like in the paper) --------
def make_transforms(use_segmentation: bool):
    return transforms.Compose([
        # SegmentationTransform(use_segmentation),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),        
    ])


def show_generator_samples(gen, z_fixed, epoch, nrow=8):
    gen.eval()
    with torch.no_grad():
        fake_imgs = gen(z_fixed).cpu()
    fake_imgs = (fake_imgs * 0.5 + 0.5).clamp(0,1)  # [-1,1] → [0,1]
    grid = make_grid(fake_imgs, nrow=nrow)
    plt.figure(figsize=(12,6))
    plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
    plt.axis("off")
    plt.title(f"Generator samples at epoch {epoch}")
    plt.show()

def train_sgan(data_root="data", use_segmentation=False, z_dim=100, batch_size=64, lr=2e-4, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = make_transforms(use_segmentation)
    train_ds = datasets.ImageFolder(os.path.join(data_root,"train"), transform=tfm)
    val_ds   = datasets.ImageFolder(os.path.join(data_root,"val"), transform=make_transforms(False))
    test_ds  = datasets.ImageFolder(os.path.join(data_root,"test"), transform=make_transforms(False))

    N = len(train_ds.classes)
    fake_index = N
    disc = DiscriminatorC(N+1).to(device)
    gen  = Generator(z_dim=z_dim).to(device)

    d_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    g_opt = torch.optim.Adam(gen.parameters(),  lr=lr)

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader   = DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=2)
    test_loader  = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=2)

    acc = MulticlassAccuracy(num_classes=N).to(device)

    # fixed noise for monitoring G’s progress
    z_fixed = torch.randn(32, z_dim, device=device)

    for ep in range(epochs):
        # train mode for models
        disc.train(); gen.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            # --- Train discriminator ---
            z = torch.randn(x.size(0), z_dim, device=device)
            with torch.no_grad():
                x_fake = gen(z)

            logits_real, logits_fake = disc(x), disc(x_fake)
            loss_d = d_loss_two_softmax(logits_real,y,logits_fake,fake_index)
            d_opt.zero_grad()
            loss_d.backward()
            d_opt.step()

            # --- Train generator ---
            z = torch.randn(x.size(0), z_dim, device=device)
            logits_fake_for_g = disc(gen(z))
            loss_g = g_loss_avoid_fake(logits_fake_for_g,fake_index)

            g_opt.zero_grad()
            loss_g.backward()
            g_opt.step()

        # --- Validation ---
        disc.eval(); acc.reset()
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                preds = disc(x)[:,:N].argmax(1)
                acc.update(preds,y)
        print(f"Epoch {ep}/{epochs} | Val acc: {acc.compute().item():.4f}")

        # --- Show generator samples ---
        show_generator_samples(gen, z_fixed, ep)

    # --- Final Test ---
    disc.eval(); acc.reset()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            preds = disc(x)[:,:N].argmax(1)
            acc.update(preds,y)
    print(f"SGAN Test acc: {acc.compute().item():.4f}")



if __name__ == "__main__":
    train_sgan(
        data_root="data",
        use_segmentation=False,
        z_dim=100,
        batch_size=64,
        lr=2e-4,
        epochs=30
    )
