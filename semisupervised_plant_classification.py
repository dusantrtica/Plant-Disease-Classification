import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Postaviti nasumične seedove za ponovljivost
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Konfiguracija uređaja
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Koristi se uređaj: {device}")

# Transformacije podataka
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class PlantLeafDataset(Dataset):
    """Prilagođeni skup podataka za slike listova biljaka sa labelima"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Kategorije: 0 = zdrav, 1 = bolestan
        healthy_keywords = ["healthy"]
        disease_keywords = ["blight", "spot", "rust", "mold", "virus", "spot"]

        if os.path.exists(root_dir):
            for class_name in os.listdir(root_dir):
                plant_sort = class_name.split("_")[0].lower()
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    # Odrediti da li je zdrav ili bolestan na osnovu imena klase
                    is_healthy = any(
                        keyword in class_name.lower() for keyword in healthy_keywords
                    )
                    is_diseased = any(
                        keyword in class_name.lower() for keyword in disease_keywords
                    )

                    if is_healthy:
                        label = 0  # zdrav
                    elif is_diseased:
                        label = 1  # bolestan
                    else:
                        continue  # preskočiti nepoznate kategorije

                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.images.append(os.path.join(class_path, img_name))
                            plant_label_with_sort = f"{plant_sort}_{label}"
                            self.labels.append(plant_label_with_sort)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Greška pri učitavanju slike {img_path}: {e}")
            # Vratiti nasumičnu sliku ako učitavanje ne uspe
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class Generator(nn.Module):
    """
    Semi-Supervised GAN Generator koji prati arhitekturu iz rada
    Baziran na poboljšanoj arhitekturi spomenutoj u radu
    """
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        
        # Prati poboljšanu arhitekturu generatora iz rada
        self.main = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 32*32*128),
            nn.BatchNorm1d(32*32*128),
            nn.ReLU(True),
        )
        
        self.unflatten = nn.Unflatten(1, (128, 32, 32))
        
        # Konvolucioni transpose slojevi
        self.conv_layers = nn.Sequential(
            # Stanje: 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # Stanje: 64 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # Stanje: 32 x 128 x 128 -> potrebno smanjiti na 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(16, nc, kernel_size=2, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Tanh(),
            nn.Flatten() # Proveri da li radi
            # Izlaz: nc x 64 x 64
        )
    
    def forward(self, input):
        x = self.main(input.view(input.size(0), -1))
        x = self.unflatten(x)
        x = self.conv_layers(x)
        return x


def train_and_save_model():
    train_loader = DataLoader(PlantLeafDataset("data/train", transform_train))
    noise_dim = 100
    batch_size = 32
    
    generator = Generator().to(device)
    for i, (real_data, real_labels) in enumerate(train_loader):
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_data = generator(noise)
        

def test_model():
    pass


if __name__ == "__main__":
    train_and_save_model()
    test_model()
