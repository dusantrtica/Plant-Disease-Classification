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
transform_train = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class PlantLeafDataset(Dataset):
    """Prilagođeni skup podataka za slike listova biljaka sa labelima"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.label_values = []

        is_train = "train" in root_dir

        # Kategorije: 0 = zdrav, 1 = bolestan
        healthy_keywords = ["healthy"]
        disease_keywords = ["blight", "spot", "rust", "mold", "virus", "spot"]

        if os.path.exists(root_dir):
            for class_name in os.listdir(root_dir):
                plant_sort = class_name.split("_")[0].lower()

                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    total_files_in_dir = len(os.listdir(class_path))
                    print(f"Files in {class_path}: {total_files_in_dir}")
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

                    for i, img_name in enumerate(os.listdir(class_path)):
                        # process only 1% of images for training
                        
                        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.images.append(os.path.join(class_path, img_name))
                            plant_label = f"{plant_sort}_{label}"

                            if plant_label not in self.label_values:
                                self.label_values.append(plant_label)

                            self.labels.append(self.label_values.index(plant_label))

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

class CNN(nn.Module):

    def __init__(self, num_channels=3, num_classes=1):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        # Prati poboljšanu arhitekturu diskriminatora iz rada
        self.features = nn.Sequential(
            # Ulaz: nc x 64 x 64
            nn.Conv2d(num_channels, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stanje: 128 x 30 x 30
            nn.Conv2d(128, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stanje: 64 x 13 x 13
            nn.Conv2d(64, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stanje: 32 x 4 x 4
        )

        # Flatten sloj
        self.flatten = nn.Flatten()
        
        # diskriminator može da deluje kao klasifikator
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 4 * 4 * 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(4 * 4 * 64, num_classes)
        )

    def forward(self, input):
        features = self.features(input)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits


def train_model(num_epochs=1, lr=0.0001, noise_dim=100):
    """
    Obučiti CNN
    """
    dataset = PlantLeafDataset("data/train", transform=transform_train)
    num_classes = len(dataset.label_values)
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    cnn = CNN(num_channels=3, num_classes=num_classes).to(device)

    # Optimizatori - koristi learning rate iz rada (1e-5 spomenuto za ResNet baseline)
    optimizerCnn = optim.Adam(cnn.parameters(), lr=lr, betas=(0.5, 0.999)) 
    criterion = nn.CrossEntropyLoss()

    # Liste za praćenje napretka
    CNN_losses = []

    print("Počinje CNN petlja obučavanja...")
    print(f"Obučavanje za {num_epochs} epoha sa learning rate {lr}")

    for epoch in range(num_epochs):
        for i, (real_data, real_labels) in enumerate(train_loader):
            batch_size = real_data.size(0)
          
            optimizerCnn.zero_grad()
            # Pravi podaci
            real_data = real_data.to(device)
            real_labels = real_labels.to(device)

            # Forward pass kroz DC sa pravim podacima
            logits = cnn(real_data).to(device)

            # Izračunati DC gubitak koristeći tačnu funkciju iz rada            
            cnn_loss = criterion(logits, real_labels)
            cnn_loss.backward()
            optimizerCnn.step()

            # Ispisati statistike obučavanja
            if i % 50 == 0:
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] "
                    f"Loss CNN: {cnn_loss.item():.4f}"
                )

            # Sačuvati gubitke za crtanje
            CNN_losses.append(cnn_loss.item())            

    return cnn, CNN_losses

def save_model(model, name):
    torch.save(model.state_dict(), f"{name}.pth")

def train_and_save_model():
    model, losses = train_model(num_epochs=5)
    save_model(model, "cnn")    
    plot_training_losses(losses)


def plot_training_losses(losses):
    """Plotovati gubitke obučavanja"""
    plt.figure(figsize=(10, 5))
    plt.title("CNN Gubici Tokom Obučavanja")
    plt.plot(losses, label="CNN")
    plt.xlabel("Iteracije")
    plt.ylabel("Gubitak")
    plt.legend()
    plt.savefig('cnn_classification.png')
    plt.show()

def load_model(name, num_channels, num_classes):
    model = CNN(num_channels=num_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f"{name}.pth"))
    return model

def evaluate_model(model_name):
    """
    Evaluirati SGAN diskriminator/klasifikator na pravim test podacima
    Evaluirati samo prave klase (isključiti lažnu klasu)
    """

    test_dataset = PlantLeafDataset("data/test", transform=transform_test)
    num_classes = len(test_dataset.label_values)
    test_loader = DataLoader(test_dataset)
    model = load_model(model_name, num_channels=3, num_classes=num_classes)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            # Razmotriti samo prave klase (prvih num_classes izlaza)
            real_class_logits = logits[:, :num_classes]
            _, predicted = torch.max(real_class_logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Izračunati metrike
    accuracy = accuracy_score(all_labels, all_predictions)

    print(
        f"\nTačnost CNN Klasifikatora na test skupu: {accuracy:.4f} ({accuracy*100:.2f}%)"
    )

    # Detaljan izveštaj klasifikacije
    class_names = test_dataset.label_values
    print("\nDetaljan Izveštaj Klasifikacije:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Matrica konfuzije
    cm = confusion_matrix(all_labels, all_predictions)

    # Plotovati matricu konfuzije
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("SGAN Klasifikator - Matrica Konfuzije")
    plt.ylabel("Prava Klasa")
    plt.xlabel("Predviđena Klasa")
    plt.savefig("cnn_plant_classification.png", dpi=150, bbox_inches="tight")
    plt.show()

    return accuracy, all_predictions, all_labels


if __name__ == '__main__':
    train_and_save_model()
    evaluate_model("cnn")
