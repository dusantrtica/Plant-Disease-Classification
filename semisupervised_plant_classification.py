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
            nn.Linear(1024, 32 * 32 * 128),
            nn.BatchNorm1d(32 * 32 * 128),
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
            nn.Conv2d(16, nc, kernel_size=2, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Tanh(),
            # nn.Flatten(),  # Proveri da li radi
            # Izlaz: nc x 64 x 64
        )

    def forward(self, input):
        x = self.main(input.view(input.size(0), -1))
        x = self.unflatten(x)
        x = self.conv_layers(x)
        return x


class DCDiscriminator(nn.Module):
    """
    Semi-Supervised GAN Diskriminator/Klasifikator (D/C) koji prati rad
    Ova mreža obavlja oba zadatka:
    1. Real/Fake diskriminaciju (adversarial zadatak)
    2. Klasifikaciju bolesti (supervised zadatak)

    Prati arhitekturu iz rada sa NUM_CLASSES + 1 izlaza
    gde dodatna klasa predstavlja "lažne" uzorke
    """

    def __init__(self, num_channels=3, num_classes=1):
        super(DCDiscriminator, self).__init__()
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

        # Klasifikator glava - izlaz NUM_CLASSES + 1 (dodatna klasa za lažne)
        # diskriminator može da deluje kao klasifikator
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 4 * 4 * 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(4 * 4 * 64, num_classes + 1),  # +1 za lažnu klasu
        )

    def forward(self, input):
        features = self.features(input)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits


def discriminator_loss(logits_real, logits_fake, true_labels, softmax_loss):
    """
    Funkcija gubitka diskriminatora tačno kako je specificirana u radu

    Iz rada:
    def discriminator_loss_for_mcgan(logits_real, logits_fake, true_labels, softmax_loss):
        Inputs:
        - logits_real: PyTorch Variable oblika (N, C) koji daje skorove za prave podatke.
        - logits_fake: PyTorch Variable oblika (N, C) koji daje skorove za lažne podatke.
        - true_labels: PyTorch Variable oblika (N, ) koji daje labele za prave podatke.
        loss = None
        N, C = logits_real.size()
        fake_labels = Variable((torch.zeros(N) + 23).type(dtype).long())
        return softmax_loss(logits_real, true_labels) + softmax_loss(logits_fake, fake_labels)

    Gubitak diskriminatora sadrži zbir dve softmax funkcije:
    1. Jedna smanjuje negativnu log verovatnoću u odnosu na date labele pravih podataka
    2. Druga smanjuje negativnu log verovatnoću u odnosu na lažne labele lažnih podataka
    """
    N, C = logits_real.size()

    # Kreirati lažne labele tačno kao u radu: Variable((torch.zeros(N) + 23).type(dtype).long())
    # U radu, 23 je bio indeks za lažnu klasu (imali su 57 pravih klasa + 1 lažna = 58 ukupno)
    # U našem slučaju, koristimo C-1 kao indeks lažne klase (poslednja klasa)
    fake_labels = (torch.zeros(N, device=logits_real.device) + (C + 1)).long()

    # Vratiti zbir dve softmax funkcije kako je specificirano u radu
    return softmax_loss(logits_real, true_labels) + softmax_loss(
        logits_fake, fake_labels
    )


def generator_loss(logits_fake, num_classes):
    """
    Funkcija gubitka generatora
    Generator želi da diskriminator klasifikuje lažne slike kao prave klase (ne kao lažnu klasu)
    Cilj generatora je da prevari diskriminatora - tj da za lazne podatke
    diskriminator vrati da su pravi

    Args:
        logits_fake: PyTorch Variable oblika (N, C+1) koji daje skorove za lažne podatke
        num_classes: Broj pravih klasa

    Returns:
        Gubitak generatora
    """
    N = logits_fake.size(0)

    # Generator želi da lažne slike budu klasifikovane kao bilo koja prava klasa (ne lažna)
    # Koristićemo uniformnu distribuciju preko pravih klasa
    # Ovo maksimizuje negativnu log verovatnoću kako je spomenuto u radu

    # Kreirati uniformnu ciljnu distribuciju preko pravih klasa
    target_probs = torch.ones(N, num_classes, device=logits_fake.device) / num_classes

    # Dobiti verovatnoće samo za prave klase (isključiti lažnu klasu)
    real_class_logits = logits_fake[:, :num_classes]
    real_class_probs = F.softmax(real_class_logits, dim=1)

    # KL divergence gubitak da podstiče uniformnu distribuciju preko pravih klasa
    loss = F.kl_div(
        F.log_softmax(real_class_logits, dim=1), target_probs, reduction="batchmean"
    )

    return loss


def train_model(num_epochs=1, lr=0.0001, noise_dim=100):
    """
    Obučiti Semi-Supervised GAN prateći Algoritam  iz rada
    """
    dataset = PlantLeafDataset("data/train", transform=transform_train)
    num_classes = len(dataset.label_values)
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = DCDiscriminator(num_channels=3, num_classes=num_classes).to(device)

    # Optimizatori - koristi learning rate iz rada (1e-5 spomenuto za ResNet baseline)
    optimizerDC = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Liste za praćenje napretka
    G_losses = []
    DC_losses = []

    print("Počinje GAN petlja obučavanja...")
    print(f"Obučavanje za {num_epochs} epoha sa learning rate {lr}")

    for epoch in range(num_epochs):
        for i, (real_data, real_labels) in enumerate(train_loader):
            batch_size = real_data.size(0)

            ############################
            # (1) Ažurirati DC Discriminator mrežu:
            #  minimizovati DC funkciju gubitka
            ###########################
            discriminator.zero_grad()
            # Pravi podaci
            real_data = real_data.to(device)
            real_labels = real_labels.to(device)

            # Forward pass kroz DC sa pravim podacima
            logits_real = discriminator(real_data)

            # Generisati lažne podatke
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)

            # Forward pass kroz DC sa lažnim podacima
            # ali bez da "ucimo" generator
            logits_fake = discriminator(fake_data.detach())

            # Izračunati DC gubitak koristeći tačnu funkciju iz rada
            softmax_loss = nn.CrossEntropyLoss()
            dc_loss = discriminator_loss(
                logits_real, logits_fake, real_labels, softmax_loss
            )
            dc_loss.backward()
            optimizerDC.step()

            ############################
            # (2) Ažurirati Generator mrežu: maksimizovati log(Discriminator(G(z))) za prave klase
            ###########################
            generator.zero_grad()

            # Generisati nove lažne podatke za ažuriranje generatora
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)

            # Forward pass kroz DC sa novim lažnim podacima
            # ali ovog puta ne detachujemo, jer koeficijenti treba da se
            # azuriraju
            logits_fake_for_generator = discriminator(fake_data)

            # Izračunati gubitak generatora
            g_loss = generator_loss(logits_fake_for_generator, num_classes)
            g_loss.backward()
            optimizerG.step()

            # Ispisati statistike obučavanja
            if i % 50 == 0:
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] "
                    f"Loss_DC: {dc_loss.item():.4f} Loss_G: {g_loss.item():.4f}"
                )

            # Sačuvati gubitke za crtanje
            G_losses.append(g_loss.item())
            DC_losses.append(dc_loss.item())

    return generator, discriminator, G_losses, DC_losses


def plot_training_losses(G_losses, DC_losses):
    """Plotovati gubitke obučavanja"""
    plt.figure(figsize=(10, 5))
    plt.title("SGAN Generator i Diskriminator/Klasifikator Gubici Tokom Obučavanja")
    plt.plot(G_losses, label="Generator")
    plt.plot(DC_losses, label="Diskriminator/Klasifikator")
    plt.xlabel("Iteracije")
    plt.ylabel("Gubitak")
    plt.legend()
    plt.show()


def save_model(model, name):
    torch.save(model.state_dict(), f"{name}.pth")


def load_discriminator(name, num_channels, num_classes):
    discriminator = DCDiscriminator(num_channels=num_channels, num_classes=num_classes).to(device)
    discriminator.load_state_dict(torch.load(f"{name}.pth"))
    return discriminator


def evaluate_discriminator(name):
    """
    Evaluirati SGAN diskriminator/klasifikator na pravim test podacima
    Evaluirati samo prave klase (isključiti lažnu klasu)
    """

    test_dataset = PlantLeafDataset("data/test", transform=transform_test)
    num_classes = len(test_dataset.label_values)
    test_loader = DataLoader(test_dataset)
    discriminator = load_discriminator(name, num_channels=3, num_classes=num_classes)

    discriminator.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = discriminator(images)

            # Razmotriti samo prave klase (prvih num_classes izlaza)
            real_class_logits = logits[:, :num_classes]
            _, predicted = torch.max(real_class_logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Izračunati metrike
    accuracy = accuracy_score(all_labels, all_predictions)

    print(
        f"\nTačnost SGAN Klasifikatora na test skupu: {accuracy:.4f} ({accuracy*100:.2f}%)"
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
    plt.savefig("sgan_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    return accuracy, all_predictions, all_labels


def train_and_save_model():
    generator, discriminator, G_losses, DC_losses = train_model(num_epochs=10)
    save_model(discriminator, "dc_discriminator")    
    plot_training_losses(G_losses, DC_losses)


if __name__ == "__main__":
    train_and_save_model()
    evaluate_discriminator("dc_discriminator")
