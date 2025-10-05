import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from imageutils import remove_background_from
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import cv2
import random
from imageutils import (
    download_sample_data,
    save_generated_images,
    generate_specific_class,
)

# Postaviti nasumični seed za ponovljivost
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Konfiguracija uređaja
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Koristi se uređaj: {device}")


class PlantLeafDataset(Dataset):
    """Prilagođeni skup podataka za slike listova biljaka sa labelima"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Kategorije: 0 = zdrav, 1 = bolest/kvar
        healthy_keywords = ["healthy", "Healthy"]
        disease_keywords = [
            "blight",
            "Blight",
            "spot",
            "Spot",
            "rust",
            "Rust",
            "mold",
            "Mold",
        ]

        if os.path.exists(root_dir):
            for class_name in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    # Odrediti da li je zdrav ili bolestan na osnovu imena klase
                    is_healthy = any(
                        keyword in class_name for keyword in healthy_keywords
                    )
                    is_diseased = any(
                        keyword in class_name for keyword in disease_keywords
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
                            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path)  # .convert("RGB")
            image = remove_background_from(image).convert("RGB")

            # image = image_hsv_mask(image)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Greška pri učitavanju slike {img_path}: {e}")
            # Vratiti nasumičnu sliku ako učitavanje ne uspe
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class ConditionalGenerator(nn.Module):
    """Conditional Generator mreža za kreiranje slika listova biljaka sa class labelima"""

    def __init__(self, nz=100, ngf=64, nc=3, num_classes=2, embed_dim=50):
        super(ConditionalGenerator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Embedding sloj za class labels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Generator mreža - prima noise + embedded label
        self.main = nn.Sequential(
            # Ulaz: (nz + embed_dim) x 1 x 1
            nn.ConvTranspose2d(nz + embed_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Stanje: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Stanje: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Stanje: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Stanje: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Izlaz: nc x 64 x 64
        )

        self.main_from_paper = nn.Sequential(
            nn.Conv2d(num_classes, 128, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 4 * 4 * 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(4 * 4 * 64, num_classes),
        )

    def forward(self, noise, labels):
        # Embed labels
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(
            embedded_labels.size(0), self.embed_dim, 1, 1
        )

        # Concatenate noise and embedded labels
        gen_input = torch.cat([noise, embedded_labels], 1)

        return self.main(gen_input)


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator mreža za klasifikaciju real/fake i healthy/blight"""

    def __init__(self, nc=3, ndf=64, num_classes=2, embed_dim=50):
        super(ConditionalDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Embedding sloj za class labels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Projekcija embedded labela na spatial dimenzije
        self.label_projection = nn.Linear(embed_dim, 64 * 64)

        # Discriminator mreža - prima sliku + projected label
        self.main = nn.Sequential(
            # Ulaz: (nc + 1) x 64 x 64 (slika + label channel)
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Stanje: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Stanje: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Stanje: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Stanje: (ndf*8) x 4 x 4
        )

        # Dve glave: jedna za real/fake, druga za class classification
        self.adversarial_head = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

        self.classification_head = nn.Sequential(
            nn.Conv2d(ndf * 8, num_classes, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Softmax(dim=1),
        )

    def forward(self, images, labels):
        batch_size = images.size(0)

        # Embed i project labels
        embedded_labels = self.label_embedding(labels)
        projected_labels = self.label_projection(embedded_labels)
        projected_labels = projected_labels.view(batch_size, 1, 64, 64)

        # Concatenate images and projected labels
        disc_input = torch.cat([images, projected_labels], 1)

        # Promeniti kroz glavnu mrežu
        features = self.main(disc_input)

        # Dve glave za različite zadatke
        adversarial_output = self.adversarial_head(features).view(-1, 1).squeeze(1)
        class_output = self.classification_head(features)

        return adversarial_output, class_output


def train_conditional_gan(dataloader, num_epochs=50, lr=0.0002, beta1=0.5, nz=100):
    """Obučiti Conditional GAN"""
    num_classes = 2  # zdrav, bolestan

    # Inicijalizovati mreže
    netG = ConditionalGenerator(nz=nz, num_classes=num_classes).to(device)
    netD = ConditionalDiscriminator(num_classes=num_classes).to(device)

    # Loss funkcije
    adversarial_criterion = nn.BCELoss()
    classification_criterion = nn.CrossEntropyLoss()

    # Optimizatori
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Labeli za obučavanje
    real_label = 1.0
    fake_label = 0.0

    # Fiksni šum i labeli za vizualizaciju
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    # Kreirati balansiran set labela za vizualizaciju
    fixed_labels = torch.cat(
        [torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]
    ).to(device)

    # Liste za praćenje napretka
    G_losses = []
    D_losses = []

    print("Počinje petlja obučavanja...")

    for epoch in range(num_epochs):
        for i, (data, class_labels) in enumerate(dataloader):
            ############################
            # (1) Ažurirati D mrežu
            ###########################
            ## Obučavati sa pravim batch-om
            netD.zero_grad()
            real_images = data.to(device)
            real_class_labels = class_labels.to(device)
            b_size = real_images.size(0)

            # Real/fake labels
            real_adv_labels = torch.full(
                (b_size,), real_label, dtype=torch.float, device=device
            )

            # Forward pass kroz discriminator
            real_adv_output, real_class_output = netD(real_images, real_class_labels)

            # Gubici za prave slike
            errD_real_adv = adversarial_criterion(real_adv_output, real_adv_labels)
            errD_real_class = classification_criterion(
                real_class_output, real_class_labels
            )
            errD_real = errD_real_adv + errD_real_class
            errD_real.backward()
            D_x = real_adv_output.mean().item()

            ## Obučavati sa lažnim batch-om
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generisati nasumične class labels za fake slike
            fake_class_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_images = netG(noise, fake_class_labels)

            fake_adv_labels = torch.full(
                (b_size,), fake_label, dtype=torch.float, device=device
            )
            fake_adv_output, fake_class_output = netD(
                fake_images.detach(), fake_class_labels
            )

            # Gubici za lažne slike
            errD_fake_adv = adversarial_criterion(fake_adv_output, fake_adv_labels)
            errD_fake_class = classification_criterion(
                fake_class_output, fake_class_labels
            )
            errD_fake = errD_fake_adv + errD_fake_class
            errD_fake.backward()
            D_G_z1 = fake_adv_output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Ažurirati G mrežu
            ###########################
            netG.zero_grad()

            # Za generator, želimo da discriminator misli da su fake slike prave
            real_adv_labels_for_gen = torch.full(
                (b_size,), real_label, dtype=torch.float, device=device
            )

            fake_adv_output, fake_class_output = netD(fake_images, fake_class_labels)

            # Generator loss - želi da fool-uje discriminator i da generiše pravilne klase
            errG_adv = adversarial_criterion(fake_adv_output, real_adv_labels_for_gen)
            errG_class = classification_criterion(fake_class_output, fake_class_labels)
            errG = errG_adv + errG_class
            errG.backward()
            D_G_z2 = fake_adv_output.mean().item()
            optimizerG.step()

            # Ispisati statistike obučavanja
            if i % 50 == 0:
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )

            # Sačuvati gubitke za crtanje kasnije
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Generisati i sačuvati uzorke slika svakih 10 epoha
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels)
                save_generated_images(fake, epoch, fixed_labels)

    return netG, netD, G_losses, D_losses


def plot_losses(G_losses, D_losses):
    """Nacrtati gubitke tokom obučavanja"""
    plt.figure(figsize=(10, 5))
    plt.title("Gubici Conditional Generatora i Diskriminatora tokom Obučavanja")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iteracije")
    plt.ylabel("Gubitak")
    plt.legend()
    plt.savefig("conditional_training_losses.png")
    plt.show()


def main():
    """Glavna funkcija obučavanja"""
    # Hiperparametri
    batch_size = 32
    image_size = 64
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5
    nz = 100  # Veličina latentnog vektora

    # Transformacije podataka
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Preuzeti/pripremiti podatke
    print("Priprema skupa podataka...")
    data_dir = download_sample_data()

    # Kreirati skup podataka i dataloader
    dataset = PlantLeafDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Skup podataka učitan sa {len(dataset)} slika")

    if len(dataset) == 0:
        print(
            "Nisu pronađene slike u skupu podataka. Molimo dodajte slike listova biljaka u direktorijum sa podacima."
        )
        return

    # Obučiti Conditional GAN
    print("Obučavanje Conditional GAN-a...")
    generator, discriminator, G_losses, D_losses = train_conditional_gan(
        dataloader, num_epochs=num_epochs, lr=lr, beta1=beta1, nz=nz
    )

    # Nacrtati gubitke obučavanja
    plot_losses(G_losses, D_losses)

    # Generisati uzorke određenih klasa
    print("Generisanje uzoraka određenih klasa...")
    generate_specific_class(generator, class_label=0, num_samples=8, nz=nz)  # Zdrav
    generate_specific_class(generator, class_label=1, num_samples=8, nz=nz)  # Bolestan

    # Sačuvati modele
    torch.save(generator.state_dict(), "conditional_generator.pth")
    torch.save(discriminator.state_dict(), "conditional_discriminator.pth")
    print(
        "Modeli su sačuvani kao 'conditional_generator.pth' i 'conditional_discriminator.pth'"
    )

    print("Conditional GAN obučavanje završeno!")


if __name__ == "__main__":
    main()
