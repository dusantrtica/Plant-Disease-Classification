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

class PlantLeafDataset(Dataset):
    """Prilagođeni skup podataka za slike listova biljaka sa labelima"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Kategorije: 0 = zdrav, 1 = bolestan
        healthy_keywords = ['healthy', 'Healthy']
        disease_keywords = ['blight', 'Blight', 'spot', 'Spot', 'rust', 'Rust', 'mold', 'Mold']
        
        if os.path.exists(root_dir):
            for class_name in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    # Odrediti da li je zdrav ili bolestan na osnovu imena klase
                    is_healthy = any(keyword in class_name for keyword in healthy_keywords)
                    is_diseased = any(keyword in class_name for keyword in disease_keywords)
                    
                    if is_healthy:
                        label = 0  # zdrav
                    elif is_diseased:
                        label = 1  # bolestan
                    else:
                        continue  # preskočiti nepoznate kategorije
                    
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Greška pri učitavanju slike {img_path}: {e}")
            # Vratiti nasumičnu sliku ako učitavanje ne uspe
            return self.__getitem__(random.randint(0, len(self.images)-1))

class SGANGenerator(nn.Module):
    """
    Semi-Supervised GAN Generator koji prati arhitekturu iz rada
    Baziran na poboljšanoj arhitekturi spomenutoj u radu
    """
    def __init__(self, nz=100, ngf=64, nc=3):
        super(SGANGenerator, self).__init__()
        
        # Prati poboljšanu arhitekturu generatora iz rada
        self.main = nn.Sequential(
            # Ulaz: nz x 1 x 1
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            # Preoblikovati u prostorne dimenzije
            nn.Linear(1024, 32*32*128),
            nn.BatchNorm1d(32*32*128),
            nn.ReLU(True),
        )
        
        # Unflatten u prostorne dimenzije
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
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16, nc, kernel_size=2, stride=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Tanh()
            # Izlaz: nc x 64 x 64
        )
    
    def forward(self, input):
        x = self.main(input.view(input.size(0), -1))
        x = self.unflatten(x)
        x = self.conv_layers(x)
        return x

class SGANDiscriminatorClassifier(nn.Module):
    """
    Semi-Supervised GAN Diskriminator/Klasifikator (D/C) koji prati rad
    Ova mreža obavlja oba zadatka:
    1. Real/Fake diskriminaciju (adversarial zadatak)
    2. Klasifikaciju bolesti (supervised zadatak)
    
    Prati arhitekturu iz rada sa NUM_CLASSES + 1 izlaza
    gde dodatna klasa predstavlja "lažne" uzorke
    """
    def __init__(self, nc=3, ndf=64, num_classes=2):
        super(SGANDiscriminatorClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Prati poboljšanu arhitekturu diskriminatora iz rada
        self.features = nn.Sequential(
            # Ulaz: nc x 64 x 64
            nn.Conv2d(nc, 128, kernel_size=5, stride=1),
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
        # Prati rad: "diskriminator može da deluje kao klasifikator"
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4, 4*4*64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(4*4*64, num_classes + 1)  # +1 za lažnu klasu
        )
    
    def forward(self, input):
        features = self.features(input)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits

def discriminator_loss_for_mcgan(logits_real, logits_fake, true_labels, softmax_loss):
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
    loss = None
    N, C = logits_real.size()
    
    # Kreirati lažne labele tačno kao u radu: Variable((torch.zeros(N) + 23).type(dtype).long())
    # U radu, 23 je bio indeks za lažnu klasu (imali su 57 pravih klasa + 1 lažna = 58 ukupno)
    # U našem slučaju, koristimo C-1 kao indeks lažne klase (poslednja klasa)
    fake_labels = (torch.zeros(N, device=logits_real.device) + (C - 1)).long()
    
    # Vratiti zbir dve softmax funkcije kako je specificirano u radu
    return softmax_loss(logits_real, true_labels) + softmax_loss(logits_fake, fake_labels)

def generator_loss_for_sgan(logits_fake, num_classes):
    """
    Funkcija gubitka generatora za SGAN
    Generator želi da diskriminator klasifikuje lažne slike kao prave klase (ne kao lažnu klasu)
    
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
    loss = F.kl_div(F.log_softmax(real_class_logits, dim=1), target_probs, reduction='batchmean')
    
    return loss

def weights_init(m):
    """Inicijalizovati težine mreže"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def train_sgan(dataloader, num_epochs=50, lr=0.0001, nz=100, num_classes=2):
    """
    Obučiti Semi-Supervised GAN prateći Algoritam 1 iz rada
    
    Algoritam 1 SGAN Algoritam Obučavanja:
    Ulaz: I : broj ukupnih iteracija
    for i=1 to i==I do
        Izvući m uzoraka šuma
        Izvući m primera iz pravih podataka
        Izvršiti gradijentni spust da se minimizuje D/C gubitak
        funkcija u odnosu na prave i lažne labele
        Izvući m uzoraka šuma
        Izvršiti gradijentni spust na parametrima G
        u odnosu na NLL D/C izlaza na minibatch-u veličine m
    """
    
    # Inicijalizovati mreže
    netG = SGANGenerator(nz=nz).to(device)
    netDC = SGANDiscriminatorClassifier(num_classes=num_classes).to(device)
    
    # Primeniti inicijalizaciju težina
    netG.apply(weights_init)
    netDC.apply(weights_init)
    
    # Optimizatori - koristi learning rate iz rada (1e-5 spomenuto za ResNet baseline)
    optimizerDC = optim.Adam(netDC.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Fiksni šum za vizualizaciju
    fixed_noise = torch.randn(16, nz, device=device)
    
    # Liste za praćenje napretka
    G_losses = []
    DC_losses = []
    
    print("Počinje SGAN petlja obučavanja...")
    print(f"Obučavanje za {num_epochs} epoha sa learning rate {lr}")
    
    for epoch in range(num_epochs):
        for i, (real_data, real_labels) in enumerate(dataloader):
            batch_size = real_data.size(0)
            
            ############################
            # (1) Ažurirati D/C mrežu: minimizovati D/C funkciju gubitka
            ###########################
            netDC.zero_grad()
            
            # Pravi podaci
            real_data = real_data.to(device)
            real_labels = real_labels.to(device)
            
            # Forward pass kroz D/C sa pravim podacima
            logits_real = netDC(real_data)
            
            # Generisati lažne podatke
            noise = torch.randn(batch_size, nz, device=device)
            fake_data = netG(noise)
            
            # Forward pass kroz D/C sa lažnim podacima
            logits_fake = netDC(fake_data.detach())
            
            # Izračunati D/C gubitak koristeći tačnu funkciju iz rada
            softmax_loss = nn.CrossEntropyLoss()
            dc_loss = discriminator_loss_for_mcgan(logits_real, logits_fake, real_labels, softmax_loss)
            dc_loss.backward()
            optimizerDC.step()
            
            ############################
            # (2) Ažurirati G mrežu: maksimizovati log(D/C(G(z))) za prave klase
            ###########################
            netG.zero_grad()
            
            # Generisati nove lažne podatke za ažuriranje generatora
            noise = torch.randn(batch_size, nz, device=device)
            fake_data = netG(noise)
            
            # Forward pass kroz D/C sa novim lažnim podacima
            logits_fake_for_g = netDC(fake_data)
            
            # Izračunati gubitak generatora
            g_loss = generator_loss_for_sgan(logits_fake_for_g, num_classes)
            g_loss.backward()
            optimizerG.step()
            
            # Ispisati statistike obučavanja
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_DC: {dc_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
            
            # Sačuvati gubitke za crtanje
            G_losses.append(g_loss.item())
            DC_losses.append(dc_loss.item())
        
        # Generisati i sačuvati uzorke slika svakih 10 epoha
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                fake = netG(fixed_noise)
                save_generated_images(fake, epoch, 'sgan')
    
    return netG, netDC, G_losses, DC_losses

def save_generated_images(fake_images, epoch, prefix='sgan'):
    """Sačuvati generisane slike"""
    os.makedirs(f'{prefix}_generated_images', exist_ok=True)
    vutils.save_image(fake_images.detach(),
                      f'{prefix}_generated_images/samples_epoch_{epoch:03d}.png',
                      normalize=True, nrow=4)

def evaluate_sgan_classifier(netDC, test_loader, num_classes=2):
    """
    Evaluirati SGAN diskriminator/klasifikator na pravim test podacima
    Evaluirati samo prave klase (isključiti lažnu klasu)
    """
    netDC.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = netDC(images)
            
            # Razmotriti samo prave klase (prvih num_classes izlaza)
            real_class_logits = logits[:, :num_classes]
            _, predicted = torch.max(real_class_logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Izračunati metrike
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\nTačnost SGAN Klasifikatora na test skupu: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detaljan izveštaj klasifikacije
    class_names = ['Zdrav', 'Bolestan']
    print("\nDetaljan Izveštaj Klasifikacije:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Matrica konfuzije
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plotovati matricu konfuzije
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('SGAN Klasifikator - Matrica Konfuzije')
    plt.ylabel('Prava Klasa')
    plt.xlabel('Predviđena Klasa')
    plt.savefig('sgan_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, all_predictions, all_labels

def plot_training_losses(G_losses, DC_losses):
    """Plotovati gubitke obučavanja"""
    plt.figure(figsize=(10, 5))
    plt.title("SGAN Generator i Diskriminator/Klasifikator Gubici Tokom Obučavanja")
    plt.plot(G_losses, label="Generator")
    plt.plot(DC_losses, label="Diskriminator/Klasifikator")
    plt.xlabel("Iteracije")
    plt.ylabel("Gubitak")
    plt.legend()
    plt.savefig('sgan_training_losses.png')
    plt.show()

def generate_samples_sgan(generator, num_samples=16, nz=100):
    """Generisati nove uzorke koristeći obučeni SGAN generator"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, nz, device=device)
        fake_images = generator(noise)
        
        # Sačuvati generisane uzorke
        vutils.save_image(fake_images.detach(),
                          'sgan_generated_samples.png',
                          normalize=True, nrow=4)
        
        print(f"Generisano {num_samples} uzoraka sačuvano kao 'sgan_generated_samples.png'")
        return fake_images

def compare_sgan_with_baseline(data_dir, num_epochs=30):
    """
    Porediti performanse SGAN-a sa baseline CNN
    Prateći eksperimentalni setup iz rada
    """
    
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
    
    # Učitati skup podataka
    full_dataset = PlantLeafDataset(data_dir, transform=transform_train)
    
    # Podeliti skup podataka
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Kreirati data loader-e
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Podela skupa podataka: {train_size} train, {val_size} val, {test_size} test")
    
    # Obučiti SGAN
    print("\n" + "="*60)
    print("OBUČAVANJE SEMI-SUPERVISED GAN (SGAN)")
    print("="*60)
    
    generator, discriminator_classifier, G_losses, DC_losses = train_sgan(
        train_loader, num_epochs=num_epochs, lr=0.0001, nz=100, num_classes=2
    )
    
    # Plotovati gubitke obučavanja
    plot_training_losses(G_losses, DC_losses)
    
    # Evaluirati SGAN klasifikator
    sgan_accuracy, _, _ = evaluate_sgan_classifier(discriminator_classifier, test_loader)
    
    # Generisati uzorke
    print("\nGenerisanje uzoraka sa obučenim SGAN...")
    generate_samples_sgan(generator, num_samples=16, nz=100)
    
    # Sačuvati modele
    torch.save(generator.state_dict(), 'sgan_generator.pth')
    torch.save(discriminator_classifier.state_dict(), 'sgan_discriminator_classifier.pth')
    
    print(f"\nSGAN je postigao tačnost od {sgan_accuracy:.4f} ({sgan_accuracy*100:.2f}%)")
    print("Modeli sačuvani kao 'sgan_generator.pth' i 'sgan_discriminator_classifier.pth'")
    
    return generator, discriminator_classifier, sgan_accuracy

def main():
    """Glavna funkcija koja prati eksperimentalni setup iz rada"""
    print("Semi-Supervised GAN za Klasifikaciju Bolesti Biljaka")
    print("Prati: Plant Disease Classification Using Convolutional Networks and Generative Adversarial Networks")
    print("="*80)
    
    # Direktorijum podataka
    data_dir = "data/plant_leaves"
    
    # Proveriti da li podaci postoje
    if not os.path.exists(data_dir):
        print(f"Direktorijum podataka {data_dir} ne postoji!")
        print("Molimo prvo pokrenite conditional GAN skript da kreirate uzorke podataka.")
        return
    
    # Pokrenuti SGAN eksperiment
    generator, discriminator_classifier, accuracy = compare_sgan_with_baseline(
        data_dir, num_epochs=50
    )
    
    print(f"\nFinalna SGAN tačnost: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nEksperiment uspešno završen!")
    print("Generisane slike i težine modela su sačuvane.")

if __name__ == "__main__":
    main()
