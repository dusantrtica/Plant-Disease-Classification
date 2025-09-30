import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Uvoziti GAN generator za augmentaciju podataka
from conditional_gan import ConditionalGenerator, PlantLeafDataset

# Postaviti nasumični seed za ponovljivost
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Konfiguracija uređaja
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Koristi se uređaj: {device}")

class PlantLeafCNN(nn.Module):
    """Konvoluciona Neuronska Mreža za klasifikaciju listova biljaka"""
    def __init__(self, num_classes=2):
        super(PlantLeafCNN, self).__init__()
        
        # Konvolucioni slojevi za ekstrakciju karakteristika
        self.features = nn.Sequential(
            # Prvi konvolucioni blok
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Dropout2d(0.25),
            
            # Drugi konvolucioni blok
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Dropout2d(0.25),
            
            # Treći konvolucioni blok
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Dropout2d(0.25),
            
            # Četvrti konvolucioni blok
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            nn.Dropout2d(0.25),
        )
        
        # Globalno prosečno poolovanje
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Klasifikator
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Ekstrakcija karakteristika
        x = self.features(x)
        
        # Globalno prosečno poolovanje
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Klasifikacija
        x = self.classifier(x)
        
        return x

class GANAugmentedDataset(Dataset):
    """Dataset koji kombinuje originalne slike sa GAN-generisanim slikama"""
    def __init__(self, original_dataset, generator_path=None, num_generated_per_class=100, nz=100, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform
        self.generated_images = []
        self.generated_labels = []
        
        # Učitati GAN generator ako je putanja data
        if generator_path and os.path.exists(generator_path):
            print("Učitavanje GAN generatora za augmentaciju podataka...")
            self.generator = ConditionalGenerator(nz=nz).to(device)
            self.generator.load_state_dict(torch.load(generator_path, map_location=device))
            self.generator.eval()
            
            # Generisati dodatne slike za svaku klasu
            self._generate_augmented_data(num_generated_per_class, nz)
        else:
            print("GAN generator nije pronađen, koriste se samo originalni podaci.")
    
    def _generate_augmented_data(self, num_per_class, nz):
        """Generisati dodatne slike koristeći GAN"""
        print(f"Generisanje {num_per_class} slika po klasi...")
        
        with torch.no_grad():
            for class_label in [0, 1]:  # 0 = zdrav, 1 = bolestan
                # Generisati slike za trenutnu klasu
                noise = torch.randn(num_per_class, nz, 1, 1, device=device)
                labels = torch.full((num_per_class,), class_label, dtype=torch.long, device=device)
                
                fake_images = self.generator(noise, labels)
                
                # Konvertovati u CPU i dodati u dataset
                for i in range(num_per_class):
                    img_tensor = fake_images[i].cpu()
                    # Denormalizovati iz [-1,1] u [0,1]
                    img_tensor = (img_tensor + 1) / 2.0
                    self.generated_images.append(img_tensor)
                    self.generated_labels.append(class_label)
        
        print(f"Generisano ukupno {len(self.generated_images)} slika.")
    
    def __len__(self):
        return len(self.original_dataset) + len(self.generated_images)
    
    def __getitem__(self, idx):
        # Prvo vratiti originalne slike
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        
        # Zatim GAN-generisane slike
        gen_idx = idx - len(self.original_dataset)
        image = self.generated_images[gen_idx]
        label = self.generated_labels[gen_idx]
        
        # Primeniti transformacije ako su date
        if self.transform:
            # Konvertovati tensor u PIL sliku za transformacije
            image_pil = transforms.ToPILImage()(image)
            image = self.transform(image_pil)
        
        return image, label

def train_cnn(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """Obučiti CNN klasifikator"""
    
    # Loss funkcija i optimizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Liste za praćenje napretka
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("Počinje obučavanje CNN klasifikatora...")
    
    for epoch in range(num_epochs):
        # Obučavanje
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Nulirati gradijente
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistike
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if i % 10 == 0:
                print(f'Epoha [{epoch+1}/{num_epochs}], Korak [{i+1}/{len(train_loader)}], '
                      f'Gubitak: {loss.item():.4f}')
        
        # Validacija
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Izračunati prosečne vrednosti
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        
        # Sačuvati za grafike
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoha [{epoch+1}/{num_epochs}]:')
        print(f'  Obučavanje - Gubitak: {epoch_train_loss:.4f}, Tačnost: {epoch_train_acc:.2f}%')
        print(f'  Validacija - Gubitak: {epoch_val_loss:.4f}, Tačnost: {epoch_val_acc:.2f}%')
        print('-' * 60)
        
        # Ažurirati learning rate
        scheduler.step()
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader, class_names=['Zdrav', 'Bolestan']):
    """Evaluirati model na test skupu"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Izračunati metriku
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\\nFinalna tačnost na test skupu: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detaljni izveštaj
    print("\\nDetaljni izveštaj klasifikacije:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plotovati confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrica Konfuzije')
    plt.ylabel('Stvarna Klasa')
    plt.xlabel('Predviđena Klasa')
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, all_predictions, all_labels

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plotovati istoriju obučavanja"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gubici
    ax1.plot(train_losses, label='Obučavanje', color='blue')
    ax1.plot(val_losses, label='Validacija', color='red')
    ax1.set_title('Gubici tokom Obučavanja')
    ax1.set_xlabel('Epoha')
    ax1.set_ylabel('Gubitak')
    ax1.legend()
    ax1.grid(True)
    
    # Tačnosti
    ax2.plot(train_accuracies, label='Obučavanje', color='blue')
    ax2.plot(val_accuracies, label='Validacija', color='red')
    ax2.set_title('Tačnost tokom Obučavanja')
    ax2.set_xlabel('Epoha')
    ax2.set_ylabel('Tačnost (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, test_loader, num_samples=16):
    """Vizualizovati predviđanja modela"""
    model.eval()
    class_names = ['Zdrav', 'Bolestan']
    
    # Uzeti prvi batch iz test loader-a
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Plotovati rezultate
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(min(num_samples, len(images))):
        ax = axes[i // 4, i % 4]
        
        # Denormalizovati sliku za prikaz
        img = images[i].cpu()
        img = img * 0.5 + 0.5  # Denormalizovati iz [-1,1] u [0,1]
        img = torch.clamp(img, 0, 1)
        
        ax.imshow(img.permute(1, 2, 0))
        
        # Dodati labele
        true_class = class_names[labels[i].item()]
        pred_class = class_names[predicted[i].item()]
        confidence = probabilities[i][predicted[i]].item()
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        ax.set_title(f'Stvarno: {true_class}\\nPredviđeno: {pred_class}\\nPouzdanost: {confidence:.2f}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_with_without_gan(original_data_dir, generator_path=None):
    """Porediti performanse sa i bez GAN augmentacije"""
    
    # Transformacije
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Učitati originalni dataset
    original_dataset = PlantLeafDataset(original_data_dir, transform=transform_train)
    
    # Podeliti na train/val/test
    total_size = len(original_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        original_dataset, [train_size, val_size, test_size]
    )
    
    # Test bez GAN augmentacije
    print("\\n" + "="*60)
    print("OBUČAVANJE BEZ GAN AUGMENTACIJE")
    print("="*60)
    
    train_loader_orig = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_orig = PlantLeafCNN(num_classes=2).to(device)
    train_losses_orig, train_acc_orig, val_losses_orig, val_acc_orig = train_cnn(
        model_orig, train_loader_orig, val_loader, num_epochs=30, lr=0.001
    )
    
    accuracy_orig, _, _ = evaluate_model(model_orig, test_loader)
    
    # Test sa GAN augmentacijom
    if generator_path and os.path.exists(generator_path):
        print("\\n" + "="*60)
        print("OBUČAVANJE SA GAN AUGMENTACIJOM")
        print("="*60)
        
        # Kreirati augmentovani dataset
        augmented_dataset = GANAugmentedDataset(
            train_dataset, 
            generator_path=generator_path, 
            num_generated_per_class=50,
            transform=transform_train
        )
        
        train_loader_aug = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
        
        model_aug = PlantLeafCNN(num_classes=2).to(device)
        train_losses_aug, train_acc_aug, val_losses_aug, val_acc_aug = train_cnn(
            model_aug, train_loader_aug, val_loader, num_epochs=30, lr=0.001
        )
        
        accuracy_aug, _, _ = evaluate_model(model_aug, test_loader)
        
        # Poređenje rezultata
        print("\\n" + "="*60)
        print("POREĐENJE REZULTATA")
        print("="*60)
        print(f"Tačnost bez GAN augmentacije: {accuracy_orig:.4f} ({accuracy_orig*100:.2f}%)")
        print(f"Tačnost sa GAN augmentacijom:  {accuracy_aug:.4f} ({accuracy_aug*100:.2f}%)")
        improvement = (accuracy_aug - accuracy_orig) * 100
        print(f"Poboljšanje: {improvement:+.2f} procentnih poena")
        
        # Sačuvati oba modela
        torch.save(model_orig.state_dict(), 'cnn_without_gan.pth')
        torch.save(model_aug.state_dict(), 'cnn_with_gan.pth')
        
        return model_aug, accuracy_aug
    else:
        torch.save(model_orig.state_dict(), 'cnn_without_gan.pth')
        return model_orig, accuracy_orig

def main():
    """Glavna funkcija"""
    print("CNN Klasifikator za Listove Biljaka")
    print("===================================")
    
    # Putanje
    data_dir = "data/plant_leaves"
    generator_path = "simple_conditional_generator.pth"
    
    # Proveriti da li postoje podaci
    if not os.path.exists(data_dir):
        print(f"Direktorijum sa podacima {data_dir} ne postoji!")
        print("Molimo prvo pokrenite GAN obučavanje da kreirate podatke.")
        return
    
    # Porediti performanse sa i bez GAN augmentacije
    best_model, best_accuracy = compare_with_without_gan(data_dir, generator_path)
    
    print(f"\\nNajbolji model postigao je tačnost od {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("Modeli su sačuvani kao 'cnn_without_gan.pth' i 'cnn_with_gan.pth'")

if __name__ == "__main__":
    main()
