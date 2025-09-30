# Conditional GAN za Klasifikaciju Listova Biljaka

Napredna implementacija Conditional Generative Adversarial Network (cGAN) u PyTorch-u koja može da:
1. **Generiše slike** određene klase (zdrav ili bolestan list)
2. **Klasifikuje slike** kao real/fake I healthy/blight
3. **Kontrolisano generiše** uzorke sa željenim karakteristikama

## Ključne Karakteristike

### **Conditional Generator**
- Prima noise vektor + class label (zdrav/bolestan)
- Koristi embedding sloj za class labels
- Generiše slike određene klase na zahtev
- DCGAN arhitektura sa label conditioning

### **Multi-task Discriminator**
- **Adversarial Head**: Klasifikuje real vs fake slike
- **Classification Head**: Klasifikuje healthy vs blight
- Dual-loss obučavanje za oba zadatka
- Koristi embedded labels za bolje performanse

### **Napredne Funkcionalnosti**
- Generisanje specifičnih klasa na zahtev
- Testiranje tačnosti klasifikacije
- Interpolacija između klasa
- Poređenje različitih klasa
- Balansiran dataset handling

## Upotreba

### 1. Obučavanje Conditional GAN-a

```bash
uv run conditional_gan.py
```

Ovo će:
- Kreirati sintetičke podatke ako stvarni nisu dostupni
- Obučiti conditional generator i discriminator
- Generisati uzorke tokom obučavanja sa labelima
- Sačuvati obučene modele

### 2. Testiranje i Generisanje

```bash
uv run test_conditional_gan.py
```

Ovo će:
- Učitati obučene modele
- Testirati tačnost klasifikacije
- Generisati poređenja između klasa
- Kreirati interpolacije
- Sačuvati raznolike uzorke

### 3. Generisanje Specifičnih Klasa

```python
from conditional_gan import ConditionalGenerator, generate_specific_class
import torch

# Učitati generator
generator = ConditionalGenerator(num_classes=2)
generator.load_state_dict(torch.load('conditional_generator.pth'))

# Generisati zdrave listove (klasa 0)
generate_specific_class(generator, class_label=0, num_samples=8)

# Generisati bolesne listove (klasa 1)
generate_specific_class(generator, class_label=1, num_samples=8)
```

## Arhitektura

### Conditional Generator

```
Ulaz: Noise (100D) + Embedded Label (50D)
├── ConvTranspose2d + BatchNorm + ReLU
├── ConvTranspose2d + BatchNorm + ReLU  
├── ConvTranspose2d + BatchNorm + ReLU
├── ConvTranspose2d + BatchNorm + ReLU
└── ConvTranspose2d + Tanh
Izlaz: RGB slika (3x64x64)
```

### Conditional Discriminator

```
Ulaz: RGB slika (3x64x64) + Projected Label (1x64x64)
├── Conv2d + LeakyReLU
├── Conv2d + BatchNorm + LeakyReLU
├── Conv2d + BatchNorm + LeakyReLU
├── Conv2d + BatchNorm + LeakyReLU
├─┬─ Adversarial Head: Conv2d + Sigmoid → Real/Fake
└─┴─ Classification Head: Conv2d + Softmax → Healthy/Blight
```

## Generisani Fajlovi

### Tokom Obučavanja
- `conditional_generated_images/`: Uzorci sa labelima tokom epoha
- `conditional_training_losses.png`: Grafik gubitaka

### Nakon Obučavanja
- `conditional_generator.pth`: Obučeni generator model
- `conditional_discriminator.pth`: Obučeni discriminator model

### Testiranje
- `class_comparison_grid.png`: Poređenje zdravih vs bolesnih
- `diverse_healthy_samples.png`: Raznovrsni zdravi uzorci
- `diverse_diseased_samples.png`: Raznovrsni bolesni uzorci
- `final_healthy_samples.png`: Finalni zdravi uzorci
- `final_diseased_samples.png`: Finalni bolesni uzorci

## Napredne Funkcije

### 1. Kontrolisano Generisanje

```python
# Generisati samo zdrave listove
healthy_images = generate_class_samples(generator, device, class_label=0, num_samples=16)

# Generisati samo bolesne listove  
diseased_images = generate_class_samples(generator, device, class_label=1, num_samples=16)
```

### 2. Testiranje Klasifikacije

```python
# Testirati koliko dobro discriminator klasifikuje
results = test_discriminator_classification(discriminator, generator, device)
```

### 3. Interpolacija Između Klasa

```python
# Kreirati smooth prelaz između zdravih i bolesnih
interpolate_between_classes(generator, device, num_steps=10)
```

## Hiperparametri

```python
batch_size = 32          # Veličina batch-a
image_size = 64          # Dimenzije slike
num_epochs = 50          # Broj epoha
lr = 0.0002             # Brzina učenja
beta1 = 0.5             # Adam beta1
nz = 100                # Veličina noise vektora
embed_dim = 50          # Dimenzija label embedding-a
num_classes = 2         # Broj klasa (zdrav, bolestan)
```

## Optimizacija Performansi

### 1. Balansiranje Gubitaka
```python
# Generator loss kombinuje adversarial i classification
errG = errG_adv + errG_class

# Discriminator loss kombinuje oba zadatka
errD = errD_real + errD_fake
```

### 2. Label Embedding
- Koristi se embedding sloj umesto one-hot encoding
- Omogućava bolje generalizovanje
- Smanjuje dimenzionalnost

### 3. Multi-head Discriminator
- Odvojene glave za različite zadatke
- Bolje performanse nego single-task
- Stabilnije obučavanje

## Rešavanje Problema

### Problem: Loša klasifikacija
**Rešenje**: Povećajte težinu classification loss-a
```python
errG_class = 2.0 * classification_criterion(fake_class_output, fake_class_labels)
```

### Problem: Mode collapse na jednu klasu
**Rešenje**: Balansirajte dataset i koristite različite noise
```python
# Osigurajte balansiran broj uzoraka po klasi
fake_class_labels = torch.randint(0, num_classes, (b_size,), device=device)
```

### Problem: Generator ne prati labele
**Rešenje**: Povećajte embedding dimenziju
```python
ConditionalGenerator(embed_dim=100)  # Umesto 50
```

## Primeri Upotrebe

### Medicinska Dijagnostika
```python
# Generisati zdrave uzorke za augmentaciju
healthy_samples = generate_specific_class(generator, 0, 100)

# Generisati bolesne uzorke za retke bolesti
diseased_samples = generate_specific_class(generator, 1, 100)
```

## Licenca

Ovaj projekat je za obrazovne i istraživačke svrhe. Poštujte licence korišćenih dataset-a.
