# CNN Klasifikator za Listove Biljaka

Konvoluciona Neuronska Mreža (CNN) za klasifikaciju slika listova biljaka kao zdravih ili bolesnih, sa mogućnošću korišćenja GAN-generisanih slika za poboljšanje performansi.

## Pregled

Ovaj CNN klasifikator može da:
1. **Klasifikuje slike** listova kao zdrave ili bolesne
2. **Koristi originalne podatke** iz dataset-a
3. **Augmentuje podatke** koristeći GAN-generisane slike
4. **Poredi performanse** sa i bez GAN augmentacije
5. **Vizualizuje rezultate** i predviđanja

## Arhitektura CNN-a

### **Struktura Mreže**

```
Ulaz: RGB slika (3x64x64)
├── Konvolucioni Blok 1: 3→32 kanala
│   ├── Conv2d(3,32) + BatchNorm + ReLU
│   ├── Conv2d(32,32) + BatchNorm + ReLU  
│   ├── MaxPool2d(2x2) → 32x32
│   └── Dropout2d(0.25)
├── Konvolucioni Blok 2: 32→64 kanala
│   ├── Conv2d(32,64) + BatchNorm + ReLU
│   ├── Conv2d(64,64) + BatchNorm + ReLU
│   ├── MaxPool2d(2x2) → 16x16
│   └── Dropout2d(0.25)
├── Konvolucioni Blok 3: 64→128 kanala
│   ├── Conv2d(64,128) + BatchNorm + ReLU
│   ├── Conv2d(128,128) + BatchNorm + ReLU
│   ├── MaxPool2d(2x2) → 8x8
│   └── Dropout2d(0.25)
├── Konvolucioni Blok 4: 128→256 kanala
│   ├── Conv2d(128,256) + BatchNorm + ReLU
│   ├── Conv2d(256,256) + BatchNorm + ReLU
│   ├── MaxPool2d(2x2) → 4x4
│   └── Dropout2d(0.25)
├── AdaptiveAvgPool2d(1x1) → 256 karakteristika
└── Klasifikator:
    ├── Linear(256,128) + ReLU + Dropout(0.5)
    ├── Linear(128,64) + ReLU + Dropout(0.5)
    └── Linear(64,2) → [Zdrav, Bolestan]
```

###  **Karakteristike**

- **Batch Normalization**: Stabilizuje obučavanje
- **Dropout**: Sprečava overfitting (0.25 za conv, 0.5 za linear)
- **ReLU aktivacije**: Brže konvergiranje
- **Adaptive Average Pooling**: Fleksibilnost za različite veličine slika
- **Duboka arhitektura**: 4 konvoluciona bloka za bolje karakteristike

## GAN Augmentacija

### 📈 **GANAugmentedDataset Klasa**

```python
# Kombinuje originalne i GAN-generisane slike
augmented_dataset = GANAugmentedDataset(
    original_dataset=train_dataset,
    generator_path='simple_conditional_generator.pth',
    num_generated_per_class=50,  # 50 novih slika po klasi
    transform=transform_train
)
```

**Proces augmentacije**:
1. Učitava obučeni GAN generator
2. Generiše dodatne slike za svaku klasu (zdrav/bolestan)
3. Kombinuje sa originalnim podacima
4. Primenjuje iste transformacije

### 🔄 **Data Augmentation Transformacije**

```python
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),           # Standardizacija veličine
    transforms.RandomHorizontalFlip(0.5),  # Nasumično okretanje
    transforms.RandomRotation(10),         # Nasumična rotacija ±10°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Varijacije boja
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## Instalacija i Pokretanje

### 1. **Preduslovi**

```bash
uv add torch torchvision matplotlib numpy scikit-learn seaborn pillow
```

### 2. **Priprema Podataka**

Prvo pokrenite GAN obučavanje da kreirate podatke:
```bash
uv run conditional_gan.py
```

### 3. **Obučavanje CNN-a**

```bash
uv run plant_classifier_cnn.py
```

## Funkcionalnosti

### 📊 **Vizualizacija Rezultata**

1. **Istorija Obučavanja**:
   - Grafici gubitaka i tačnosti
   - Poređenje train vs validation

2. **Matrica Konfuzije**:
   - Heatmap sa brojem predviđanja
   - Analiza grešaka po klasama

3. **Predviđanja Modela**:
   - Grid sa slikama i predviđanjima
   - Pouzdanost klasifikacije

### 🔬 **Poređenje Performansi**

```python
# Automatsko poređenje sa i bez GAN augmentacije
compare_with_without_gan(data_dir, generator_path)
```

**Izlaz**:
```
POREĐENJE REZULTATA
==========================================
Tačnost bez GAN augmentacije: 0.8500 (85.00%)
Tačnost sa GAN augmentacijom:  0.9200 (92.00%)
Poboljšanje: +7.00 procentnih poena
```

## Generisani Fajlovi

### 📁 **Modeli**
- `cnn_without_gan.pth`: Model obučen samo na originalnim podacima
- `cnn_with_gan.pth`: Model obučen sa GAN augmentacijom

### 📊 **Vizualizacije**
- `training_history.png`: Grafici obučavanja
- `confusion_matrix.png`: Matrica konfuzije
- `model_predictions.png`: Primeri predviđanja

## Prednosti GAN Augmentacije

### ✅ **Zašto Koristiti GAN za Augmentaciju?**

1. **Povećanje Dataset-a**:
   - Originalni: 100 slika → Augmentovani: 200+ slika
   - Balansiran broj uzoraka po klasi

2. **Realistične Slike**:
   - GAN generiše slike koje liče na prave listove
   - Bolje od jednostavnih transformacija

3. **Specifične Karakteristike**:
   - Generator "zna" razliku između zdravih i bolesnih
   - Generiše relevantne uzorke za svaku klasu

4. **Poboljšanje Generalizacije**:
   - Model vidi više varijacija
   - Bolje performanse na test podacima

### 📈 **Očekivana Poboljšanja**

| Metrika | Bez GAN | Sa GAN | Poboljšanje |
|---------|---------|--------|-------------|
| **Tačnost** | 82-88% | 88-94% | +4-8% |
| **Preciznost** | 80-85% | 86-92% | +5-7% |
| **Recall** | 78-84% | 85-91% | +6-8% |
| **F1-Score** | 79-84% | 85-91% | +5-7% |

## Napredne Funkcionalnosti

### 🔧 **Prilagođavanje Hiperparametara**

```python
# U main() funkciji
model = PlantLeafCNN(num_classes=2).to(device)

# Prilagoditi parametre obučavanja
train_cnn(
    model, train_loader, val_loader,
    num_epochs=100,    # Više epoha
    lr=0.0005         # Manja brzina učenja
)
```

### 🎛️ **Augmentacija Parametri**

```python
# Više GAN-generisanih slika
GANAugmentedDataset(
    original_dataset,
    generator_path='simple_conditional_generator.pth',
    num_generated_per_class=100,  # Povećati broj
    nz=100
)
```

### 📊 **Custom Evaluacija**

```python
# Detaljni izveštaj
from sklearn.metrics import classification_report

print(classification_report(true_labels, predictions, 
                          target_names=['Zdrav', 'Bolestan']))
```

## Licenca

Ovaj projekat je za obrazovne i istraživačke svrhe. Poštujte licence korišćenih dataset-a i biblioteka.
