# Klasifikacija Bolesti Biljaka Korišćenjem Semi-Supervised GAN-a

## 📋 Pregled Projekta

Ovaj projekat implementira **Semi-Supervised Generative Adversarial Network (SGAN)** za klasifikaciju bolesti biljaka na osnovu slika listova. Implementacija je zasnovana na naučnom radu sa Stanforda: [Plant Disease Classification Using Convolutional Networks and Generative Adversarial Networks](https://cs231n.stanford.edu/reports/2017/pdfs/325.pdf).

### 🎯 Cilj

Identifikacija bolesti biljaka iz fotografija listova, što omogućava brže intervencije i smanjenje uticaja bolesti na prinos hrane. Ovo je posebno važno za male farmere u zemljama u razvoju gde se 80% poljoprivredne proizvodnje generiše na malim gazdinstvima.

### 🔬 Šta Je Semi-Supervised GAN?

Tradicionalni GAN (Generative Adversarial Network) se sastoji od dva dela:
1. **Generator (G)** - generiše "lažne" slike iz nasumičnog šuma
2. **Diskriminator (D)** - razlikuje prave slike od generisanih

**Semi-Supervised GAN** proširuje ovu arhitekturu tako što diskriminator postaje i **klasifikator (D/C)**:
- Umesto binarne klasifikacije (pravo/lažno), D/C klasifikuje slike u `N+1` klasa
- `N` pravih klasa (vrste bolesti + zdrave biljke)
- `1` dodatna klasa za "lažne" slike iz generatora

**Ključna prednost**: SGAN postiže bolje rezultate od običnih CNN mreža kada imamo **mali skup podataka** za obučavanje, jer generator pomaže u učenju boljeg predstavljanja podataka.

## 📊 Skup Podataka

Projekat koristi skup od **86,147 slika** bolenih i zdravih listova biljaka:
- **Trening set**: 55,135 slika
- **Validacioni set**: 13,783 slika
- **Test set**: 17,229 slika

### Klase Biljaka i Bolesti

Skup sadrži 15 različitih kategorija:
- **Paprika**: Bakterijska pegavost, Zdrava
- **Krompir**: Rana pegavost, Zdrav, Kasna pegavost
- **Paradajz**: Plamenjača, Bakterijska pegavost, Rana pegavost, Zdrav, Kasna pegavost, Plesni, Septorijska pegavost listova, Grinje, Virus mozaika, Virus žutila listova

## 🏗️ Arhitektura Mreže

### Generator

Generator prati poboljšanu arhitekturu iz rada:
```
Ulaz: Vektor šuma (100 dimenzija)
↓
Linear (100 → 1024) + BatchNorm + ReLU
↓
Linear (1024 → 131,072) + BatchNorm + ReLU
↓
Unflatten → 128 × 32 × 32
↓
ConvTranspose2d (128 → 64, kernel=4, stride=2)
↓
ConvTranspose2d (64 → 32, kernel=4, stride=2)
↓
ConvTranspose2d (32 → 16, kernel=1)
↓
Conv2d + MaxPool2d
↓
Izlaz: Slika 3 × 64 × 64
```

### Diskriminator/Klasifikator (D/C)

D/C mreža prati arhitekturu iz rada sa dubokim konvolucionim slojevima:
```
Ulaz: Slika 3 × 64 × 64
↓
Conv2d (3 → 128, kernel=5) + LeakyReLU + MaxPool2d
↓
Conv2d (128 → 64, kernel=5) + LeakyReLU + MaxPool2d
↓
Conv2d (64 → 32, kernel=5) + LeakyReLU + MaxPool2d
↓
Flatten
↓
Linear (512 → 1024) + LeakyReLU
↓
Linear (1024 → NUM_CLASSES + 1)
↓
Izlaz: Logits za N+1 klasa
```

## 🔧 Funkcije Gubitka

### Gubitak Diskriminatora

Implementiran **tačno kao u radu**:
```python
def discriminator_loss(logits_real, logits_fake, true_labels, softmax_loss):
    N, C = logits_real.size()
    fake_labels = (torch.zeros(N) + (C - 1)).long()
    return softmax_loss(logits_real, true_labels) + softmax_loss(logits_fake, fake_labels)
```

Gubitak je zbir:
1. Cross-entropy gubitak za prave slike (klasifikacija u N klasa)
2. Cross-entropy gubitak za lažne slike (klasifikacija u "lažnu" klasu)

### Gubitak Generatora

Generator želi da prevari D/C - da lažne slike budu klasifikovane kao neka od pravih klasa:
```python
def generator_loss(logits_fake, num_classes):
    # Cilj: uniformna distribucija preko pravih klasa
    target_probs = torch.ones(N, num_classes) / num_classes
    real_class_logits = logits_fake[:, :num_classes]
    return F.kl_div(F.log_softmax(real_class_logits, dim=1), target_probs, reduction='batchmean')
```

## 🚀 Kako Pokrenuti Projekat

### Preduslov

- Python 3.12 ili noviji
- `uv` menadžer paketa ([instalacija](https://github.com/astral-sh/uv))

### 1. Instalacija Zavisnosti

```bash
# uv automatski čita pyproject.toml i instalira zavisnosti
uv sync
```

### 2. Preuzimanje Skupa Podataka

Skup podataka treba da bude organizovan u sledećoj strukturi:
```
data/
├── train/
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   ├── Potato___Early_blight/
│   └── ...
├── val/
│   └── (ista struktura)
└── test/
    └── (ista struktura)
```

Možete koristiti `download_dataset.py` skriptu za automatsko preuzimanje:
```bash
uv run python download_dataset.py
```

### 3. Trening Modela

```bash
uv run python semisupervised_plant_classification.py
```

#### Parametri Treninga

Možete prilagoditi parametre u `train_model()` funkciji:
- `num_epochs`: Broj epoha (podrazumevano: 10)
- `lr`: Learning rate (podrazumevano: 0.0001)
- `noise_dim`: Dimenzija vektora šuma (podrazumevano: 100)
- `batch_size`: Veličina batch-a (podrazumevano: 32)

### 4. Evaluacija Modela

Evaluacija se automatski pokreće nakon treninga. Možete je pokrenuti i ručno:
```python
from semisupervised_plant_classification import evaluate_discriminator
evaluate_discriminator("dc_discriminator")
```

## 📈 Očekivani Rezultati

### Tokom Treninga

- **Gubitak Diskriminatora (Loss_DC)**: Trebao bi da se smanjuje i stabilizuje oko 1.0-2.0
- **Gubitak Generatora (Loss_G)**: Trebao bi da oscilira i polako se smanjuje
- **Vreme treninga**: ~10-30 minuta po epohi (zavisno od hardvera)

### Nakon Treninga

Program generiše:

1. **Grafikon gubitaka** (`sgan_losses.png`)
   - Prikaz Loss_DC i Loss_G tokom iteracija
   - Omogućava vizuelnu proveru konvergencije

2. **Matrica konfuzije** (`sgan_confusion_matrix.png`)
   - Heatmap koja prikazuje performanse po klasama
   - Dijagonala pokazuje tačne predikcije
   - Van-dijagonalni elementi pokazuju greške

3. **Izveštaj o klasifikaciji**
   ```
   Tačnost SGAN Klasifikatora na test skupu: 0.7800 (78.00%)
   
   Detaljan Izveštaj Klasifikacije:
                                    precision    recall  f1-score   support
   pepper_0                            0.85      0.82      0.83       500
   pepper_1                            0.79      0.81      0.80       450
   ...
   ```

### Očekivane Performanse (Prema Radu)

- **Tačnost na test skupu**: 69-78% (zavisno od arhitekture)
- **Poboljšanje**: SGAN radi bolje od običnog CNN-a kada je skup podataka mali
- **Baseline (ResNet)**: ~80% tačnosti sa velikim skupom podataka

## 📂 Struktura Projekta

```
Plant-Disease-Classification/
├── semisupervised_plant_classification.py  # Glavna implementacija
├── conditional_gan.py                      # Conditional GAN (eksperiment)
├── sgan_plant_disease.py                   # Alternativna SGAN implementacija
├── download_dataset.py                     # Skript za preuzimanje podataka
├── imageutils.py                           # Pomoćne funkcije za slike
├── pyproject.toml                          # Zavisnosti projekta
├── dc_discriminator.pth                    # Sačuvan D/C model
├── sgan_confusion_matrix.png               # Matrica konfuzije
├── data/                                   # Skup podataka
│   ├── train/
│   ├── val/
│   └── test/
└── README.md                               # Ova dokumentacija
```

## 🔬 Tehnički Detalji

### Augmentacija Podataka

Trening set koristi:
- `RandomHorizontalFlip(0.5)` - Nasumično horizontalno okretanje
- `RandomRotation(10)` - Nasumična rotacija do ±10°
- `Resize((64, 64))` - Skaliranje na 64×64 piksela
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` - Normalizacija u [-1, 1]

### Optimizacija

- **Optimizator**: Adam sa `betas=(0.5, 0.999)`
- **Learning rate**: 0.0001 (kao u radu)
- **Batch size**: 32

### Uređaj za Trening

Kod automatski detektuje dostupan hardver:
- **MPS** (Apple Silicon - M1/M2/M3)
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 🤔 Zašto SGAN, a ne običan CNN?

**Prednosti SGAN-a**:
1. **Bolje performanse sa malim skupovima podataka**
   - Generator pomaže D/C da nauči bolje reprezentacije
   - Regularizacija kroz adversarial trening

2. **Robusnost**
   - Manje overfitting-a
   - Bolja generalizacija na nove podatke

3. **Implicitna augmentacija podataka**
   - Generator stvara "virtuelne" primere
   - Diskriminator uči da razlikuje distribucije

**Kada koristiti CNN umesto SGAN**:
- Kada imate **veliki skup podataka** (>100k slika)
- Kada je vreme treninga kritično (CNN je brži)
- Kada je jednostavnost važnija od performansi

## 📚 Reference

1. **Glavni Rad**:
   - Emanuel Cortes (2017). "Plant Disease Classification Using Convolutional Networks and Generative Adversarial Networks"
   - Stanford University CS231n
   - [PDF Link](https://cs231n.stanford.edu/reports/2017/pdfs/325.pdf)

2. **Povezani Radovi**:
   - Mohanty et al. (2016). "Using Deep Learning for Image-Based Plant Disease Detection"
   - Odena et al. (2016). "Semi-Supervised Learning with Generative Adversarial Networks"

3. **Dataset**:
   - PlantVillage Dataset (86,147 slika, 25 vrsta biljaka)

## 🛠️ Troubleshooting

### Problem: "Out of Memory" greška

**Rešenje**: Smanjite `batch_size` u `train_model()`:
```python
batch_size = 16  # ili 8
```

### Problem: Spor trening na CPU

**Rešenje**: 
- Smanjite `num_epochs` na 5
- Smanjite dimenzije slike na 32×32
- Koristite GPU ako je dostupan

### Problem: Generator ne konvergira

**Simptomi**: Loss_G raste ili ne pada
**Rešenje**:
- Smanjite learning rate za generator: `lr=0.00005`
- Povećajte broj D/C koraka pre G koraka
- Proverite balans između Loss_DC i Loss_G

### Problem: Loša tačnost (<50%)

**Mogući uzroci**:
1. Premalo epoha - povećajte `num_epochs` na 20-50
2. Previsok learning rate - smanjite na `1e-5`
3. Neujednačen skup podataka - proverite distribuciju klasa

## 📧 Kontakt i Doprinosi

Ovaj projekat je edukativna implementacija za istraživačke svrhe. Za pitanja, predloge ili doprinose, molimo:
1. Otvorite Issue na GitHub-u
2. Pošaljite Pull Request sa opisom promena
3. Uverite se da kodujete u skladu sa postojećim stilom

## 📄 Licenca

Ovaj projekat je otvorenog koda i namenjen je za edukativne i istraživačke svrhe.

---

**Napomena**: Rezultati mogu varirati u zavisnosti od hardvera, parametara treninga i specifičnog skupa podataka. Za najbolje rezultate, preporučuje se eksperimentisanje sa hiperparametrima i arhitekturom mreže.

**Sretno sa klasifikacijom bolesti biljaka! 🌱🔬**

