"""
Skript za testiranje i generisanje slika pomoću obučenog Conditional GAN-a
"""

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from conditional_gan import ConditionalGenerator, ConditionalDiscriminator
import os
import numpy as np

def load_conditional_models(gen_path='conditional_generator.pth', disc_path='conditional_discriminator.pth', nz=100):
    """Učitati obučene conditional modele"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Kreirati modele
    generator = ConditionalGenerator(nz=nz, num_classes=2).to(device)
    discriminator = ConditionalDiscriminator(num_classes=2).to(device)
    
    if os.path.exists(gen_path) and os.path.exists(disc_path):
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        generator.eval()
        discriminator.eval()
        print(f"Modeli učitani iz {gen_path} i {disc_path}")
        return generator, discriminator, device
    else:
        print(f"Fajlovi modela nisu pronađeni. Molimo prvo obučite Conditional GAN pokretanjem: python conditional_gan.py")
        return None, None, device

def generate_class_samples(generator, device, class_label, num_samples=16, nz=100):
    """Generisati uzorke određene klase"""
    
    with torch.no_grad():
        # Kreirati nasumični šum
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        
        # Kreirati labele za određenu klasu
        labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        
        # Generisati slike
        fake_images = generator(noise, labels)
        
        return fake_images

def test_discriminator_classification(discriminator, generator, device, nz=100):
    """Testirati sposobnost diskriminatora da klasifikuje klase"""
    
    print("\\nTestiranje klasifikacije diskriminatora...")
    
    with torch.no_grad():
        results = []
        class_names = ["Zdrav", "Bolestan"]
        
        for class_label in [0, 1]:  # 0 = zdrav, 1 = bolestan
            # Generisati uzorke za svaku klasu
            noise = torch.randn(10, nz, 1, 1, device=device)
            labels = torch.full((10,), class_label, dtype=torch.long, device=device)
            fake_images = generator(noise, labels)
            
            # Testirati diskriminator na generisanim slikama
            adv_output, class_output = discriminator(fake_images, labels)
            
            # Analizirati rezultate
            predicted_classes = torch.argmax(class_output, dim=1)
            accuracy = (predicted_classes == labels).float().mean().item()
            
            print(f"Klasa {class_names[class_label]}:")
            print(f"  - Tačnost klasifikacije: {accuracy:.2%}")
            print(f"  - Prosečna real/fake score: {adv_output.mean().item():.3f}")
            
            results.append({
                'class': class_label,
                'accuracy': accuracy,
                'avg_score': adv_output.mean().item()
            })
    
    return results

def create_class_comparison_grid(generator, device, nz=100):
    """Kreirati grid poređenja između klasa"""
    
    print("Kreiranje grid poređenja...")
    
    with torch.no_grad():
        # Generisati uzorke za obe klase
        noise = torch.randn(16, nz, 1, 1, device=device)
        
        # Prva polovina - zdrav (0), druga polovina - bolestan (1)
        labels = torch.cat([torch.zeros(8, dtype=torch.long), 
                           torch.ones(8, dtype=torch.long)]).to(device)
        
        fake_images = generator(noise, labels)
        
        # Kreirati matplotlib grid
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle('Conditional GAN - Poređenje Klasa', fontsize=16)
        
        for i in range(16):
            row = i // 8
            col = i % 8
            
            img = fake_images[i].cpu().detach()
            img = (img + 1) / 2.0  # Denormalizovati
            img = img.permute(1, 2, 0)
            
            axes[row, col].imshow(img)
            
            if row == 0:
                axes[row, col].set_title('Zdrav', fontsize=10, color='green')
            else:
                axes[row, col].set_title('Bolestan', fontsize=10, color='red')
            
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('class_comparison_grid.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Sačuvati i kao običan grid
        vutils.save_image(fake_images.detach(),
                          'conditional_comparison.png',
                          normalize=True, nrow=8)
        
        print("Grid poređenja sačuvan kao 'class_comparison_grid.png' i 'conditional_comparison.png'")

def interpolate_between_classes(generator, device, nz=100, num_steps=10):
    """Interpolacija između različitih klasa"""
    
    print("Kreiranje interpolacije između klasa...")
    
    with torch.no_grad():
        # Fiksni šum za konzistentnost
        noise = torch.randn(1, nz, 1, 1, device=device)
        
        interpolated_images = []
        
        for i in range(num_steps):
            # Interpolacija između klasa 0 i 1
            alpha = i / (num_steps - 1)
            
            # Za demonstraciju, koristićemo label embedding interpolaciju
            if alpha < 0.5:
                label = torch.tensor([0], dtype=torch.long, device=device)  # Zdrav
            else:
                label = torch.tensor([1], dtype=torch.long, device=device)  # Bolestan
            
            fake_img = generator(noise, label)
            interpolated_images.append(fake_img)
        
        # Spojiti sve uzorke
        interpolated_batch = torch.cat(interpolated_images, dim=0)
        
        # Sačuvati interpolovanu sekvencu
        vutils.save_image(interpolated_batch.detach(),
                          'class_interpolation.png',
                          normalize=True, nrow=num_steps)
        
        print("Interpolacija između klasa sačuvana kao 'class_interpolation.png'")

def generate_diverse_samples(generator, device, samples_per_class=20, nz=100):
    """Generisati raznolike uzorke za svaku klasu"""
    
    print(f"Generisanje {samples_per_class} uzoraka po klasi...")
    
    class_names = ["zdrav", "bolestan"]
    
    for class_label in [0, 1]:
        with torch.no_grad():
            # Generisati više uzoraka za bolju raznolikost
            noise = torch.randn(samples_per_class, nz, 1, 1, device=device)
            labels = torch.full((samples_per_class,), class_label, dtype=torch.long, device=device)
            
            fake_images = generator(noise, labels)
            
            # Sačuvati uzorke
            filename = f'diverse_{class_names[class_label]}_samples.png'
            vutils.save_image(fake_images.detach(),
                              filename,
                              normalize=True, nrow=5)
            
            print(f"Sačuvano {samples_per_class} {class_names[class_label]} uzoraka kao '{filename}'")

def main():
    """Glavna funkcija testiranja"""
    
    print("Conditional GAN - Testiranje i Generisanje")
    print("==========================================")
    
    # Učitati modele
    generator, discriminator, device = load_conditional_models()
    
    if generator is None or discriminator is None:
        return
    
    print(f"Koristi se uređaj: {device}")
    
    # 1. Testirati klasifikaciju diskriminatora
    test_results = test_discriminator_classification(discriminator, generator, device)
    
    # 2. Kreirati grid poređenja
    create_class_comparison_grid(generator, device)
    
    # 3. Generisati interpolaciju između klasa
    interpolate_between_classes(generator, device)
    
    # 4. Generisati raznolike uzorke
    generate_diverse_samples(generator, device, samples_per_class=16)
    
    # 5. Generisati specifične uzorke za svaku klasu
    print("\\nGenerisanje specifičnih uzoraka...")
    
    healthy_samples = generate_class_samples(generator, device, class_label=0, num_samples=8)
    diseased_samples = generate_class_samples(generator, device, class_label=1, num_samples=8)
    
    vutils.save_image(healthy_samples.detach(),
                      'final_healthy_samples.png',
                      normalize=True, nrow=4)
    
    vutils.save_image(diseased_samples.detach(),
                      'final_diseased_samples.png',
                      normalize=True, nrow=4)
    
    print("Finalni uzorci sačuvani kao 'final_healthy_samples.png' i 'final_diseased_samples.png'")
    
    print("\\nTestiranje završeno! Proverite generisane fajlove.")

if __name__ == "__main__":
    main()
