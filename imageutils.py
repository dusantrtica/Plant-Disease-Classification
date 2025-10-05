import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch

def download_sample_data():
    """Preuzeti uzorke slika bolesti biljaka"""
    data_dir = "data/plant_leaves"
    os.makedirs(data_dir, exist_ok=True)
    
    # Kreirati direktorijume za uzorke
    healthy_dir = os.path.join(data_dir, "Tomato_healthy")
    diseased_dir = os.path.join(data_dir, "Tomato_blight")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(diseased_dir, exist_ok=True)
    
    print("Direktorijumi za uzorke podataka su kreirani.")
    print("Molimo dodajte vaše slike listova biljaka u:")
    print(f"- Zdravi listovi: {healthy_dir}")
    print(f"- Bolesni listovi: {diseased_dir}")
    print("Za ovu demonstraciju, kreiraćemo neke sintetičke uzorke podataka...")
    
    # Kreirati neke sintetičke uzorke podataka za demonstraciju
    create_synthetic_samples(healthy_dir, diseased_dir)
    
    return data_dir

def create_synthetic_samples(healthy_dir, diseased_dir):
    """Kreirati sintetičke uzorke slika za demonstraciju"""
    from PIL import Image, ImageDraw
    import random
    
    def create_leaf_image(is_healthy=True, size=(64, 64)):
        # Kreirati jednostavnu sliku sličnu listu
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Osnovna zelena boja za list
        if is_healthy:
            green = (34, 139, 34)  # Šumsko zelena za zdrav
            spots = []
        else:
            green = (85, 107, 47)  # Tamno maslinasto zelena za bolestan
            # Dodati braon mrlje za bolest
            spots = [(random.randint(15, size[0]-15), random.randint(15, size[1]-15), 
                     random.randint(3, 8)) for _ in range(random.randint(2, 6))]
        
        # Nacrtati oblik lista (elipsa)
        margin = 10
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin], fill=green)
        
        # Dodati mrlje bolesti ako nije zdrav
        for spot in spots:
            x, y, radius = spot
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=(139, 69, 19))  # Braon mrlje
        
        # Dodati neku teksturu/žile
        for i in range(2):
            y = margin + (size[1] - 2*margin) * (i+1) // 3
            draw.line([margin, y, size[0]-margin, y], 
                     fill=(0, 80, 0), width=1)
        
        return img
    
    # Kreirati sintetičke zdrave uzorke
    for i in range(50):
        img = create_leaf_image(is_healthy=True)
        img.save(os.path.join(healthy_dir, f"healthy_{i:03d}.png"))
    
    # Kreirati sintetičke bolesne uzorke
    for i in range(50):
        img = create_leaf_image(is_healthy=False)
        img.save(os.path.join(diseased_dir, f"diseased_{i:03d}.png"))
    
    print("Kreirano je 100 sintetičkih uzoraka slika (50 zdravih, 50 bolesnih)")

def save_generated_images(fake_images, epoch, labels=None):
    """Sačuvati generisane slike"""
    os.makedirs('conditional_generated_images', exist_ok=True)
    
    if labels is not None:
        # Kreirati grid sa labelima
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(16):
            row = i // 8
            col = i % 8
            img = fake_images[i].cpu().detach()
            img = (img + 1) / 2.0  # Denormalizovati iz [-1,1] u [0,1]
            img = img.permute(1, 2, 0)
            axes[row, col].imshow(img)
            class_name = "Zdrav" if labels[i].item() == 0 else "Bolestan"
            axes[row, col].set_title(f'{class_name}', fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'conditional_generated_images/conditional_samples_epoch_{epoch:03d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Sačuvati i kao grid bez labela
    vutils.save_image(fake_images.detach(),
                      f'conditional_generated_images/grid_epoch_{epoch:03d}.png',
                      normalize=True, nrow=8)

def generate_specific_class(generator, class_label, num_samples=8, nz=100):
    """Generisati slike određene klase"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
        fake_images = generator(noise, labels)
        
        class_name = "zdrav" if class_label == 0 else "bolestan"
        filename = f'generated_{class_name}_leaves.png'
        
        vutils.save_image(fake_images.detach(), filename, normalize=True, nrow=4)
        print(f"Generisano je {num_samples} {class_name} listova, sačuvano kao '{filename}'")
        
        return fake_images
