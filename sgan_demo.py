"""
Semi-Supervised GAN Demo for Plant Disease Classification
Demonstrates the SGAN concept without requiring full PyTorch installation
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class SGANDemo:
    """
    Demonstration of SGAN concepts for plant disease classification
    Following the paper: "Plant Disease Classification Using Convolutional Networks and Generative Adversarial Networks"
    """
    
    def __init__(self, num_classes=2):
        self.num_classes = num_classes  # healthy, diseased
        self.fake_class_idx = num_classes  # extra class for fake images
        self.total_classes = num_classes + 1  # real classes + fake class
        
        print("SGAN Demo Initialized")
        print(f"Real classes: {num_classes} (Healthy, Diseased)")
        print(f"Total classes: {self.total_classes} (including Fake class)")
        print("="*60)
    
    def create_synthetic_leaf(self, is_healthy=True, size=(64, 64), add_noise=False):
        """
        Create synthetic leaf images similar to the paper's dataset
        Following the paper's preprocessing: 64x64 pixels, background segmentation
        """
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Base green color for leaf
        if is_healthy:
            green = (34, 139, 34)  # Forest green for healthy
            spots = []
        else:
            green = (85, 107, 47)  # Dark olive green for diseased
            # Add brown spots for disease
            num_spots = random.randint(3, 8)
            spots = [(random.randint(15, size[0]-15), random.randint(15, size[1]-15), 
                     random.randint(3, 8)) for _ in range(num_spots)]
        
        # Draw leaf shape (ellipse)
        margin = 10
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin], fill=green)
        
        # Add disease spots if not healthy
        for spot in spots:
            x, y, radius = spot
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=(139, 69, 19))  # Brown spots
        
        # Add leaf veins/texture
        for i in range(2):
            y = margin + (size[1] - 2*margin) * (i+1) // 3
            draw.line([margin, y, size[0]-margin, y], 
                     fill=(0, 80, 0), width=1)
        
        # Add noise if requested (simulating GAN generation artifacts)
        if add_noise:
            # Convert to numpy for noise addition
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            img = Image.fromarray(img_array)
        
        return img
    
    def simulate_discriminator_classifier_output(self, image_type):
        """
        Simulate the output of the SGAN Discriminator/Classifier
        Returns logits for [healthy, diseased, fake] classes
        
        Following the paper's D/C architecture that outputs NUM_CLASSES + 1
        """
        if image_type == "healthy":
            # Real healthy image: high confidence for healthy class
            logits = [0.8, 0.1, 0.1]  # [healthy, diseased, fake]
        elif image_type == "diseased":
            # Real diseased image: high confidence for diseased class
            logits = [0.1, 0.8, 0.1]  # [healthy, diseased, fake]
        elif image_type == "fake_healthy":
            # Fake healthy image: should be detected as fake
            logits = [0.3, 0.2, 0.5]  # [healthy, diseased, fake]
        elif image_type == "fake_diseased":
            # Fake diseased image: should be detected as fake
            logits = [0.2, 0.3, 0.5]  # [healthy, diseased, fake]
        else:
            # Unknown type
            logits = [0.33, 0.33, 0.34]
        
        # Add some random noise to make it more realistic
        noise = np.random.normal(0, 0.05, 3)
        logits = np.array(logits) + noise
        logits = np.clip(logits, 0.01, 0.99)  # Ensure valid probabilities
        logits = logits / np.sum(logits)  # Normalize to sum to 1
        
        return logits
    
    def calculate_discriminator_loss(self, real_logits, fake_logits, true_labels):
        """
        Calculate discriminator loss exactly as specified in the paper
        
        From paper code:
        def discriminator_loss_for_mcgan(logits_real, logits_fake, true_labels, softmax_loss):
            N, C = logits_real.size()
            fake_labels = Variable((torch.zeros(N) + 23).type(dtype).long())
            return softmax_loss(logits_real, true_labels) + softmax_loss(logits_fake, fake_labels)
        
        "sum of two softmax functions"
        1. Real data classification loss
        2. Fake data detection loss
        """
        # Convert logits to probabilities
        real_probs = real_logits
        fake_probs = fake_logits
        
        # Real data loss: cross-entropy for true class
        real_loss = 0
        for i, label in enumerate(true_labels):
            real_loss += -np.log(real_probs[i][label] + 1e-8)
        real_loss /= len(true_labels)
        
        # Fake data loss: cross-entropy for fake class
        fake_loss = 0
        for i in range(len(fake_probs)):
            fake_loss += -np.log(fake_probs[i][self.fake_class_idx] + 1e-8)
        fake_loss /= len(fake_probs)
        
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss
    
    def calculate_generator_loss(self, fake_logits):
        """
        Calculate generator loss for SGAN
        
        From paper: "maximize log(D(G(z)))" for real classes
        Generator wants fake images to be classified as real classes (not fake)
        """
        # Generator wants fake images to be classified as any real class
        # We use KL divergence to encourage uniform distribution over real classes
        
        gen_loss = 0
        for logits in fake_logits:
            # Only consider real classes (exclude fake class)
            real_class_probs = logits[:self.num_classes]
            real_class_probs = real_class_probs / np.sum(real_class_probs)  # Normalize
            
            # Target: uniform distribution over real classes
            target_prob = 1.0 / self.num_classes
            
            # KL divergence loss
            for prob in real_class_probs:
                gen_loss += target_prob * np.log(target_prob / (prob + 1e-8))
        
        gen_loss /= len(fake_logits)
        return gen_loss
    
    def demonstrate_training_step(self):
        """
        Demonstrate one training step of the SGAN algorithm
        Following Algorithm 1 from the paper
        """
        print("Demonstrating SGAN Training Step (Algorithm 1 from paper)")
        print("-" * 50)
        
        # Step 1: Draw real examples and their labels
        print("1. Drawing real data samples...")
        real_images = []
        real_labels = []
        real_logits = []
        
        for i in range(4):  # Batch size = 4 for demo
            is_healthy = random.choice([True, False])
            label = 0 if is_healthy else 1
            image_type = "healthy" if is_healthy else "diseased"
            
            img = self.create_synthetic_leaf(is_healthy=is_healthy)
            logits = self.simulate_discriminator_classifier_output(image_type)
            
            real_images.append(img)
            real_labels.append(label)
            real_logits.append(logits)
            
            print(f"   Real image {i+1}: {image_type} (label={label})")
            print(f"   D/C output: [healthy={logits[0]:.3f}, diseased={logits[1]:.3f}, fake={logits[2]:.3f}]")
        
        # Step 2: Generate fake samples
        print("\n2. Generating fake samples...")
        fake_images = []
        fake_logits = []
        
        for i in range(4):  # Same batch size
            is_healthy = random.choice([True, False])
            image_type = f"fake_{'healthy' if is_healthy else 'diseased'}"
            
            img = self.create_synthetic_leaf(is_healthy=is_healthy, add_noise=True)
            logits = self.simulate_discriminator_classifier_output(image_type)
            
            fake_images.append(img)
            fake_logits.append(logits)
            
            print(f"   Fake image {i+1}: {image_type}")
            print(f"   D/C output: [healthy={logits[0]:.3f}, diseased={logits[1]:.3f}, fake={logits[2]:.3f}]")
        
        # Step 3: Calculate D/C loss
        print("\n3. Calculating Discriminator/Classifier loss...")
        dc_loss, real_loss, fake_loss = self.calculate_discriminator_loss(
            real_logits, fake_logits, real_labels
        )
        print(f"   Real classification loss: {real_loss:.4f}")
        print(f"   Fake detection loss: {fake_loss:.4f}")
        print(f"   Total D/C loss: {dc_loss:.4f}")
        
        # Step 4: Calculate Generator loss
        print("\n4. Calculating Generator loss...")
        gen_loss = self.calculate_generator_loss(fake_logits)
        print(f"   Generator loss: {gen_loss:.4f}")
        
        # Step 5: Show training progress
        print("\n5. Training Progress:")
        print(f"   D/C learns to classify real images AND detect fakes")
        print(f"   Generator learns to create images that fool D/C")
        print(f"   Semi-supervised: Uses both labeled and unlabeled data")
        
        return {
            'real_images': real_images,
            'fake_images': fake_images,
            'dc_loss': dc_loss,
            'gen_loss': gen_loss,
            'real_logits': real_logits,
            'fake_logits': fake_logits
        }
    
    def visualize_results(self, results):
        """Visualize the training step results"""
        print("\n" + "="*60)
        print("VISUALIZATION OF SGAN TRAINING STEP")
        print("="*60)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Plot real images
        for i in range(4):
            axes[0, i].imshow(results['real_images'][i])
            logits = results['real_logits'][i]
            predicted_class = np.argmax(logits)
            class_names = ['Healthy', 'Diseased', 'Fake']
            axes[0, i].set_title(f'Real Image {i+1}\nPredicted: {class_names[predicted_class]}\nConf: {logits[predicted_class]:.3f}')
            axes[0, i].axis('off')
        
        # Plot fake images
        for i in range(4):
            axes[1, i].imshow(results['fake_images'][i])
            logits = results['fake_logits'][i]
            predicted_class = np.argmax(logits)
            class_names = ['Healthy', 'Diseased', 'Fake']
            axes[1, i].set_title(f'Fake Image {i+1}\nPredicted: {class_names[predicted_class]}\nConf: {logits[predicted_class]:.3f}')
            axes[1, i].axis('off')
        
        plt.suptitle('SGAN Training Step: Real vs Fake Images\nFollowing "Plant Disease Classification Using CNNs and GANs"', fontsize=14)
        plt.tight_layout()
        plt.savefig('sgan_demo_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot loss progression (simulated)
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot simulated training curves showing SGAN convergence"""
        # Simulate training progression
        epochs = np.arange(1, 51)
        
        # Discriminator loss: starts high, decreases as it learns to classify and detect fakes
        dc_loss = 2.0 * np.exp(-epochs/20) + 0.3 + 0.1 * np.random.normal(0, 1, 50)
        
        # Generator loss: starts high, decreases as it learns to fool discriminator
        gen_loss = 1.8 * np.exp(-epochs/25) + 0.4 + 0.1 * np.random.normal(0, 1, 50)
        
        # Classification accuracy: improves over time
        accuracy = 0.5 + 0.4 * (1 - np.exp(-epochs/15)) + 0.05 * np.random.normal(0, 1, 50)
        accuracy = np.clip(accuracy, 0, 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, dc_loss, label='Discriminator/Classifier Loss', color='blue', linewidth=2)
        ax1.plot(epochs, gen_loss, label='Generator Loss', color='red', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('SGAN Training Losses\n(Following Algorithm 1 from Paper)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, accuracy * 100, label='Classification Accuracy', color='green', linewidth=2)
        ax2.axhline(y=78, color='orange', linestyle='--', label='Paper Result (78%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('SGAN Classification Performance\n(Expected: 69-78% as per paper)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig('sgan_demo_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def explain_sgan_architecture(self):
        """Explain the SGAN architecture following the paper"""
        print("\n" + "="*80)
        print("SGAN ARCHITECTURE EXPLANATION")
        print("Following: 'Plant Disease Classification Using CNNs and GANs'")
        print("="*80)
        
        print("\n1. DISCRIMINATOR/CLASSIFIER (D/C) NETWORK:")
        print("   - Single network with dual purpose")
        print("   - Outputs: [Class_0, Class_1, ..., Class_N, Fake_Class]")
        print("   - In our case: [Healthy, Diseased, Fake]")
        print("   - Architecture from paper (improved version):")
        print("     * Conv2d(3->128, k=5) + LeakyReLU + MaxPool")
        print("     * Conv2d(128->64, k=5) + LeakyReLU + MaxPool") 
        print("     * Conv2d(64->32, k=5) + LeakyReLU + MaxPool")
        print("     * Linear(32*4*4 -> 4*4*64) + LeakyReLU")
        print("     * Linear(4*4*64 -> NUM_CLASSES + 1)")
        
        print("\n2. GENERATOR NETWORK:")
        print("   - Creates synthetic plant leaf images")
        print("   - Architecture from paper (improved version):")
        print("     * Linear(noise_dim -> 1024) + BatchNorm")
        print("     * Linear(1024 -> 32*32*128) + BatchNorm")
        print("     * ConvTranspose2d layers to upsample to 64x64")
        print("     * Output: 3-channel RGB images")
        
        print("\n3. TRAINING ALGORITHM (Algorithm 1 from paper):")
        print("   For each iteration:")
        print("   a) Draw m noise samples")
        print("   b) Draw m real data examples")
        print("   c) Update D/C to minimize combined loss:")
        print("      - Real data classification loss")
        print("      - Fake data detection loss")
        print("   d) Update Generator to maximize D/C confusion")
        print("      - Encourage fake images to be classified as real classes")
        
        print("\n4. LOSS FUNCTIONS:")
        print("   Discriminator Loss = Real_Classification_Loss + Fake_Detection_Loss")
        print("   Generator Loss = -log(P(fake_image in real_classes))")
        
        print("\n5. SEMI-SUPERVISED LEARNING:")
        print("   - Uses labeled real data for classification")
        print("   - Uses unlabeled generated data for regularization")
        print("   - Improves generalization on unseen data")
        
        print("\n6. PAPER RESULTS:")
        print("   - Dataset: 86,147 images, 57 plant disease classes")
        print("   - Baseline CNN: >80% accuracy")
        print("   - SGAN D/C: 69-78% accuracy on test data")
        print("   - Improvement: Better performance on unstructured data")
        
        print("\n7. KEY INNOVATION:")
        print("   - Single network for both adversarial and classification tasks")
        print("   - Semi-supervised learning through adversarial training")
        print("   - Leverages synthetic data to improve real data classification")

def main():
    """Main demonstration of SGAN concepts"""
    print("SEMI-SUPERVISED GAN FOR PLANT DISEASE CLASSIFICATION")
    print("Implementation following Stanford CS231n paper by Emanuel Cortes")
    print("Paper: 'Plant Disease Classification Using CNNs and GANs'")
    print("="*80)
    
    # Initialize SGAN demo
    sgan_demo = SGANDemo(num_classes=2)
    
    # Explain the architecture
    sgan_demo.explain_sgan_architecture()
    
    # Demonstrate training step
    print("\n" + "="*80)
    print("DEMONSTRATING SGAN TRAINING STEP")
    print("="*80)
    results = sgan_demo.demonstrate_training_step()
    
    # Visualize results
    sgan_demo.visualize_results(results)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Files generated:")
    print("- sgan_demo_results.png: Training step visualization")
    print("- sgan_demo_training_curves.png: Simulated training curves")
    print("\nFor full implementation, see: sgan_plant_disease.py")
    print("For detailed explanation, see: README_SGAN.md")

if __name__ == "__main__":
    main()
