#!/usr/bin/env python3
"""
Generate synthetic training data for plant disease detection
Creates realistic-looking diseased plant images for training
"""

import numpy as np
import cv2
from pathlib import Path
import random
import os

class SyntheticDataGenerator:
    def __init__(self):
        self.base_dir = Path("datasets/master_field_dataset")
        self.categories = [
            'Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus',
            'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust'
        ]
        
        # Create directories
        for split in ['train', 'val']:
            for category in self.categories:
                (self.base_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    def generate_leaf_base(self, size=(224, 224)):
        """Generate a basic leaf shape"""
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Create leaf shape using ellipse
        center = (size[0]//2, size[1]//2)
        axes = (size[0]//3, size[1]//4)
        
        # Base green color with variation
        green_base = (40 + random.randint(0, 40), 
                     100 + random.randint(0, 100), 
                     30 + random.randint(0, 30))
        
        # Draw leaf shape
        cv2.ellipse(img, center, axes, random.randint(0, 180), 0, 360, green_base, -1)
        
        # Add veins
        for _ in range(3):
            start = (center[0] + random.randint(-30, 30), center[1] + random.randint(-50, 50))
            end = (center[0] + random.randint(-50, 50), center[1] + random.randint(-30, 30))
            vein_color = (green_base[0]-10, green_base[1]-20, green_base[2]-10)
            cv2.line(img, start, end, vein_color, 1)
        
        # Add texture
        noise = np.random.randint(-20, 20, size=(size[0], size[1], 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def add_blight(self, img):
        """Add blight symptoms (brown/dark spots)"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Add 3-7 brown spots
        num_spots = random.randint(3, 7)
        for _ in range(num_spots):
            # Random position
            x = random.randint(20, w-20)
            y = random.randint(20, h-20)
            
            # Brown color variations
            brown = (random.randint(40, 80),
                    random.randint(30, 60),
                    random.randint(20, 40))
            
            # Irregular spot shape
            radius = random.randint(5, 20)
            cv2.circle(img_copy, (x, y), radius, brown, -1)
            
            # Add darker center
            if radius > 10:
                cv2.circle(img_copy, (x, y), radius//2, 
                          (brown[0]-20, brown[1]-20, brown[2]-20), -1)
        
        # Blend with original
        return cv2.addWeighted(img, 0.3, img_copy, 0.7, 0)
    
    def add_healthy_variation(self, img):
        """Add variations to healthy leaves"""
        img_copy = img.copy()
        
        # Make it more vibrant green
        img_copy[:, :, 1] = np.clip(img_copy[:, :, 1] * 1.2, 0, 255)  # Boost green channel
        
        # Add slight shine/gloss effect
        h, w = img.shape[:2]
        for _ in range(2):
            x = random.randint(30, w-30)
            y = random.randint(30, h-30)
            cv2.circle(img_copy, (x, y), 15, (100, 180, 100), -1)
        
        return cv2.addWeighted(img, 0.5, img_copy, 0.5, 0)
    
    def add_leaf_spot(self, img):
        """Add leaf spot symptoms"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Add multiple small spots
        num_spots = random.randint(8, 15)
        for _ in range(num_spots):
            x = random.randint(10, w-10)
            y = random.randint(10, h-10)
            
            # Yellowish-brown spots
            color = (random.randint(60, 100),
                    random.randint(70, 110),
                    random.randint(40, 60))
            
            radius = random.randint(3, 8)
            cv2.circle(img_copy, (x, y), radius, color, -1)
            
            # Add halo effect
            cv2.circle(img_copy, (x, y), radius+2, 
                      (color[0]+20, color[1]+20, color[2]+10), 1)
        
        return cv2.addWeighted(img, 0.4, img_copy, 0.6, 0)
    
    def add_mosaic_virus(self, img):
        """Add mosaic virus pattern"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Create mosaic pattern
        block_size = 20
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if random.random() > 0.5:
                    # Light green/yellow patches
                    color_shift = (random.randint(20, 40),
                                  random.randint(30, 60),
                                  random.randint(10, 30))
                    img_copy[y:y+block_size, x:x+block_size] = np.clip(
                        img_copy[y:y+block_size, x:x+block_size].astype(np.int16) + color_shift,
                        0, 255
                    ).astype(np.uint8)
        
        # Blur to make it more natural
        return cv2.GaussianBlur(img_copy, (5, 5), 0)
    
    def add_nutrient_deficiency(self, img):
        """Add nutrient deficiency symptoms (yellowing)"""
        img_copy = img.copy()
        
        # Overall yellowing
        img_copy[:, :, 0] = np.clip(img_copy[:, :, 0] * 0.8, 0, 255)  # Reduce blue
        img_copy[:, :, 1] = np.clip(img_copy[:, :, 1] * 1.1, 0, 255)  # Slight green boost
        img_copy[:, :, 2] = np.clip(img_copy[:, :, 2] * 0.6, 0, 255)  # Reduce red
        
        # Add yellow patches
        h, w = img.shape[:2]
        for _ in range(5):
            x = random.randint(20, w-20)
            y = random.randint(20, h-20)
            yellow = (random.randint(80, 120),
                     random.randint(100, 150),
                     random.randint(40, 60))
            cv2.circle(img_copy, (x, y), random.randint(10, 25), yellow, -1)
        
        return cv2.addWeighted(img, 0.3, img_copy, 0.7, 0)
    
    def add_powdery_mildew(self, img):
        """Add powdery mildew (white patches)"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Add white powdery patches
        num_patches = random.randint(5, 10)
        for _ in range(num_patches):
            x = random.randint(10, w-10)
            y = random.randint(10, h-10)
            
            # White/gray color
            white = (random.randint(180, 220),
                    random.randint(180, 220),
                    random.randint(180, 220))
            
            # Irregular shape using ellipse
            axes = (random.randint(10, 30), random.randint(10, 25))
            angle = random.randint(0, 180)
            cv2.ellipse(img_copy, (x, y), axes, angle, 0, 360, white, -1)
        
        # Add fuzzy edges
        return cv2.GaussianBlur(img_copy, (3, 3), 0)
    
    def add_rust(self, img):
        """Add rust symptoms (orange/brown pustules)"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        # Add orange/rust colored spots
        num_spots = random.randint(10, 20)
        for _ in range(num_spots):
            x = random.randint(5, w-5)
            y = random.randint(5, h-5)
            
            # Orange/rust color
            rust_color = (random.randint(30, 60),
                         random.randint(60, 100),
                         random.randint(120, 180))
            
            radius = random.randint(2, 6)
            cv2.circle(img_copy, (x, y), radius, rust_color, -1)
            
            # Add raised appearance
            if radius > 3:
                highlight = (rust_color[0]+30, rust_color[1]+30, rust_color[2]+30)
                cv2.circle(img_copy, (x-1, y-1), radius//2, highlight, -1)
        
        return img_copy
    
    def add_background(self, img):
        """Add realistic background"""
        h, w = img.shape[:2]
        
        # Create background
        bg_type = random.choice(['soil', 'grass', 'mixed'])
        
        if bg_type == 'soil':
            # Brown soil background
            bg = np.random.randint(60, 100, (h, w, 3), dtype=np.uint8)
            bg[:, :, 0] = bg[:, :, 0] * 0.8  # Less blue
            bg[:, :, 2] = bg[:, :, 2] * 0.9  # Slightly less red
        elif bg_type == 'grass':
            # Green grass background
            bg = np.random.randint(30, 80, (h, w, 3), dtype=np.uint8)
            bg[:, :, 1] = np.clip(bg[:, :, 1] * 1.5, 0, 255)  # More green
        else:
            # Mixed background
            bg = np.random.randint(40, 120, (h, w, 3), dtype=np.uint8)
        
        # Create mask for leaf
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Combine
        fg = cv2.bitwise_and(img, img, mask=mask)
        bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
        
        return cv2.add(fg, bg)
    
    def add_augmentations(self, img):
        """Add random augmentations for variety"""
        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            center = (img.shape[1]//2, img.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
        # Random blur
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return img
    
    def generate_images(self, num_per_category=100):
        """Generate synthetic images for each category"""
        
        print("\n" + "="*60)
        print("GENERATING SYNTHETIC TRAINING DATA")
        print("="*60)
        
        total_generated = 0
        
        for category in self.categories:
            print(f"\nGenerating {category} images...")
            category_count = 0
            
            for i in range(num_per_category):
                # Generate base leaf
                img = self.generate_leaf_base()
                
                # Add disease symptoms
                if category == 'Blight':
                    img = self.add_blight(img)
                elif category == 'Healthy':
                    img = self.add_healthy_variation(img)
                elif category == 'Leaf_Spot':
                    img = self.add_leaf_spot(img)
                elif category == 'Mosaic_Virus':
                    img = self.add_mosaic_virus(img)
                elif category == 'Nutrient_Deficiency':
                    img = self.add_nutrient_deficiency(img)
                elif category == 'Powdery_Mildew':
                    img = self.add_powdery_mildew(img)
                elif category == 'Rust':
                    img = self.add_rust(img)
                
                # Add background
                img = self.add_background(img)
                
                # Add augmentations
                img = self.add_augmentations(img)
                
                # Save image
                split = 'train' if random.random() < 0.8 else 'val'
                filename = f"synthetic_{category}_{i:04d}.jpg"
                save_path = self.base_dir / split / category / filename
                
                cv2.imwrite(str(save_path), img)
                category_count += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  Generated {i+1}/{num_per_category}")
            
            total_generated += category_count
            print(f"  Total {category}: {category_count} images")
        
        print(f"\nTotal synthetic images generated: {total_generated}")
        return total_generated
    
    def get_stats(self):
        """Get dataset statistics"""
        print("\n" + "="*60)
        print("FINAL DATASET STATISTICS")
        print("="*60)
        
        total = 0
        for split in ['train', 'val']:
            print(f"\n{split.upper()} SET:")
            split_total = 0
            
            for category in self.categories:
                count = len(list((self.base_dir / split / category).glob('*.jpg')))
                print(f"  {category}: {count} images")
                split_total += count
            
            print(f"  Total: {split_total} images")
            total += split_total
        
        print(f"\nTOTAL DATASET: {total} images")
        return total


def main():
    generator = SyntheticDataGenerator()
    
    # Generate synthetic data
    generated = generator.generate_images(num_per_category=150)
    
    # Get final statistics
    total = generator.get_stats()
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {generated} synthetic training images")
    print(f"Total dataset size: {total} images")
    print("\nDataset is ready for training!")
    print("Next: python train_ultimate_model.py")


if __name__ == "__main__":
    main()