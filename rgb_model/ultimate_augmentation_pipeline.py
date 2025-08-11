#!/usr/bin/env python3
"""
Ultimate augmentation pipeline for real-world robustness
This creates realistic field conditions from any input image
"""

import cv2
import numpy as np
import albumentations as A
from albumentations import *
import tensorflow as tf
from pathlib import Path
import random
from PIL import Image, ImageEnhance

class UltimateAugmentor:
    """
    Creates realistic field conditions through advanced augmentation
    """
    
    def __init__(self):
        # Create multiple augmentation pipelines
        self.setup_augmentation_pipelines()
        
    def setup_augmentation_pipelines(self):
        """Setup different augmentation strategies"""
        
        # 1. Weather augmentation pipeline
        self.weather_pipeline = A.Compose([
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=1),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.1, p=1),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, 
                                  num_flare_circles_lower=3, num_flare_circles_upper=7, p=1),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                              num_shadows_upper=3, shadow_dimension=5, p=1),
            ], p=0.7),
        ])
        
        # 2. Camera/photography augmentation
        self.camera_pipeline = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.Defocus(radius=(3, 5), alias_blur=(0.1, 0.5), p=1),
            ], p=0.3),
            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            ], p=0.3),
            A.OneOf([
                A.ImageCompression(quality_lower=60, quality_upper=90, p=1),
                A.Downscale(scale_min=0.5, scale_max=0.75, p=1),
            ], p=0.2),
        ])
        
        # 3. Realistic field conditions
        self.field_pipeline = A.Compose([
            # Perspective changes (different viewing angles)
            A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.3),
            
            # Rotation (phone not perfectly aligned)
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.5),
            
            # Color variations (different times of day)
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, 
                                   val_shift_limit=20, p=1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
            ], p=0.7),
            
            # Lighting conditions
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.RandomGamma(gamma_limit=(60, 140), p=1),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
            ], p=0.5),
        ])
        
        # 4. Advanced augmentation (MixUp, CutMix style)
        self.advanced_pipeline = A.Compose([
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=1, fill_value=0, p=1),
                A.GridDropout(ratio=0.2, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            ], p=0.3),
        ])
        
        # 5. Extreme augmentation for hard examples
        self.extreme_pipeline = A.Compose([
            A.RandomRain(brightness_coefficient=0.5, drop_width=2, blur_value=7, p=0.5),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, p=0.3),
            A.RandomShadow(num_shadows_lower=2, num_shadows_upper=5, p=0.4),
            A.MotionBlur(blur_limit=11, p=0.3),
            A.ISONoise(intensity=(0.3, 0.8), p=0.4),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.6),
            A.Rotate(limit=45, p=0.5),
            A.Perspective(scale=(0.05, 0.2), p=0.4),
        ])
    
    def add_background(self, image, background_type='natural'):
        """Replace background with realistic outdoor scenes"""
        
        # Simple background replacement (in production, use segmentation)
        h, w = image.shape[:2]
        
        if background_type == 'soil':
            # Brown soil texture
            background = np.random.randint(60, 120, (h, w, 3), dtype=np.uint8)
            background[:, :, 0] = background[:, :, 0] * 0.7  # Less blue
            background[:, :, 1] = background[:, :, 1] * 0.9  # Moderate green
            
        elif background_type == 'grass':
            # Green grass texture
            background = np.random.randint(30, 100, (h, w, 3), dtype=np.uint8)
            background[:, :, 1] = background[:, :, 1] * 1.5  # More green
            background = np.clip(background, 0, 255).astype(np.uint8)
            
        elif background_type == 'garden':
            # Mixed garden background
            background = np.random.randint(40, 140, (h, w, 3), dtype=np.uint8)
            
        else:
            return image
        
        # Blend with original (simple alpha blending)
        alpha = 0.3
        result = cv2.addWeighted(image, 1-alpha, background, alpha, 0)
        
        return result
    
    def add_realistic_artifacts(self, image):
        """Add realistic artifacts found in field photography"""
        
        h, w = image.shape[:2]
        result = image.copy()
        
        # Add water droplets
        if random.random() < 0.2:
            num_drops = random.randint(3, 10)
            for _ in range(num_drops):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                radius = random.randint(2, 8)
                cv2.circle(result, (x, y), radius, (240, 240, 240), -1)
                cv2.circle(result, (x, y), radius, (200, 200, 200), 1)
        
        # Add dirt/dust specs
        if random.random() < 0.3:
            num_specs = random.randint(5, 20)
            for _ in range(num_specs):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                result[y, x] = [random.randint(50, 100)] * 3
        
        # Add partial shadows (like from other leaves)
        if random.random() < 0.3:
            shadow_mask = np.ones((h, w), dtype=np.float32)
            num_shadows = random.randint(1, 3)
            for _ in range(num_shadows):
                x1 = random.randint(0, w//2)
                x2 = random.randint(w//2, w)
                y1 = random.randint(0, h//2)
                y2 = random.randint(h//2, h)
                shadow_mask[y1:y2, x1:x2] *= random.uniform(0.5, 0.8)
            
            result = (result * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
        
        return result
    
    def mixup(self, image1, image2, alpha=0.2):
        """MixUp augmentation - blend two images"""
        
        # Ensure same size
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h))
        
        # Random mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Mix images
        mixed = lam * image1 + (1 - lam) * image2
        
        return mixed.astype(np.uint8)
    
    def cutmix(self, image1, image2):
        """CutMix augmentation - cut and paste regions"""
        
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h))
        
        # Random box
        lam = np.random.beta(1.0, 1.0)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        result = image1.copy()
        result[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        return result
    
    def apply_all_augmentations(self, image, intensity='medium'):
        """Apply complete augmentation pipeline"""
        
        # Select pipeline based on intensity
        if intensity == 'light':
            augmented = self.field_pipeline(image=image)['image']
            
        elif intensity == 'medium':
            augmented = self.field_pipeline(image=image)['image']
            augmented = self.weather_pipeline(image=augmented)['image']
            augmented = self.camera_pipeline(image=augmented)['image']
            
        elif intensity == 'heavy':
            augmented = self.extreme_pipeline(image=image)['image']
            augmented = self.add_realistic_artifacts(augmented)
            
        else:  # extreme
            augmented = self.extreme_pipeline(image=image)['image']
            augmented = self.add_realistic_artifacts(augmented)
            # Add background
            bg_type = random.choice(['soil', 'grass', 'garden'])
            augmented = self.add_background(augmented, bg_type)
        
        return augmented
    
    def generate_augmented_batch(self, image, num_augmented=10):
        """Generate multiple augmented versions of an image"""
        
        augmented_images = []
        
        for i in range(num_augmented):
            # Vary intensity
            if i < 3:
                intensity = 'light'
            elif i < 6:
                intensity = 'medium'
            elif i < 8:
                intensity = 'heavy'
            else:
                intensity = 'extreme'
            
            aug_img = self.apply_all_augmentations(image, intensity)
            augmented_images.append(aug_img)
        
        return augmented_images


def test_augmentation_pipeline():
    """Test the augmentation pipeline"""
    
    print("Testing Ultimate Augmentation Pipeline")
    print("="*60)
    
    # Create augmentor
    augmentor = UltimateAugmentor()
    
    # Create synthetic test image
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 100
    test_image[50:150, 50:150, 1] = 200  # Green square (leaf)
    
    # Generate augmented versions
    augmented_batch = augmentor.generate_augmented_batch(test_image, num_augmented=5)
    
    # Save examples
    output_dir = Path("augmentation_examples")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "original.jpg"), test_image)
    
    for i, aug_img in enumerate(augmented_batch):
        cv2.imwrite(str(output_dir / f"augmented_{i}.jpg"), aug_img)
    
    print(f"Saved {len(augmented_batch)} augmented examples to {output_dir}")
    
    # Test specific augmentations
    print("\nTesting specific augmentations:")
    
    # Weather
    weather_aug = augmentor.weather_pipeline(image=test_image)['image']
    cv2.imwrite(str(output_dir / "weather.jpg"), weather_aug)
    print("- Weather augmentation applied")
    
    # Camera effects
    camera_aug = augmentor.camera_pipeline(image=test_image)['image']
    cv2.imwrite(str(output_dir / "camera.jpg"), camera_aug)
    print("- Camera effects applied")
    
    # Field conditions
    field_aug = augmentor.field_pipeline(image=test_image)['image']
    cv2.imwrite(str(output_dir / "field.jpg"), field_aug)
    print("- Field conditions applied")
    
    # Extreme
    extreme_aug = augmentor.extreme_pipeline(image=test_image)['image']
    cv2.imwrite(str(output_dir / "extreme.jpg"), extreme_aug)
    print("- Extreme augmentation applied")
    
    print("\nAugmentation pipeline ready for training!")


if __name__ == "__main__":
    # Install required packages
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "albumentations", "--quiet"])
    
    # Test the pipeline
    test_augmentation_pipeline()