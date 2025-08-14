#!/usr/bin/env python3
"""
CycleGAN-Style Augmentation for Plant Disease Images
Transforms clean lab images to realistic field-like images
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentationType(Enum):
    """Types of augmentation to apply"""
    BACKGROUND = "background"
    LIGHTING = "lighting"
    BLUR = "blur"
    NOISE = "noise"
    SHADOWS = "shadows"
    ARTIFACTS = "artifacts"
    WEATHER = "weather"

@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    background_prob: float = 0.7
    lighting_prob: float = 0.8
    blur_prob: float = 0.5
    noise_prob: float = 0.6
    shadow_prob: float = 0.4
    artifact_prob: float = 0.3
    weather_prob: float = 0.4
    intensity: float = 0.7  # Overall intensity 0-1

class CycleGANAugmentor:
    """
    Simulates CycleGAN-style transformations without needing a trained GAN
    Converts lab images to field-like images
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.background_textures = self._generate_background_textures()
        
    def _generate_background_textures(self) -> List[np.ndarray]:
        """Generate various natural background textures"""
        textures = []
        
        # Soil texture
        soil = self._create_perlin_noise(224, 224, scale=30, octaves=4)
        soil = (soil * 80 + 120).astype(np.uint8)  # Brown tones
        soil = cv2.cvtColor(soil, cv2.COLOR_GRAY2BGR)
        soil[:,:,0] = soil[:,:,0] * 0.6  # Reduce blue
        soil[:,:,1] = soil[:,:,1] * 0.8  # Reduce green
        textures.append(soil)
        
        # Grass texture
        grass = self._create_perlin_noise(224, 224, scale=10, octaves=6)
        grass = (grass * 60 + 80).astype(np.uint8)
        grass = cv2.cvtColor(grass, cv2.COLOR_GRAY2BGR)
        grass[:,:,0] = grass[:,:,0] * 0.4  # Reduce blue
        grass[:,:,2] = grass[:,:,2] * 0.5  # Reduce red
        textures.append(grass)
        
        # Mulch texture
        mulch = self._create_perlin_noise(224, 224, scale=5, octaves=3)
        mulch = (mulch * 100 + 60).astype(np.uint8)
        mulch = cv2.cvtColor(mulch, cv2.COLOR_GRAY2BGR)
        textures.append(mulch)
        
        return textures
    
    def _create_perlin_noise(self, width: int, height: int, scale: float = 100, 
                            octaves: int = 6, persistence: float = 0.5) -> np.ndarray:
        """Generate Perlin noise for natural textures"""
        noise = np.zeros((height, width))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave
            
            x = np.linspace(0, freq * scale / 100, width)
            y = np.linspace(0, freq * scale / 100, height)
            X, Y = np.meshgrid(x, y)
            
            # Simple noise generation
            noise_octave = np.sin(X) * np.cos(Y) + np.sin(X * 1.5) * np.cos(Y * 1.5)
            noise_octave = (noise_octave - noise_octave.min()) / (noise_octave.max() - noise_octave.min())
            noise += noise_octave * amp
            
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return (noise * 255).astype(np.uint8)
    
    def apply_background_replacement(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Replace white/uniform backgrounds with natural textures"""
        if mask is None:
            # Create mask using color thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_not(mask)
            
            # Dilate mask to include edges
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Select random background
        background = random.choice(self.background_textures)
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
        
        # Add variation to background
        variation = np.random.normal(1.0, 0.1, background.shape)
        background = np.clip(background * variation, 0, 255).astype(np.uint8)
        
        # Blend with original
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = image * mask_3channel + background * (1 - mask_3channel)
        
        return result.astype(np.uint8)
    
    def apply_realistic_lighting(self, image: np.ndarray) -> np.ndarray:
        """Simulate various outdoor lighting conditions"""
        lighting_type = random.choice(['sunny', 'cloudy', 'shade', 'golden_hour'])
        
        if lighting_type == 'sunny':
            # Bright, high contrast
            image = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
            # Add slight yellow tint
            image[:,:,2] = np.clip(image[:,:,2] * 1.1, 0, 255)
            
        elif lighting_type == 'cloudy':
            # Lower contrast, cooler tones
            image = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
            # Add blue tint
            image[:,:,0] = np.clip(image[:,:,0] * 1.1, 0, 255)
            
        elif lighting_type == 'shade':
            # Darker, lower contrast
            image = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)
            
        elif lighting_type == 'golden_hour':
            # Warm, orange tones
            image[:,:,1] = np.clip(image[:,:,1] * 1.1, 0, 255)
            image[:,:,2] = np.clip(image[:,:,2] * 1.2, 0, 255)
            
        return image
    
    def apply_camera_blur(self, image: np.ndarray) -> np.ndarray:
        """Simulate camera focus issues and motion blur"""
        blur_type = random.choice(['motion', 'focus', 'lens'])
        
        if blur_type == 'motion':
            # Motion blur
            size = random.randint(5, 15)
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            image = cv2.filter2D(image, -1, kernel)
            
        elif blur_type == 'focus':
            # Out of focus blur
            ksize = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            
        elif blur_type == 'lens':
            # Lens blur (radial)
            rows, cols = image.shape[:2]
            center = (cols//2, rows//2)
            
            # Create radial gradient
            Y, X = np.ogrid[:rows, :cols]
            dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            blur_mask = dist / max_dist
            
            # Apply varying blur
            blurred = cv2.GaussianBlur(image, (9, 9), 0)
            image = image * (1 - blur_mask[:,:,np.newaxis]) + blurred * blur_mask[:,:,np.newaxis]
            
        return image.astype(np.uint8)
    
    def apply_camera_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic camera sensor noise"""
        noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson'])
        
        if noise_type == 'gaussian':
            # Gaussian noise (common in low light)
            noise = np.random.normal(0, random.uniform(5, 15), image.shape)
            image = np.clip(image + noise, 0, 255)
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            prob = random.uniform(0.01, 0.03)
            rnd = np.random.random(image.shape[:2])
            image[rnd < prob/2] = 0
            image[rnd > 1 - prob/2] = 255
            
        elif noise_type == 'poisson':
            # Poisson noise (shot noise)
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            image = np.random.poisson(image * vals) / float(vals)
            image = np.clip(image, 0, 255)
            
        return image.astype(np.uint8)
    
    def apply_shadows(self, image: np.ndarray) -> np.ndarray:
        """Add realistic shadows from leaves, branches, etc."""
        shadow_type = random.choice(['diagonal', 'patches', 'edge'])
        
        if shadow_type == 'diagonal':
            # Diagonal shadow across image
            rows, cols = image.shape[:2]
            shadow = np.ones_like(image, dtype=np.float32)
            
            # Create diagonal gradient
            for i in range(rows):
                for j in range(cols):
                    if i + j < rows:
                        shadow[i, j] *= random.uniform(0.5, 0.8)
            
            image = (image * shadow).astype(np.uint8)
            
        elif shadow_type == 'patches':
            # Random shadow patches
            num_patches = random.randint(2, 5)
            shadow_mask = np.ones(image.shape[:2], dtype=np.float32)
            
            for _ in range(num_patches):
                center_x = random.randint(0, image.shape[1])
                center_y = random.randint(0, image.shape[0])
                radius = random.randint(20, 60)
                
                Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                
                shadow_intensity = random.uniform(0.4, 0.7)
                shadow_mask[dist <= radius] *= shadow_intensity
            
            image = (image * shadow_mask[:,:,np.newaxis]).astype(np.uint8)
            
        elif shadow_type == 'edge':
            # Vignetting effect
            rows, cols = image.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/4)
            kernel_y = cv2.getGaussianKernel(rows, rows/4)
            kernel = kernel_y * kernel_x.T
            mask = kernel / kernel.max()
            mask = 0.3 + 0.7 * mask
            
            image = (image * mask[:,:,np.newaxis]).astype(np.uint8)
            
        return image
    
    def apply_field_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add dust, water droplets, and other field artifacts"""
        artifact_type = random.choice(['dust', 'water_droplets', 'debris'])
        
        if artifact_type == 'dust':
            # Add dust particles
            dust_layer = np.random.random(image.shape[:2]) * 50
            dust_mask = np.random.random(image.shape[:2]) > 0.98
            image = image.astype(np.float32)
            image[:,:,0] += dust_layer * dust_mask
            image[:,:,1] += dust_layer * dust_mask * 0.9
            image[:,:,2] += dust_layer * dust_mask * 0.8
            image = np.clip(image, 0, 255)
            
        elif artifact_type == 'water_droplets':
            # Simulate water droplets
            num_drops = random.randint(3, 10)
            for _ in range(num_drops):
                center = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
                radius = random.randint(2, 8)
                
                # Create refractive effect
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.circle(mask, center, radius, 255, -1)
                
                # Distort through droplet
                droplet_area = image[max(0, center[1]-radius):min(image.shape[0], center[1]+radius),
                                   max(0, center[0]-radius):min(image.shape[1], center[0]+radius)]
                if droplet_area.size > 0:
                    droplet_area = cv2.GaussianBlur(droplet_area, (3, 3), 0)
                    
        elif artifact_type == 'debris':
            # Add small debris/dirt
            num_debris = random.randint(5, 15)
            for _ in range(num_debris):
                x = random.randint(0, image.shape[1]-5)
                y = random.randint(0, image.shape[0]-5)
                size = random.randint(1, 3)
                color = random.randint(30, 80)
                cv2.circle(image, (x, y), size, (color, color, color), -1)
                
        return image.astype(np.uint8)
    
    def apply_weather_effects(self, image: np.ndarray) -> np.ndarray:
        """Simulate weather conditions like humidity, rain, etc."""
        weather = random.choice(['humid', 'misty', 'after_rain'])
        
        if weather == 'humid':
            # Hazy, lower contrast
            image = cv2.convertScaleAbs(image, alpha=0.8, beta=20)
            # Add slight blur
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
        elif weather == 'misty':
            # Add fog effect
            fog = np.ones_like(image) * 200
            blend_factor = random.uniform(0.2, 0.4)
            image = cv2.addWeighted(image, 1-blend_factor, fog, blend_factor, 0)
            
        elif weather == 'after_rain':
            # Increase saturation and add shine
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            img_hsv[:,:,1] *= 1.3  # Increase saturation
            img_hsv[:,:,2] *= 1.1  # Slight brightness increase
            img_hsv = np.clip(img_hsv, 0, 255)
            image = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        return image
    
    def augment_image(self, image: np.ndarray, intensity: Optional[float] = None) -> np.ndarray:
        """
        Apply CycleGAN-style augmentation to convert lab image to field-like image
        
        Args:
            image: Input image (BGR format)
            intensity: Augmentation intensity (0-1), None uses config default
            
        Returns:
            Augmented image
        """
        if intensity is None:
            intensity = self.config.intensity
            
        augmented = image.copy()
        
        # Apply augmentations based on probability
        if random.random() < self.config.background_prob * intensity:
            augmented = self.apply_background_replacement(augmented)
            
        if random.random() < self.config.lighting_prob * intensity:
            augmented = self.apply_realistic_lighting(augmented)
            
        if random.random() < self.config.blur_prob * intensity:
            augmented = self.apply_camera_blur(augmented)
            
        if random.random() < self.config.noise_prob * intensity:
            augmented = self.apply_camera_noise(augmented)
            
        if random.random() < self.config.shadow_prob * intensity:
            augmented = self.apply_shadows(augmented)
            
        if random.random() < self.config.artifact_prob * intensity:
            augmented = self.apply_field_artifacts(augmented)
            
        if random.random() < self.config.weather_prob * intensity:
            augmented = self.apply_weather_effects(augmented)
            
        return augmented
    
    def batch_augment(self, images: List[np.ndarray], 
                     intensity: Optional[float] = None) -> List[np.ndarray]:
        """Augment a batch of images"""
        return [self.augment_image(img, intensity) for img in images]

def test_augmentor():
    """Test the augmentor with a sample image"""
    augmentor = CycleGANAugmentor()
    
    # Create a simple test image (white background with green blob)
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.circle(test_image, (112, 112), 50, (0, 255, 0), -1)
    
    # Apply augmentation
    augmented = augmentor.augment_image(test_image)
    
    # Save results
    cv2.imwrite('test_original.jpg', test_image)
    cv2.imwrite('test_augmented.jpg', augmented)
    print("Test images saved: test_original.jpg and test_augmented.jpg")

if __name__ == "__main__":
    test_augmentor()