"""
Dataset loader for real thermal plant images
Supports IEEE DataPort Hydroponic Dataset and custom thermal datasets
"""

import os
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import zipfile
import requests
from tqdm import tqdm

class HydroponicDatasetLoader:
    """
    Loader for IEEE DataPort Hydroponic Lettuce Dataset
    Dataset: 30 plants monitored for 66 days with thermal and RGB images
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.metadata = None
        self.thermal_images = []
        self.rgb_images = []
        self.labels = []
        
    def download_dataset(self, url: Optional[str] = None):
        """Download dataset from IEEE DataPort (requires authentication)"""
        print("To download the IEEE DataPort Hydroponic Dataset:")
        print("1. Visit: https://ieee-dataport.org/open-access/lettuce-dataset")
        print("2. Create a free IEEE DataPort account")
        print("3. Download the dataset ZIP file")
        print(f"4. Extract to: {self.data_path}")
        print("\nThe dataset contains:")
        print("- 30 hydroponic lettuce plants")
        print("- 66 days of monitoring")
        print("- Thermal + RGB images")
        print("- Environmental sensor data")
        
    def load_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load and parse the hydroponic dataset"""
        
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}")
            self.download_dataset()
            return np.array([]), {}
        
        # Load metadata
        metadata_path = os.path.join(self.data_path, 'metadata.csv')
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        
        # Dataset structure
        thermal_dir = os.path.join(self.data_path, 'thermal')
        rgb_dir = os.path.join(self.data_path, 'rgb')
        labels_file = os.path.join(self.data_path, 'labels.json')
        
        # Load labels if available
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        # Load thermal images
        thermal_files = sorted(os.listdir(thermal_dir)) if os.path.exists(thermal_dir) else []
        
        images = []
        labels = {
            'water_stress': [],
            'disease': [],
            'nutrients': [],
            'segmentation': [],
            'plant_id': [],
            'day': [],
            'timestamp': []
        }
        
        for thermal_file in tqdm(thermal_files, desc="Loading thermal images"):
            if not thermal_file.endswith(('.png', '.jpg', '.tiff', '.npy')):
                continue
                
            # Parse filename (expected format: plant{ID}_day{DAY}_{TIMESTAMP}.ext)
            parts = thermal_file.split('_')
            plant_id = int(parts[0].replace('plant', ''))
            day = int(parts[1].replace('day', ''))
            timestamp = parts[2].split('.')[0]
            
            # Load thermal image
            thermal_path = os.path.join(thermal_dir, thermal_file)
            
            if thermal_file.endswith('.npy'):
                # Direct temperature data
                thermal_data = np.load(thermal_path)
            else:
                # Convert from image format
                thermal_img = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
                thermal_data = self.convert_thermal_image(thermal_img)
            
            # Resize to model input size
            thermal_data = cv2.resize(thermal_data, (224, 224))
            images.append(thermal_data)
            
            # Get labels for this sample
            sample_labels = self.get_labels_for_sample(plant_id, day, timestamp)
            
            labels['water_stress'].append(sample_labels['water_stress'])
            labels['disease'].append(sample_labels['disease'])
            labels['nutrients'].append(sample_labels['nutrients'])
            labels['segmentation'].append(sample_labels['segmentation'])
            labels['plant_id'].append(plant_id)
            labels['day'].append(day)
            labels['timestamp'].append(timestamp)
        
        # Convert to numpy arrays
        images = np.array(images)
        for key in labels:
            if key not in ['plant_id', 'timestamp']:
                labels[key] = np.array(labels[key])
        
        return images, labels
    
    def convert_thermal_image(self, img: np.ndarray) -> np.ndarray:
        """Convert thermal image to temperature values"""
        if img.dtype == np.uint16:
            # 16-bit thermal image - common format
            # Assuming linear mapping from pixel values to temperature
            # Typical range: 0-65535 -> -20°C to 120°C
            temp_data = (img.astype(np.float32) / 65535.0) * 140.0 - 20.0
        elif img.dtype == np.uint8:
            # 8-bit thermal image
            # Typical range: 0-255 -> 0°C to 50°C
            temp_data = (img.astype(np.float32) / 255.0) * 50.0
        else:
            # Already float temperature data
            temp_data = img.astype(np.float32)
            
        return temp_data
    
    def get_labels_for_sample(self, plant_id: int, day: int, timestamp: str) -> Dict:
        """Get or generate labels for a specific sample"""
        
        # Default healthy plant labels
        labels = {
            'water_stress': 0.0,
            'disease': [1.0, 0.0, 0.0, 0.0],  # [healthy, bacterial, fungal, viral]
            'nutrients': [0.5, 0.5, 0.5],  # [N, P, K] - optimal
            'segmentation': np.zeros((224, 224), dtype=np.float32)
        }
        
        # Check if we have manual labels
        if self.labels:
            key = f"plant{plant_id}_day{day}"
            if key in self.labels:
                sample_label = self.labels[key]
                
                # Water stress (based on irrigation schedule)
                if 'water_stress' in sample_label:
                    labels['water_stress'] = sample_label['water_stress']
                elif 'last_watered_hours' in sample_label:
                    # Estimate water stress from time since watering
                    hours = sample_label['last_watered_hours']
                    labels['water_stress'] = min(1.0, hours / 72.0)  # Max stress at 72 hours
                
                # Disease status
                if 'disease' in sample_label:
                    disease_type = sample_label['disease']
                    if disease_type != 'healthy':
                        labels['disease'] = [0.0, 0.0, 0.0, 0.0]
                        idx = ['healthy', 'bacterial', 'fungal', 'viral'].index(disease_type)
                        labels['disease'][idx] = 1.0
                
                # Nutrient status
                if 'nutrients' in sample_label:
                    labels['nutrients'] = [
                        sample_label['nutrients'].get('nitrogen', 0.5),
                        sample_label['nutrients'].get('phosphorus', 0.5),
                        sample_label['nutrients'].get('potassium', 0.5)
                    ]
        
        # Synthetic labels based on growth stage
        else:
            # Early stress signs typically appear after day 30
            if day > 30:
                # Simulate gradual water stress
                labels['water_stress'] = min(0.8, (day - 30) / 36.0)
                
                # Simulate nutrient depletion
                labels['nutrients'][0] = max(0.1, 0.5 - (day - 30) / 60.0)  # N depletion
                
            # Disease simulation (random, low probability)
            if np.random.random() < 0.05 and day > 20:
                disease_type = np.random.choice(['bacterial', 'fungal', 'viral'])
                labels['disease'] = [0.0, 0.0, 0.0, 0.0]
                idx = ['healthy', 'bacterial', 'fungal', 'viral'].index(disease_type)
                labels['disease'][idx] = 1.0
        
        return labels

class ThermalAugmentation:
    """Data augmentation specifically for thermal images"""
    
    @staticmethod
    def add_thermal_noise(img: np.ndarray, noise_level: float = 0.5) -> np.ndarray:
        """Add realistic thermal sensor noise"""
        noise = np.random.normal(0, noise_level, img.shape)
        return img + noise
    
    @staticmethod
    def simulate_atmospheric_effects(img: np.ndarray, humidity: float = 0.5) -> np.ndarray:
        """Simulate atmospheric absorption effects"""
        # Higher humidity reduces apparent temperature
        attenuation = 1.0 - (humidity * 0.1)
        return img * attenuation
    
    @staticmethod
    def add_dead_pixels(img: np.ndarray, num_pixels: int = 5) -> np.ndarray:
        """Simulate dead pixels in thermal sensor"""
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        for _ in range(num_pixels):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            # Dead pixels often show as very hot or very cold
            img_copy[y, x] = np.random.choice([img.min(), img.max()])
            
        return img_copy
    
    @staticmethod
    def apply_vignetting(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """Apply lens vignetting effect"""
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * strength
        
        return img * vignette
    
    @staticmethod
    def simulate_emissivity_variation(img: np.ndarray, variation: float = 0.05) -> np.ndarray:
        """Simulate variations in surface emissivity"""
        # Different parts of the plant have slightly different emissivity
        emissivity_map = np.random.uniform(1 - variation, 1 + variation, img.shape)
        return img * emissivity_map

def create_training_dataset(
    base_path: str,
    augment: bool = True,
    train_split: float = 0.8
) -> Tuple[Tuple[np.ndarray, Dict], Tuple[np.ndarray, Dict]]:
    """Create training and validation datasets"""
    
    # Load base dataset
    loader = HydroponicDatasetLoader(base_path)
    images, labels = loader.load_dataset()
    
    if len(images) == 0:
        print("No real dataset found. Using synthetic data generator...")
        from train_plant_health_model import ThermalDataGenerator, create_dataset
        generator = ThermalDataGenerator()
        return create_dataset(generator, 8000), create_dataset(generator, 2000)
    
    # Augmentation
    if augment:
        augmentor = ThermalAugmentation()
        augmented_images = []
        augmented_labels = {k: [] for k in labels.keys()}
        
        for i in range(len(images)):
            # Original image
            augmented_images.append(images[i])
            for k in labels.keys():
                if k in ['water_stress', 'disease', 'nutrients', 'segmentation']:
                    augmented_labels[k].append(labels[k][i])
            
            # Apply augmentations
            for _ in range(3):  # 3 augmented versions per image
                aug_img = images[i].copy()
                
                # Random augmentations
                if np.random.random() > 0.5:
                    aug_img = augmentor.add_thermal_noise(aug_img)
                if np.random.random() > 0.5:
                    aug_img = augmentor.simulate_atmospheric_effects(
                        aug_img, humidity=np.random.uniform(0.3, 0.8)
                    )
                if np.random.random() > 0.8:
                    aug_img = augmentor.add_dead_pixels(aug_img, num_pixels=np.random.randint(1, 5))
                if np.random.random() > 0.5:
                    aug_img = augmentor.apply_vignetting(aug_img)
                if np.random.random() > 0.5:
                    aug_img = augmentor.simulate_emissivity_variation(aug_img)
                
                augmented_images.append(aug_img)
                for k in labels.keys():
                    if k in ['water_stress', 'disease', 'nutrients', 'segmentation']:
                        augmented_labels[k].append(labels[k][i])
        
        images = np.array(augmented_images)
        for k in augmented_labels.keys():
            labels[k] = np.array(augmented_labels[k])
    
    # Split into train/validation
    num_samples = len(images)
    num_train = int(num_samples * train_split)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create train set
    train_images = images[train_indices]
    train_labels = {}
    for k in ['water_stress', 'disease', 'nutrients', 'segmentation']:
        train_labels[k] = labels[k][train_indices]
    
    # Create validation set
    val_images = images[val_indices]
    val_labels = {}
    for k in ['water_stress', 'disease', 'nutrients', 'segmentation']:
        val_labels[k] = labels[k][val_indices]
    
    print(f"Dataset created: {len(train_images)} training, {len(val_images)} validation samples")
    
    return (train_images, train_labels), (val_images, val_labels)

if __name__ == "__main__":
    # Example usage
    dataset_path = "./hydroponic_dataset"
    train_data, val_data = create_training_dataset(dataset_path, augment=True)
    
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    print(f"Image shape: {train_data[0][0].shape}")
    print(f"Temperature range: {train_data[0].min():.1f}°C - {train_data[0].max():.1f}°C")