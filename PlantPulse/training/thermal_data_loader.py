"""
Real thermal dataset loader for PlantPulse
Supports ETH Zurich, Date Palm, and other thermal datasets
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import glob
from PIL import Image
import re

class ThermalDatasetLoader:
    """Load and preprocess real thermal images for plant health analysis"""
    
    def __init__(self, dataset_path: str, dataset_type: str = "eth_zurich"):
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type
        self.img_size = 224
        
        # Dataset-specific configurations
        self.configs = {
            "eth_zurich": {
                "pattern": "**/*thermal*.png",
                "label_mapping": {
                    "optimal": 0,
                    "drought": 1,
                    "low_nitrogen": 2,
                    "surplus_nitrogen": 3,
                    "weed_pressure": 4
                },
                "temperature_range": (15, 35)  # Celsius
            },
            "date_palm": {
                "pattern": "**/thermal/*.jpg",
                "label_mapping": {
                    "non_infected": 0,
                    "infected": 1,
                    "badly_damaged": 2,
                    "dead": 3
                },
                "temperature_range": (20, 45)  # Higher for desert conditions
            },
            "generic_thermal": {
                "pattern": "**/*.tif*",
                "label_mapping": {
                    "healthy": 0,
                    "stressed": 1,
                    "diseased": 2,
                    "unknown": 3
                },
                "temperature_range": (10, 40)
            },
            "combined": {
                "pattern": "**/*.png",
                "label_mapping": {
                    # Healthy conditions
                    "healthy": 0,
                    
                    # Water stress levels
                    "water_stressed": 1,
                    "water_stressed_mild": 1,
                    "water_stressed_severe": 2,
                    
                    # Disease types
                    "diseased": 3,
                    "diseased_bacterial": 3,
                    "diseased_fungal": 4,
                    "diseased_viral": 5,
                    "diseased_severe": 6,
                    
                    # Nutrient issues
                    "nutrient_deficient": 7,
                    "nutrient_deficient_n": 7,
                    "nutrient_excess_n": 8,
                    
                    # Other stresses
                    "competition_stress": 9,
                    "pest_damage": 10,
                    "dead": 11,
                    "unknown": 12
                },
                "temperature_range": (10, 45)  # Wide range for combined data
            }
        }
        
        self.config = self.configs.get(dataset_type, self.configs["generic_thermal"])
        
    def load_thermal_image(self, image_path: str) -> np.ndarray:
        """Load thermal image and convert to temperature array"""
        
        # Try different loading methods based on format
        if image_path.endswith(('.tif', '.tiff')):
            # TIFF files often contain raw thermal data
            img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                img = np.array(Image.open(image_path))
        else:
            # Standard image formats
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.array(Image.open(image_path).convert('L'))
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to temperature values
        # Assumption: pixel values map linearly to temperature range
        temp_min, temp_max = self.config["temperature_range"]
        
        if img.dtype == np.uint16:
            # 16-bit thermal images
            temp_array = (img / 65535.0) * (temp_max - temp_min) + temp_min
        else:
            # 8-bit images
            temp_array = (img / 255.0) * (temp_max - temp_min) + temp_min
        
        # Resize to model input size
        temp_array = cv2.resize(temp_array, (self.img_size, self.img_size))
        
        return temp_array.astype(np.float32)
    
    def extract_label_from_path(self, file_path: str) -> int:
        """Extract label from file path based on dataset structure"""
        path_lower = file_path.lower()
        
        # Check each label in the mapping
        for label_name, label_id in self.config["label_mapping"].items():
            if label_name in path_lower:
                return label_id
        
        # Default to unknown/healthy if no match
        return 0
    
    def create_multi_task_labels(self, primary_label: int) -> Dict[str, np.ndarray]:
        """Convert primary label to multi-task labels"""
        
        # Determine number of disease classes based on dataset type
        num_disease_classes = 4
        if self.dataset_type == "combined":
            num_disease_classes = 13  # More classes for combined dataset
        
        # Initialize default values
        labels = {
            'water_stress': 0.0,
            'disease': np.zeros(num_disease_classes),
            'nutrients': np.array([0.7, 0.7, 0.7]),  # Default optimal
            'segmentation': None  # Will be created from thermal patterns
        }
        
        # Map based on dataset type
        if self.dataset_type == "eth_zurich":
            if primary_label == 0:  # optimal
                labels['water_stress'] = 0.1
                labels['disease'][0] = 1.0  # healthy
            elif primary_label == 1:  # drought
                labels['water_stress'] = 0.8
                labels['disease'][0] = 1.0
            elif primary_label == 2:  # low_nitrogen
                labels['water_stress'] = 0.3
                labels['disease'][0] = 1.0
                labels['nutrients'][0] = 0.2  # N deficient
            elif primary_label == 3:  # surplus_nitrogen
                labels['water_stress'] = 0.2
                labels['disease'][0] = 1.0
                labels['nutrients'][0] = 0.9  # N excess
            elif primary_label == 4:  # weed_pressure
                labels['water_stress'] = 0.4
                labels['disease'][0] = 1.0
                
        elif self.dataset_type == "date_palm":
            if primary_label == 0:  # non_infected
                labels['water_stress'] = 0.2
                labels['disease'][0] = 1.0
            elif primary_label == 1:  # infected
                labels['water_stress'] = 0.4
                labels['disease'][2] = 1.0  # fungal
            elif primary_label == 2:  # badly_damaged
                labels['water_stress'] = 0.6
                labels['disease'][1] = 1.0  # bacterial
            elif primary_label == 3:  # dead
                labels['water_stress'] = 0.9
                labels['disease'][3] = 1.0  # viral
        
        elif self.dataset_type == "combined":
            # For combined dataset, map to the extended label set
            if primary_label < num_disease_classes:
                labels['disease'][primary_label] = 1.0
            else:
                labels['disease'][0] = 1.0  # Default to healthy
            
            # Set water stress based on label
            if primary_label in [1, 2]:  # water stressed
                labels['water_stress'] = 0.6
            elif primary_label in [3, 4, 5, 6]:  # diseased
                labels['water_stress'] = 0.4
            elif primary_label in [7, 8]:  # nutrient issues
                labels['water_stress'] = 0.3
                labels['nutrients'][0] = 0.3 if primary_label == 7 else 0.9
            elif primary_label == 10:  # pest damage
                labels['water_stress'] = 0.5
            elif primary_label == 11:  # dead
                labels['water_stress'] = 0.9
        
        else:  # generic_thermal and others
            # Default mapping for 4 classes
            if primary_label < 4:
                labels['disease'][primary_label] = 1.0
            else:
                labels['disease'][0] = 1.0  # Default to healthy
            
            # Simple water stress mapping
            labels['water_stress'] = primary_label * 0.2
        
        return labels
    
    def create_segmentation_mask(self, thermal_image: np.ndarray) -> np.ndarray:
        """Create segmentation mask from thermal patterns"""
        # Simple threshold-based segmentation
        # Plants are typically cooler than background
        mean_temp = np.mean(thermal_image)
        std_temp = np.std(thermal_image)
        
        # Pixels significantly cooler than mean are likely plants
        plant_mask = thermal_image < (mean_temp - 0.5 * std_temp)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask.astype(np.uint8), 
                                     cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        
        return plant_mask.astype(np.float32)
    
    def load_dataset(self, split: float = 0.8) -> Tuple[Dict, Dict]:
        """Load full dataset with train/validation split"""
        
        print(f"Loading {self.dataset_type} dataset from: {self.dataset_path}")
        
        # Find all thermal images
        pattern = self.config["pattern"]
        image_files = list(self.dataset_path.glob(pattern))
        
        if not image_files:
            # Try alternative patterns
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                image_files.extend(self.dataset_path.glob(f"**/{ext}"))
        
        print(f"Found {len(image_files)} thermal images")
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {self.dataset_path}")
        
        # Load and process images
        images = []
        all_labels = {
            'water_stress': [],
            'disease': [],
            'nutrients': [],
            'segmentation': []
        }
        
        for i, img_path in enumerate(image_files):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(image_files)}")
            
            try:
                # Load thermal image
                thermal_img = self.load_thermal_image(str(img_path))
                
                # Extract label
                primary_label = self.extract_label_from_path(str(img_path))
                
                # Create multi-task labels
                labels = self.create_multi_task_labels(primary_label)
                
                # Create segmentation mask
                seg_mask = self.create_segmentation_mask(thermal_img)
                labels['segmentation'] = seg_mask
                
                # Normalize thermal image
                thermal_norm = (thermal_img - 15) / 25  # Normalize to ~[0, 1]
                thermal_norm = np.expand_dims(thermal_norm, -1)
                
                images.append(thermal_norm)
                for key in all_labels:
                    all_labels[key].append(labels[key])
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Convert to arrays
        images = np.array(images)
        for key in all_labels:
            all_labels[key] = np.array(all_labels[key])
        
        # Split into train/validation
        n_samples = len(images)
        n_train = int(n_samples * split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_data = (
            images[train_idx],
            {k: v[train_idx] for k, v in all_labels.items()}
        )
        
        val_data = (
            images[val_idx],
            {k: v[val_idx] for k, v in all_labels.items()}
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Temperature range: [{np.min(images):.2f}, {np.max(images):.2f}]")
        
        return train_data, val_data
    
    def create_tf_dataset(self, images: np.ndarray, labels: Dict, 
                         batch_size: int = 32, augment: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset with optional augmentation"""
        
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((
            images,
            {
                'water_stress': labels['water_stress'],
                'disease': labels['disease'],
                'nutrients': labels['nutrients'],
                'segmentation': labels['segmentation']
            }
        ))
        
        # Add augmentation if training
        if augment:
            def augment_image(image, labels):
                # Random rotation
                image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
                
                # Random flip
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                
                # Random brightness (temperature variation)
                image = tf.image.random_brightness(image, 0.1)
                
                # Random contrast
                image = tf.image.random_contrast(image, 0.8, 1.2)
                
                # Ensure values stay in valid range
                image = tf.clip_by_value(image, 0.0, 1.0)
                
                return image, labels
            
            dataset = dataset.map(augment_image, 
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def test_loader():
    """Test the thermal data loader"""
    
    # Test with sample data structure
    print("Testing Thermal Data Loader")
    print("=" * 50)
    
    # Create sample directory structure
    test_dir = Path("data/test_thermal")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample thermal images
    for condition in ["optimal", "drought", "diseased"]:
        condition_dir = test_dir / condition
        condition_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            # Create synthetic thermal image
            if condition == "optimal":
                temp_pattern = np.random.normal(22, 1, (224, 224))
            elif condition == "drought":
                temp_pattern = np.random.normal(28, 2, (224, 224))
            else:
                temp_pattern = np.random.normal(25, 3, (224, 224))
            
            # Add some structure
            y, x = np.ogrid[-112:112, -112:112]
            mask = x*x + y*y <= 80*80
            temp_pattern[mask] -= 3  # Cooler plant area
            
            # Convert to 8-bit image
            img = ((temp_pattern - 15) / 25 * 255).clip(0, 255).astype(np.uint8)
            
            cv2.imwrite(str(condition_dir / f"thermal_{i:03d}.png"), img)
    
    # Test loader
    loader = ThermalDatasetLoader(str(test_dir), dataset_type="generic_thermal")
    
    try:
        train_data, val_data = loader.load_dataset(split=0.8)
        print("\n✅ Loader test successful!")
        
        # Create TF dataset
        train_ds = loader.create_tf_dataset(
            train_data[0], train_data[1], 
            batch_size=4, augment=True
        )
        
        print("\nSample batch:")
        for batch_images, batch_labels in train_ds.take(1):
            print(f"Images shape: {batch_images.shape}")
            print(f"Disease labels shape: {batch_labels['disease'].shape}")
            
    except Exception as e:
        print(f"\n❌ Loader test failed: {e}")

if __name__ == "__main__":
    test_loader()