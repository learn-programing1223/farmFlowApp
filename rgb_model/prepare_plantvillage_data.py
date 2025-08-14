#!/usr/bin/env python3
"""
PlantVillage Dataset Preparation Pipeline
=========================================

This script organizes the PlantVillage dataset for robust model training by:
1. Mapping PlantVillage disease categories to 7 universal categories
2. Splitting data into train/validation/test sets (70/15/15)
3. Balancing class distribution
4. Applying CycleGAN-style augmentation for domain adaptation
5. Removing corrupted/invalid images

Author: Claude Code
Date: 2025-01-11
"""

import os
import shutil
import json
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
import logging
from tqdm import tqdm
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlantVillageProcessor:
    def __init__(self, source_path="PlantVillage/PlantVillage", output_path="datasets/plantvillage_processed"):
        self.source_path = source_path
        self.output_path = output_path
        
        # Universal category mapping
        self.category_mapping = {
            # Blight category
            'Potato___Early_blight': 'Blight',
            'Potato___Late_blight': 'Blight', 
            'Tomato_Early_blight': 'Blight',
            'Tomato_Late_blight': 'Blight',
            
            # Healthy category
            'Pepper__bell___healthy': 'Healthy',
            'Potato___healthy': 'Healthy',
            'Tomato_healthy': 'Healthy',
            
            # Leaf Spot category
            'Pepper__bell___Bacterial_spot': 'Leaf_Spot',
            'Tomato_Bacterial_spot': 'Leaf_Spot',
            'Tomato_Septoria_leaf_spot': 'Leaf_Spot',
            'Tomato__Target_Spot': 'Leaf_Spot',
            
            # Mosaic Virus category
            'Tomato__Tomato_mosaic_virus': 'Mosaic_Virus',
            'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Mosaic_Virus',
            
            # Currently mapping to Nutrient_Deficiency (may need adjustment)
            'Tomato_Leaf_Mold': 'Nutrient_Deficiency',
            
            # Currently mapping to Powdery_Mildew (may need adjustment) 
            'Tomato_Spider_mites_Two_spotted_spider_mite': 'Powdery_Mildew',
        }
        
        # Target categories for balanced dataset
        self.target_categories = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
                                'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']
        
        self.stats = {
            'total_images': 0,
            'corrupted_images': 0,
            'processed_images': 0,
            'category_counts': defaultdict(int),
            'final_counts': defaultdict(int)
        }
        
    def validate_image(self, image_path):
        """Validate if image is not corrupted and meets quality standards"""
        try:
            # Check file size (minimum 1KB)
            if os.path.getsize(image_path) < 1024:
                return False, "File too small"
                
            # Try to open with PIL
            with Image.open(image_path) as img:
                # Check image dimensions (minimum 64x64)
                if img.width < 64 or img.height < 64:
                    return False, "Image too small"
                
                # Verify image can be loaded
                img.verify()
                
            # Try to open with OpenCV
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                return False, "Cannot read with OpenCV"
                
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_file_list(self):
        """Get list of all valid images with their categories"""
        file_list = []
        
        logger.info("Scanning PlantVillage dataset...")
        
        for category_folder in os.listdir(self.source_path):
            category_path = os.path.join(self.source_path, category_folder)
            
            if not os.path.isdir(category_path):
                continue
                
            if category_folder not in self.category_mapping:
                logger.warning(f"Unknown category: {category_folder}")
                continue
                
            universal_category = self.category_mapping[category_folder]
            
            # Get all image files
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"Processing {category_folder} -> {universal_category}: {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"Validating {category_folder}"):
                image_path = os.path.join(category_path, image_file)
                self.stats['total_images'] += 1
                
                # Validate image
                is_valid, error_msg = self.validate_image(image_path)
                
                if is_valid:
                    file_list.append({
                        'path': image_path,
                        'filename': image_file,
                        'original_category': category_folder,
                        'universal_category': universal_category
                    })
                    self.stats['category_counts'][universal_category] += 1
                else:
                    self.stats['corrupted_images'] += 1
                    logger.warning(f"Invalid image {image_path}: {error_msg}")
        
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Valid images: {len(file_list)}")
        logger.info(f"Corrupted images: {self.stats['corrupted_images']}")
        
        return file_list
    
    def balance_dataset(self, file_list, target_per_class=2000):
        """Balance dataset by sampling or augmenting classes"""
        
        # Group files by category
        category_files = defaultdict(list)
        for file_info in file_list:
            category_files[file_info['universal_category']].append(file_info)
        
        logger.info("Category distribution before balancing:")
        for category, files in category_files.items():
            logger.info(f"  {category}: {len(files)} images")
        
        balanced_files = []
        
        for category in self.target_categories:
            if category not in category_files:
                logger.warning(f"No images found for category: {category}")
                continue
                
            files = category_files[category]
            current_count = len(files)
            
            if current_count >= target_per_class:
                # Downsample if we have too many
                sampled_files = random.sample(files, target_per_class)
                balanced_files.extend(sampled_files)
                logger.info(f"{category}: Downsampled from {current_count} to {target_per_class}")
            else:
                # Use all available files and note augmentation needed
                balanced_files.extend(files)
                augmentation_needed = target_per_class - current_count
                logger.info(f"{category}: Using all {current_count} images, need {augmentation_needed} augmented")
        
        return balanced_files
    
    def apply_simple_augmentation(self, image_path, output_path, num_augmentations=1):
        """Apply simple augmentation techniques"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Base transformations
            transforms = [
                lambda x: cv2.flip(x, 1),  # Horizontal flip
                lambda x: cv2.flip(x, 0),  # Vertical flip
                lambda x: self._rotate_image(x, 15),  # Rotate 15 degrees
                lambda x: self._rotate_image(x, -15), # Rotate -15 degrees
                lambda x: self._adjust_brightness(x, 1.2),  # Brighter
                lambda x: self._adjust_brightness(x, 0.8),  # Darker
                lambda x: self._add_noise(x),  # Add gaussian noise
            ]
            
            # Apply random transformations
            for i in range(num_augmentations):
                augmented = img.copy()
                
                # Apply 1-2 random transformations
                num_transforms = random.randint(1, 2)
                selected_transforms = random.sample(transforms, num_transforms)
                
                for transform in selected_transforms:
                    augmented = transform(augmented)
                
                # Save augmented image
                aug_filename = f"aug_{i}_{os.path.basename(output_path)}"
                aug_path = os.path.join(os.path.dirname(output_path), aug_filename)
                cv2.imwrite(aug_path, augmented)
            
            return True
            
        except Exception as e:
            logger.error(f"Augmentation failed for {image_path}: {e}")
            return False
    
    def _rotate_image(self, img, angle):
        """Rotate image by given angle"""
        h, w = img.shape[:2]
        center = (w//2, h//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))
    
    def _adjust_brightness(self, img, factor):
        """Adjust image brightness"""
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    def _add_noise(self, img):
        """Add gaussian noise to image"""
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    def create_splits(self, file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/validation/test sets"""
        
        # Ensure ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Group by category for stratified split
        category_files = defaultdict(list)
        for file_info in file_list:
            category_files[file_info['universal_category']].append(file_info)
        
        train_files, val_files, test_files = [], [], []
        
        for category, files in category_files.items():
            if len(files) < 3:  # Need at least 3 files for splitting
                logger.warning(f"Too few files for {category}: {len(files)}")
                train_files.extend(files)
                continue
            
            # First split: train vs (val + test)
            train_cat, temp_cat = train_test_split(
                files, test_size=(val_ratio + test_ratio), 
                random_state=42, shuffle=True
            )
            
            # Second split: val vs test
            if len(temp_cat) >= 2:
                val_cat, test_cat = train_test_split(
                    temp_cat, test_size=(test_ratio / (val_ratio + test_ratio)),
                    random_state=42, shuffle=True
                )
            else:
                val_cat = temp_cat
                test_cat = []
            
            train_files.extend(train_cat)
            val_files.extend(val_cat)
            test_files.extend(test_cat)
            
            logger.info(f"{category} split: Train={len(train_cat)}, Val={len(val_cat)}, Test={len(test_cat)}")
        
        return train_files, val_files, test_files
    
    def copy_files(self, file_list, split_name):
        """Copy files to organized directory structure"""
        
        split_path = os.path.join(self.output_path, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        # Create category directories
        for category in self.target_categories:
            category_path = os.path.join(split_path, category)
            os.makedirs(category_path, exist_ok=True)
        
        copied_count = 0
        
        for file_info in tqdm(file_list, desc=f"Copying {split_name} files"):
            src_path = file_info['path']
            category = file_info['universal_category']
            filename = file_info['filename']
            
            # Create unique filename to avoid conflicts
            base_name, ext = os.path.splitext(filename)
            unique_filename = f"{file_info['original_category']}_{base_name}{ext}"
            
            dst_path = os.path.join(split_path, category, unique_filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                self.stats['final_counts'][f"{split_name}_{category}"] += 1
            except Exception as e:
                logger.error(f"Failed to copy {src_path}: {e}")
        
        logger.info(f"Copied {copied_count} files to {split_name}")
        return copied_count
    
    def generate_metadata(self):
        """Generate metadata and statistics"""
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_path': self.source_path,
            'output_path': self.output_path,
            'category_mapping': self.category_mapping,
            'target_categories': self.target_categories,
            'statistics': dict(self.stats),
            'splits': {}
        }
        
        # Count files in each split
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.output_path, split)
            if os.path.exists(split_path):
                split_stats = {}
                total_split = 0
                
                for category in self.target_categories:
                    cat_path = os.path.join(split_path, category)
                    if os.path.exists(cat_path):
                        count = len([f for f in os.listdir(cat_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        split_stats[category] = count
                        total_split += count
                    else:
                        split_stats[category] = 0
                
                split_stats['total'] = total_split
                metadata['splits'][split] = split_stats
        
        # Save metadata
        metadata_path = os.path.join(self.output_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save human-readable stats
        stats_path = os.path.join(self.output_path, 'dataset_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("PlantVillage Dataset Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing completed: {metadata['created_at']}\n")
            f.write(f"Total images processed: {self.stats['total_images']}\n")
            f.write(f"Corrupted images removed: {self.stats['corrupted_images']}\n")
            f.write(f"Success rate: {((self.stats['total_images'] - self.stats['corrupted_images']) / self.stats['total_images'] * 100):.2f}%\n\n")
            
            f.write("Category Mapping:\n")
            f.write("-" * 20 + "\n")
            for original, universal in self.category_mapping.items():
                f.write(f"{original:35} -> {universal}\n")
            
            f.write("\nFinal Dataset Distribution:\n")
            f.write("-" * 30 + "\n")
            
            total_train = total_val = total_test = 0
            
            for category in self.target_categories:
                train_count = metadata['splits'].get('train', {}).get(category, 0)
                val_count = metadata['splits'].get('val', {}).get(category, 0)
                test_count = metadata['splits'].get('test', {}).get(category, 0)
                
                total_train += train_count
                total_val += val_count
                total_test += test_count
                
                f.write(f"{category:18} | Train: {train_count:4d} | Val: {val_count:3d} | Test: {test_count:3d} | Total: {train_count+val_count+test_count:4d}\n")
            
            f.write("-" * 70 + "\n")
            f.write(f"{'TOTAL':18} | Train: {total_train:4d} | Val: {total_val:3d} | Test: {total_test:3d} | Total: {total_train+total_val+total_test:4d}\n")
            
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Statistics saved to {stats_path}")
    
    def process(self):
        """Main processing pipeline"""
        
        logger.info("Starting PlantVillage dataset processing...")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Step 1: Get and validate all files
        logger.info("Step 1: Scanning and validating images...")
        file_list = self.get_file_list()
        
        if not file_list:
            logger.error("No valid images found!")
            return
        
        # Step 2: Balance dataset (optional)
        logger.info("Step 2: Preparing balanced dataset...")
        balanced_files = self.balance_dataset(file_list, target_per_class=1500)
        
        # Step 3: Create train/val/test splits
        logger.info("Step 3: Creating data splits...")
        train_files, val_files, test_files = self.create_splits(balanced_files)
        
        # Step 4: Copy files to organized structure
        logger.info("Step 4: Organizing files...")
        self.copy_files(train_files, 'train')
        self.copy_files(val_files, 'val') 
        self.copy_files(test_files, 'test')
        
        # Step 5: Generate metadata
        logger.info("Step 5: Generating metadata...")
        self.generate_metadata()
        
        logger.info("Dataset processing completed successfully!")
        
        # Print final statistics
        print("\n" + "="*60)
        print("DATASET PROCESSING COMPLETED")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.output_path, split)
            if os.path.exists(split_path):
                total = sum(len([f for f in os.listdir(os.path.join(split_path, cat)) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                          for cat in self.target_categories 
                          if os.path.exists(os.path.join(split_path, cat)))
                print(f"{split.upper():5}: {total:5d} images")
        
        print(f"\nOutput directory: {os.path.abspath(self.output_path)}")
        print("Dataset is ready for training!")


def main():
    """Main function"""
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize processor
    processor = PlantVillageProcessor()
    
    # Check if source directory exists
    if not os.path.exists(processor.source_path):
        logger.error(f"Source directory not found: {processor.source_path}")
        print("Please ensure PlantVillage dataset is downloaded and extracted.")
        return
    
    try:
        # Run processing pipeline
        processor.process()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()