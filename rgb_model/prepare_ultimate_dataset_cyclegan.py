#!/usr/bin/env python3
"""
ULTIMATE DATASET PREPARATION WITH CYCLEGAN AUGMENTATION
Combines ultimate dataset preparation with realistic field-style augmentation
Creates both clean and augmented versions for robust training
"""

import os
import shutil
import numpy as np
from pathlib import Path
import random
from PIL import Image
import json
from datetime import datetime
import cv2
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from cyclegan_augmentor import FieldEffectsAugmentor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMPREHENSIVE MAPPING - Every possible disease from Plant Disease dataset
ULTIMATE_MAPPING = {
    # ============= PLANTVILLAGE MAPPINGS (Existing) =============
    'Potato___Early_blight': 'Blight',
    'Potato___Late_blight': 'Blight',
    'Tomato_Early_blight': 'Blight',
    'Tomato_Late_blight': 'Blight',
    'Tomato__Target_Spot': 'Blight',
    
    'Pepper__bell___healthy': 'Healthy',
    'Potato___healthy': 'Healthy',
    'Tomato_healthy': 'Healthy',
    
    'Pepper__bell___Bacterial_spot': 'Leaf_Spot',
    'Tomato_Bacterial_spot': 'Leaf_Spot',
    'Tomato_Septoria_leaf_spot': 'Leaf_Spot',
    
    'Tomato__Tomato_mosaic_virus': 'Mosaic_Virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Mosaic_Virus',
    
    'Tomato_Leaf_Mold': 'Nutrient_Deficiency',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Powdery_Mildew',
    
    # ============= PLANT DISEASE DATASET - COMPREHENSIVE MAPPINGS =============
    
    # APPLE DISEASES
    'Apple___Apple_scab': 'Leaf_Spot',
    'Apple___Black_rot': 'Blight',
    'Apple___Cedar_apple_rust': 'Rust',
    'Apple___healthy': 'Healthy',
    
    # GRAPE DISEASES
    'Grape___Black_rot': 'Blight',
    'Grape___Esca_(Black_Measles)': 'Blight',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Blight',
    'Grape___healthy': 'Healthy',
    
    # CORN/MAIZE DISEASES
    'Corn_(maize)___Common_rust_': 'Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Blight',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Leaf_Spot',
    'Corn_(maize)___healthy': 'Healthy',
    
    # STRAWBERRY DISEASES
    'Strawberry___Leaf_scorch': 'Leaf_Spot',
    'Strawberry___healthy': 'Healthy',
    
    # PEACH DISEASES
    'Peach___Bacterial_spot': 'Leaf_Spot',
    'Peach___healthy': 'Healthy',
    
    # CHERRY DISEASES
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery_Mildew',
    'Cherry_(including_sour)___healthy': 'Healthy',
    
    # OTHER CROPS
    'Squash___Powdery_mildew': 'Powdery_Mildew',
    'Soybean___healthy': 'Healthy',
    'Orange___Haunglongbing_(Citrus_greening)': 'Nutrient_Deficiency',
    'Raspberry___healthy': 'Healthy',
    'Blueberry___healthy': 'Healthy',
    
    # Alternative naming conventions
    'Corn___Common_rust': 'Rust',
    'Corn___Northern_Leaf_Blight': 'Blight',
    'Corn___Gray_leaf_spot': 'Leaf_Spot',
    'Corn___healthy': 'Healthy',
    
    'Apple_scab': 'Leaf_Spot',
    'Apple_Black_rot': 'Blight',
    'Apple_rust': 'Rust',
    'Apple_healthy': 'Healthy',
}

# Universal categories
TARGET_CATEGORIES = [
    'Blight',
    'Healthy', 
    'Leaf_Spot',
    'Mosaic_Virus',
    'Nutrient_Deficiency',
    'Powdery_Mildew',
    'Rust'
]

class UltimateDatasetPreparator:
    """Prepares ultimate dataset with CycleGAN-style augmentation"""
    
    def __init__(self, 
                 output_path: str = 'datasets/ultimate_plant_disease_cyclegan',
                 augmentation_ratio: float = 0.3,
                 severity: float = 0.7,
                 num_workers: int = None):
        """
        Initialize dataset preparator
        
        Args:
            output_path: Where to save the prepared dataset
            augmentation_ratio: Fraction of dataset to augment with CycleGAN effects
            severity: Intensity of augmentation effects (0.0 - 1.0)
            num_workers: Number of parallel workers (None = auto-detect)
        """
        self.output_path = Path(output_path)
        self.augmentation_ratio = augmentation_ratio
        self.severity = severity
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())
        
        # Initialize augmentor
        self.augmentor = FieldEffectsAugmentor(
            severity=severity,
            enable_caching=True,
            debug=False
        )
        
        logger.info(f"UltimateDatasetPreparator initialized:")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Augmentation ratio: {augmentation_ratio:.1%}")
        logger.info(f"  Severity: {severity:.2f}")
        logger.info(f"  Workers: {num_workers}")
    
    def find_all_datasets(self) -> list:
        """Find all available datasets automatically"""
        found_datasets = []
        
        # Check common locations
        search_paths = [
            Path('.'),
            Path('datasets'),
            Path('../datasets'),
        ]
        
        for base_path in search_paths:
            # PlantVillage
            pv_path = base_path / 'PlantVillage' / 'PlantVillage'
            if pv_path.exists():
                found_datasets.append(pv_path)
                logger.info(f"[OK] Found PlantVillage: {pv_path}")
            
            # Plant Disease Dataset variations
            pd_variations = [
                'PlantDisease',
                'PlantDisease/dataset',
                'plant-disease-dataset'
            ]
            
            for variation in pd_variations:
                pd_path = base_path / variation
                if pd_path.exists():
                    found_datasets.append(pd_path)
                    logger.info(f"[OK] Found PlantDisease: {pd_path}")
                    
                    # Check for train/test structure
                    if (pd_path / 'train').exists():
                        found_datasets.append(pd_path / 'train')
                        logger.info(f"[OK] Found training split: {pd_path / 'train'}")
                    
                    if (pd_path / 'valid').exists():
                        found_datasets.append(pd_path / 'valid')
                        logger.info(f"[OK] Found validation split: {pd_path / 'valid'}")
        
        # Remove duplicates
        found_datasets = list(set(found_datasets))
        
        if not found_datasets:
            logger.warning("No datasets found! Searching in current directory...")
            # Last resort: search current directory for any plant-related folders
            current_dir = Path('.')
            for item in current_dir.iterdir():
                if item.is_dir() and any(keyword in item.name.lower() 
                                       for keyword in ['plant', 'disease', 'village', 'crop']):
                    logger.info(f"[FOUND] Potential dataset: {item}")
                    found_datasets.append(item)
        
        return found_datasets
    
    def collect_all_images(self, datasets: list) -> dict:
        """Collect images from all datasets with intelligent mapping"""
        all_images = {cat: [] for cat in TARGET_CATEGORIES}
        unmapped_folders = []
        
        for dataset_path in datasets:
            logger.info(f"📂 Processing: {dataset_path}")
            
            try:
                # Handle different dataset structures
                if 'train' in str(dataset_path) or 'valid' in str(dataset_path):
                    search_dirs = [dataset_path]
                else:
                    search_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
                
                for folder in search_dirs:
                    if not folder.is_dir():
                        continue
                    
                    folder_name = folder.name
                    category = self._map_folder_to_category(folder_name)
                    
                    if category and category in TARGET_CATEGORIES:
                        # Collect image files
                        image_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', 
                                          '*.jpeg', '*.JPEG']
                        images = []
                        for ext in image_extensions:
                            images.extend(list(folder.glob(ext)))
                        
                        if images:
                            all_images[category].extend(images)
                            logger.info(f"  ✓ {folder_name} → {category}: {len(images)} images")
                    else:
                        # Track unmapped folders
                        image_count = sum(len(list(folder.glob(ext))) 
                                        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG'])
                        if image_count > 0:
                            unmapped_folders.append((folder_name, image_count))
                            
            except Exception as e:
                logger.error(f"Error processing {dataset_path}: {e}")
        
        # Report unmapped folders
        if unmapped_folders:
            logger.warning("⚠️  Unmapped folders (not used):")
            for folder, count in unmapped_folders:
                logger.warning(f"  - {folder}: {count} images")
        
        return all_images
    
    def _map_folder_to_category(self, folder_name: str) -> str:
        """Map folder name to target category"""
        # Try exact match first
        if folder_name in ULTIMATE_MAPPING:
            return ULTIMATE_MAPPING[folder_name]
        
        # Try fuzzy matching
        folder_clean = folder_name.lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', '')
        
        for key, val in ULTIMATE_MAPPING.items():
            key_clean = key.lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', '')
            
            # Check if folder name contains the key or vice versa
            if (folder_clean in key_clean or key_clean in folder_clean or
                any(part in key_clean for part in folder_clean.split() if len(part) > 3)):
                return val
        
        return None
    
    def _process_single_image(self, args):
        """Process a single image (for parallel processing)"""
        img_path, dest_path, apply_augmentation, image_id = args
        
        try:
            # Load and validate image
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to standard size
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img) / 255.0
            
            # Apply CycleGAN augmentation if requested
            if apply_augmentation:
                try:
                    img_array = self.augmentor.transform(img_array)
                except Exception as e:
                    logger.warning(f"Augmentation failed for {img_path.name}: {e}")
                    # Continue with original image
            
            # Convert back to PIL and save
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            result_img = Image.fromarray(img_array)
            
            # Create destination directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            result_img.save(dest_path, 'JPEG', quality=95)
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def prepare_balanced_dataset(self, 
                                all_images: dict,
                                max_per_category: int = 4000) -> tuple:
        """Create balanced dataset with CycleGAN augmentation"""
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            for category in TARGET_CATEGORIES:
                (self.output_path / split / category).mkdir(parents=True, exist_ok=True)
        
        stats = {}
        total_processed = 0
        
        logger.info("="*70)
        logger.info("CREATING BALANCED DATASET WITH CYCLEGAN AUGMENTATION")
        logger.info("="*70)
        
        for category, images in all_images.items():
            if not images:
                logger.warning(f"No images found for {category}")
                stats[category] = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
                continue
            
            logger.info(f"\n📊 Processing {category}:")
            logger.info(f"  Found: {len(images)} total images")
            
            # Shuffle and cap images
            random.shuffle(images)
            images_to_use = images[:min(len(images), max_per_category)]
            logger.info(f"  Using: {len(images_to_use)} images")
            
            # Split: 70% train, 15% val, 15% test
            n_total = len(images_to_use)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)
            n_test = n_total - n_train - n_val
            
            train_images = images_to_use[:n_train]
            val_images = images_to_use[n_train:n_train + n_val]
            test_images = images_to_use[n_train + n_val:]
            
            # Process each split
            for split_name, split_images in [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]:
                if not split_images:
                    continue
                
                logger.info(f"  Processing {split_name} split: {len(split_images)} images")
                
                # Determine which images to augment
                num_to_augment = int(len(split_images) * self.augmentation_ratio)
                augment_indices = set(random.sample(range(len(split_images)), 
                                                  min(num_to_augment, len(split_images))))
                
                # Prepare processing arguments
                processing_args = []
                for i, img_path in enumerate(split_images):
                    apply_augmentation = i in augment_indices
                    
                    # Create unique filename
                    source_info = img_path.parts[-3] if len(img_path.parts) > 3 else "unknown"
                    aug_suffix = "_aug" if apply_augmentation else ""
                    new_name = f"{source_info}_{img_path.stem}{aug_suffix}.jpg"
                    dest_path = self.output_path / split_name / category / new_name
                    
                    processing_args.append((img_path, dest_path, apply_augmentation, i))
                
                # Process images in parallel
                successful = 0
                failed = 0
                
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit all tasks
                    future_to_args = {executor.submit(self._process_single_image, args): args 
                                    for args in processing_args}
                    
                    # Process results with progress bar
                    with tqdm(total=len(processing_args), 
                             desc=f"  {category} {split_name}",
                             leave=False) as pbar:
                        
                        for future in as_completed(future_to_args):
                            success, error = future.result()
                            if success:
                                successful += 1
                            else:
                                failed += 1
                                if error:
                                    logger.warning(f"Failed to process image: {error}")
                            
                            pbar.update(1)
                            total_processed += 1
                
                logger.info(f"    ✓ {successful} successful, {failed} failed")
            
            stats[category] = {
                'total': n_total,
                'train': len(train_images),
                'val': len(val_images),
                'test': len(test_images)
            }
        
        return stats, total_processed
    
    def save_metadata(self, stats: dict, total_processed: int, datasets_used: list):
        """Save comprehensive metadata"""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'datasets_used': [str(d) for d in datasets_used],
            'total_images_processed': total_processed,
            'category_mapping': ULTIMATE_MAPPING,
            'target_categories': TARGET_CATEGORIES,
            'statistics': stats,
            'augmentation_config': {
                'augmentation_ratio': self.augmentation_ratio,
                'severity': self.severity,
                'num_workers': self.num_workers
            },
            'class_distribution': {}
        }
        
        # Calculate class distribution
        for category, stat in stats.items():
            if stat['total'] > 0:
                metadata['class_distribution'][category] = {
                    'percentage': stat['total'] / sum(s['total'] for s in stats.values()) * 100,
                    'train_samples': stat['train'],
                    'val_samples': stat['val'],
                    'test_samples': stat['test']
                }
        
        # Save metadata
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📄 Metadata saved: {metadata_path}")
        
        # Save label mapping for easy reference
        label_mapping = {i: cat for i, cat in enumerate(sorted(TARGET_CATEGORIES))}
        label_path = self.output_path / 'label_mapping.json'
        with open(label_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info(f"📄 Label mapping saved: {label_path}")
    
    def create_dataset_summary(self, stats: dict):
        """Create human-readable dataset summary"""
        summary_lines = [
            "ULTIMATE PLANT DISEASE DATASET WITH CYCLEGAN AUGMENTATION",
            "="*60,
            f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output: {self.output_path}",
            f"Augmentation ratio: {self.augmentation_ratio:.1%}",
            f"Augmentation severity: {self.severity:.2f}",
            "",
            "DATASET DISTRIBUTION:",
            "-"*30
        ]
        
        total_images = sum(s['total'] for s in stats.values())
        
        for category, stat in stats.items():
            if stat['total'] > 0:
                percentage = stat['total'] / total_images * 100
                summary_lines.extend([
                    f"{category}:",
                    f"  Total: {stat['total']:4d} images ({percentage:.1f}%)",
                    f"  Train: {stat['train']:4d}",
                    f"  Val:   {stat['val']:4d}",
                    f"  Test:  {stat['test']:4d}",
                    ""
                ])
        
        summary_lines.extend([
            f"TOTAL: {total_images} images",
            "",
            "TRAINING RECOMMENDATIONS:",
            "-"*30,
            "• Use batch size: 16-32",
            "• Learning rate: 0.0001-0.001",
            "• Expected accuracy: >85% (with augmentation)",
            "• Model size target: <50MB",
            "",
            "NEXT STEPS:",
            "1. Run: python train_ultimate_cyclegan.py",
            "2. Monitor: tensorboard --logdir=logs",
            "3. Test: python test_real_world_images.py"
        ])
        
        summary_text = "\n".join(summary_lines)
        
        # Save summary
        summary_path = self.output_path / 'DATASET_SUMMARY.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"📄 Summary saved: {summary_path}")
        
        # Print summary
        print("\n" + summary_text)
    
    def run(self, max_per_category: int = 4000):
        """Execute complete dataset preparation pipeline"""
        logger.info("🚀 STARTING ULTIMATE DATASET PREPARATION WITH CYCLEGAN")
        
        try:
            # 1. Find datasets
            datasets = self.find_all_datasets()
            if not datasets:
                raise ValueError("No datasets found! Please check dataset paths.")
            
            logger.info(f"✅ Found {len(datasets)} dataset locations")
            
            # 2. Collect images
            logger.info("📸 Collecting images from all sources...")
            all_images = self.collect_all_images(datasets)
            
            total_available = sum(len(images) for images in all_images.values())
            if total_available == 0:
                raise ValueError("No images found! Check dataset structure.")
            
            logger.info(f"📊 Found {total_available} total images")
            
            # 3. Prepare balanced dataset
            logger.info("⚖️ Creating balanced dataset with augmentation...")
            stats, total_processed = self.prepare_balanced_dataset(
                all_images, max_per_category
            )
            
            # 4. Save metadata
            self.save_metadata(stats, total_processed, datasets)
            
            # 5. Create summary
            self.create_dataset_summary(stats)
            
            logger.info("✅ DATASET PREPARATION COMPLETE!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset preparation failed: {e}")
            return False
        
        finally:
            # Clean up augmentor cache
            self.augmentor.clear_cache()


def main():
    """Main execution function"""
    # Configuration
    config = {
        'output_path': 'datasets/ultimate_plant_disease_cyclegan',
        'augmentation_ratio': 0.3,  # 30% of images will be augmented
        'severity': 0.7,  # Moderate augmentation intensity
        'max_per_category': 4000,  # Maximum images per category
        'num_workers': None  # Auto-detect
    }
    
    print("="*70)
    print("🌱 ULTIMATE PLANT DISEASE DATASET WITH CYCLEGAN AUGMENTATION")
    print("="*70)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create preparator and run
    preparator = UltimateDatasetPreparator(
        output_path=config['output_path'],
        augmentation_ratio=config['augmentation_ratio'],
        severity=config['severity'],
        num_workers=config['num_workers']
    )
    
    success = preparator.run(max_per_category=config['max_per_category'])
    
    if success:
        print("\n🎉 SUCCESS! Dataset ready for training.")
        print(f"📁 Location: {config['output_path']}")
        print("\nNext step: python train_ultimate_cyclegan.py")
    else:
        print("\n❌ FAILED! Check logs for details.")
        exit(1)


if __name__ == "__main__":
    main()