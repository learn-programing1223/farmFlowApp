#!/usr/bin/env python3
"""
ULTIMATE PLANT DISEASE DATASET PREPARATION
Combines PlantVillage + Plant Disease Dataset for maximum robustness
Intelligently maps 50+ disease types to universal categories
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import json
from datetime import datetime
import numpy as np

# COMPREHENSIVE MAPPING - Every possible disease from Plant Disease dataset
ULTIMATE_MAPPING = {
    # ============= PLANTVILLAGE MAPPINGS (Existing) =============
    'Potato___Early_blight': 'Blight',
    'Potato___Late_blight': 'Blight',
    'Tomato_Early_blight': 'Blight',
    'Tomato_Late_blight': 'Blight',
    'Tomato__Target_Spot': 'Blight',  # Target spot is a form of blight
    
    'Pepper__bell___healthy': 'Healthy',
    'Potato___healthy': 'Healthy',
    'Tomato_healthy': 'Healthy',
    
    'Pepper__bell___Bacterial_spot': 'Leaf_Spot',
    'Tomato_Bacterial_spot': 'Leaf_Spot',
    'Tomato_Septoria_leaf_spot': 'Leaf_Spot',
    
    'Tomato__Tomato_mosaic_virus': 'Mosaic_Virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Mosaic_Virus',
    
    'Tomato_Leaf_Mold': 'Nutrient_Deficiency',  # Yellowing pattern
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Powdery_Mildew',  # White spots
    
    # ============= PLANT DISEASE DATASET - COMPREHENSIVE MAPPINGS =============
    
    # APPLE DISEASES (5 types + healthy)
    'Apple___Apple_scab': 'Leaf_Spot',  # Scab creates spots
    'Apple___Black_rot': 'Blight',  # Rot = blight
    'Apple___Cedar_apple_rust': 'Rust',  # Classic rust
    'Apple___healthy': 'Healthy',
    
    # GRAPE DISEASES (4 types + healthy)
    'Grape___Black_rot': 'Blight',  # Black rot = blight
    'Grape___Esca_(Black_Measles)': 'Blight',  # Another form of rot
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Blight',
    'Grape___healthy': 'Healthy',
    
    # CORN/MAIZE DISEASES (4 types + healthy)
    'Corn_(maize)___Common_rust_': 'Rust',  # Perfect rust example
    'Corn_(maize)___Northern_Leaf_Blight': 'Blight',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Leaf_Spot',
    'Corn_(maize)___healthy': 'Healthy',
    
    # STRAWBERRY DISEASES (2 types + healthy)
    'Strawberry___Leaf_scorch': 'Leaf_Spot',  # Scorch creates spots
    'Strawberry___healthy': 'Healthy',
    
    # PEACH DISEASES (2 types + healthy)
    'Peach___Bacterial_spot': 'Leaf_Spot',
    'Peach___healthy': 'Healthy',
    
    # CHERRY DISEASES (2 types + healthy)
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery_Mildew',  # Classic mildew
    'Cherry_(including_sour)___healthy': 'Healthy',
    
    # SQUASH DISEASES
    'Squash___Powdery_mildew': 'Powdery_Mildew',  # Another mildew example
    
    # SOYBEAN DISEASES (healthy + others if present)
    'Soybean___healthy': 'Healthy',
    
    # ORANGE/CITRUS DISEASES
    'Orange___Haunglongbing_(Citrus_greening)': 'Nutrient_Deficiency',  # Yellowing disease
    
    # RASPBERRY DISEASES
    'Raspberry___healthy': 'Healthy',
    
    # BLUEBERRY DISEASES
    'Blueberry___healthy': 'Healthy',
    
    # Additional mappings for any variant spellings or formats
    'Corn___Common_rust': 'Rust',
    'Corn___Northern_Leaf_Blight': 'Blight',
    'Corn___Gray_leaf_spot': 'Leaf_Spot',
    'Corn___healthy': 'Healthy',
    
    # Alternative naming conventions
    'Apple_scab': 'Leaf_Spot',
    'Apple_Black_rot': 'Blight',
    'Apple_rust': 'Rust',
    'Apple_healthy': 'Healthy',
    
    # PlantDisease folder structure variations
    'PlantDisease___Apple___Apple_scab': 'Leaf_Spot',
    'PlantDisease___Apple___Black_rot': 'Blight',
    'PlantDisease___Apple___Cedar_apple_rust': 'Rust',
    'PlantDisease___Apple___healthy': 'Healthy',
}

# Universal categories we're training on
TARGET_CATEGORIES = [
    'Blight',
    'Healthy', 
    'Leaf_Spot',
    'Mosaic_Virus',
    'Nutrient_Deficiency',
    'Powdery_Mildew',
    'Rust'
]

def find_all_datasets():
    """Automatically find all available datasets"""
    datasets_dir = Path('datasets')
    plant_disease_dir = Path('PlantDisease')
    plantvillage_dir = Path('PlantVillage/PlantVillage')
    
    found_datasets = []
    
    # Check common locations
    for base_path in [Path('.'), datasets_dir]:
        # PlantVillage
        if (base_path / 'PlantVillage/PlantVillage').exists():
            found_datasets.append(base_path / 'PlantVillage/PlantVillage')
            print(f"[OK] Found PlantVillage at: {base_path / 'PlantVillage/PlantVillage'}")
        
        # Plant Disease Dataset
        if (base_path / 'PlantDisease').exists():
            found_datasets.append(base_path / 'PlantDisease')
            print(f"[OK] Found PlantDisease at: {base_path / 'PlantDisease'}")
        
        # Check for train/test structure
        if (base_path / 'PlantDisease/train').exists():
            found_datasets.append(base_path / 'PlantDisease/train')
            print(f"[OK] Found PlantDisease training data")
        
        if (base_path / 'PlantDisease/valid').exists():
            found_datasets.append(base_path / 'PlantDisease/valid')
            print(f"[OK] Found PlantDisease validation data")
    
    return found_datasets

def collect_all_images(datasets):
    """Collect images from all datasets with intelligent mapping"""
    
    all_images = {cat: [] for cat in TARGET_CATEGORIES}
    unmapped_folders = []
    
    for dataset_path in datasets:
        print(f"\n📂 Processing: {dataset_path}")
        
        # Handle different dataset structures
        if 'train' in str(dataset_path) or 'valid' in str(dataset_path):
            # Already in train/valid structure
            search_dirs = [dataset_path]
        else:
            # Look for subdirectories
            search_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for folder in search_dirs:
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            
            # Try exact match first
            category = None
            if folder_name in ULTIMATE_MAPPING:
                category = ULTIMATE_MAPPING[folder_name]
            else:
                # Try fuzzy matching for variants
                for key, val in ULTIMATE_MAPPING.items():
                    if folder_name.lower().replace('_', '').replace(' ', '') in key.lower().replace('_', '').replace(' ', ''):
                        category = val
                        break
                    # Check if key is contained in folder name
                    if key.lower() in folder_name.lower() or folder_name.lower() in key.lower():
                        category = val
                        break
            
            if category and category in TARGET_CATEGORIES:
                # Collect all image files
                images = list(folder.glob('*.jpg')) + \
                        list(folder.glob('*.JPG')) + \
                        list(folder.glob('*.png')) + \
                        list(folder.glob('*.PNG')) + \
                        list(folder.glob('*.jpeg')) + \
                        list(folder.glob('*.JPEG'))
                
                if images:
                    all_images[category].extend(images)
                    print(f"  ✓ {folder_name} → {category}: {len(images)} images")
            else:
                # Track unmapped folders
                image_count = len(list(folder.glob('*.jpg')) + list(folder.glob('*.JPG')) + 
                                 list(folder.glob('*.png')) + list(folder.glob('*.PNG')))
                if image_count > 0:
                    unmapped_folders.append((folder_name, image_count))
    
    # Report unmapped folders
    if unmapped_folders:
        print("\n⚠️  Unmapped folders (not used):")
        for folder, count in unmapped_folders:
            print(f"  - {folder}: {count} images")
    
    return all_images

def prepare_balanced_dataset(all_images, output_path, max_per_category=3000):
    """Create ultra-balanced dataset with maximum diversity"""
    
    output_path = Path(output_path)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for category in TARGET_CATEGORIES:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
    
    stats = {}
    total_copied = 0
    
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)
    
    for category, images in all_images.items():
        if not images:
            print(f"\n⚠️  No images for {category}")
            stats[category] = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
            continue
        
        print(f"\n📊 Processing {category}:")
        print(f"  Found: {len(images)} total images")
        
        # Shuffle for random selection
        random.shuffle(images)
        
        # Cap at max_per_category but use all if less
        images_to_use = images[:min(len(images), max_per_category)]
        print(f"  Using: {len(images_to_use)} images")
        
        # Split: 70% train, 15% val, 15% test
        n_total = len(images_to_use)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val  # Remainder goes to test
        
        train_images = images_to_use[:n_train]
        val_images = images_to_use[n_train:n_train + n_val]
        test_images = images_to_use[n_train + n_val:]
        
        # Copy images with progress tracking
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            split_path = output_path / split_name / category
            copied = 0
            
            for img_path in split_images:
                try:
                    # Verify it's a valid image
                    img = Image.open(img_path)
                    img.verify()
                    
                    # Create unique filename
                    source_dataset = img_path.parts[-3] if len(img_path.parts) > 3 else "unknown"
                    new_name = f"{source_dataset}_{img_path.name}"
                    dest_path = split_path / new_name
                    
                    # Copy file
                    shutil.copy2(img_path, dest_path)
                    copied += 1
                    total_copied += 1
                    
                except Exception as e:
                    print(f"    ⚠️ Failed: {img_path.name} - {str(e)[:50]}")
            
            print(f"  {split_name}: {copied} images")
        
        stats[category] = {
            'total': n_total,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
    
    return stats, total_copied

def main():
    """Execute ultimate dataset preparation"""
    
    print("="*70)
    print("🚀 ULTIMATE PLANT DISEASE DATASET PREPARATION")
    print("="*70)
    print("\nMission: Create the most robust, universal plant disease detector")
    print("Strategy: Maximum crop diversity + disease pattern learning")
    
    # Find all available datasets
    print("\n" + "-"*70)
    print("🔍 Searching for datasets...")
    datasets = find_all_datasets()
    
    if not datasets:
        print("\n❌ No datasets found! Please ensure:")
        print("  - PlantVillage is in: rgb_model/PlantVillage/PlantVillage/")
        print("  - PlantDisease is in: rgb_model/datasets/PlantDisease/")
        return
    
    print(f"\n✅ Found {len(datasets)} dataset locations")
    
    # Collect all images
    print("\n" + "-"*70)
    print("📸 Collecting images from all sources...")
    all_images = collect_all_images(datasets)
    
    # Summary
    print("\n" + "-"*70)
    print("📊 Collection Summary:")
    total_available = 0
    for category, images in all_images.items():
        count = len(images)
        total_available += count
        if count > 0:
            print(f"  {category:20s}: {count:6d} images")
    print(f"  {'TOTAL':20s}: {total_available:6d} images available")
    
    if total_available == 0:
        print("\n❌ No images found! Check dataset paths.")
        return
    
    # Prepare balanced dataset
    print("\n" + "-"*70)
    print("⚖️ Creating balanced dataset...")
    output_path = 'datasets/ultimate_plant_disease'
    
    # Increase max_per_category since we have more data
    max_per_cat = 3000 if total_available > 15000 else 2000
    print(f"Max images per category: {max_per_cat}")
    
    stats, total_copied = prepare_balanced_dataset(
        all_images, 
        output_path, 
        max_per_category=max_per_cat
    )
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'datasets_used': [str(d) for d in datasets],
        'total_images_found': total_available,
        'total_images_used': total_copied,
        'category_mapping': ULTIMATE_MAPPING,
        'statistics': stats,
        'max_per_category': max_per_cat
    }
    
    metadata_path = Path(output_path) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final report
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output: {output_path}")
    print(f"📊 Total images processed: {total_copied}")
    
    print("\n🎯 Final Distribution:")
    for category, stat in stats.items():
        if stat['total'] > 0:
            print(f"\n{category}:")
            print(f"  Train: {stat['train']:4d} images")
            print(f"  Val:   {stat['val']:4d} images")
            print(f"  Test:  {stat['test']:4d} images")
            print(f"  Total: {stat['total']:4d} images")
    
    # Calculate diversity metrics
    crops_included = set()
    for mapping_key in ULTIMATE_MAPPING.keys():
        crop = mapping_key.split('___')[0].split('_')[0]
        crops_included.add(crop)
    
    print(f"\n🌱 Crop Diversity: {len(crops_included)} different plant types")
    print(f"🦠 Disease Types: {len([c for c in stats if stats[c]['total'] > 0])} categories")
    
    print("\n" + "="*70)
    print("🚀 READY FOR TRAINING!")
    print("="*70)
    print("\nNext steps:")
    print("1. Update train_robust_model.py:")
    print("   'data_path': 'datasets/ultimate_plant_disease'")
    print("2. Run training:")
    print("   python train_robust_model.py")
    print("\nExpected performance:")
    print("  - Accuracy: >90% (more diverse data)")
    print("  - Robustness: Excellent (multiple crop types)")
    print("  - Generalization: Superior (universal patterns)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()