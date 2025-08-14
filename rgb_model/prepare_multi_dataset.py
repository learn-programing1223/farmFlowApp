#!/usr/bin/env python3
"""
Prepare multiple datasets for robust plant disease detection
Combines PlantVillage, PlantNet, and Plant Disease datasets
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image
import json
from datetime import datetime

# Extended category mapping for multiple datasets
CATEGORY_MAPPING = {
    # PlantVillage mappings (existing)
    'Potato___Early_blight': 'Blight',
    'Potato___Late_blight': 'Blight',
    'Tomato_Early_blight': 'Blight',
    'Tomato_Late_blight': 'Blight',
    'Pepper__bell___healthy': 'Healthy',
    'Potato___healthy': 'Healthy',
    'Tomato_healthy': 'Healthy',
    'Pepper__bell___Bacterial_spot': 'Leaf_Spot',
    'Tomato_Bacterial_spot': 'Leaf_Spot',
    'Tomato_Septoria_leaf_spot': 'Leaf_Spot',
    'Tomato__Target_Spot': 'Leaf_Spot',
    'Tomato__Tomato_mosaic_virus': 'Mosaic_Virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Mosaic_Virus',
    'Tomato_Leaf_Mold': 'Nutrient_Deficiency',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Powdery_Mildew',
    
    # Plant Disease dataset mappings (new)
    'Apple___Apple_scab': 'Leaf_Spot',
    'Apple___Black_rot': 'Blight',
    'Apple___Cedar_apple_rust': 'Rust',
    'Apple___healthy': 'Healthy',
    'Grape___Black_rot': 'Blight',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Blight',
    'Grape___healthy': 'Healthy',
    'Squash___Powdery_mildew': 'Powdery_Mildew',
    'Strawberry___Leaf_scorch': 'Leaf_Spot',
    'Strawberry___healthy': 'Healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery_Mildew',
    'Cherry_(including_sour)___healthy': 'Healthy',
    'Peach___Bacterial_spot': 'Leaf_Spot',
    'Peach___healthy': 'Healthy',
    'Corn_(maize)___Common_rust_': 'Rust',
    'Corn_(maize)___healthy': 'Healthy',
    'Corn_(maize)___Northern_Leaf_Blight': 'Blight',
    'Soybean___healthy': 'Healthy',
    
    # PlantNet mappings (if folder structure exists)
    'plantnet_blight': 'Blight',
    'plantnet_healthy': 'Healthy',
    'plantnet_leaf_spot': 'Leaf_Spot',
    'plantnet_powdery_mildew': 'Powdery_Mildew',
    'plantnet_mosaic': 'Mosaic_Virus',
    'plantnet_deficiency': 'Nutrient_Deficiency',
    'plantnet_rust': 'Rust'
}

# Categories we want to train on
TARGET_CATEGORIES = [
    'Blight',
    'Healthy', 
    'Leaf_Spot',
    'Mosaic_Virus',
    'Nutrient_Deficiency',
    'Powdery_Mildew',
    'Rust'  # Can now include Rust with new data!
]

def collect_images_from_sources(data_sources):
    """Collect all images from multiple data sources"""
    
    all_images = {cat: [] for cat in TARGET_CATEGORIES}
    
    for source_path in data_sources:
        source = Path(source_path)
        if not source.exists():
            print(f"[WARNING] Source not found: {source_path}")
            continue
            
        print(f"\nProcessing source: {source_path}")
        
        # Find all image files
        for folder in source.iterdir():
            if not folder.is_dir():
                continue
                
            folder_name = folder.name
            
            # Check if this folder maps to a category
            if folder_name in CATEGORY_MAPPING:
                category = CATEGORY_MAPPING[folder_name]
                if category not in TARGET_CATEGORIES:
                    continue
                    
                # Collect all images from this folder
                images = list(folder.glob('*.jpg')) + \
                        list(folder.glob('*.JPG')) + \
                        list(folder.glob('*.png')) + \
                        list(folder.glob('*.PNG')) + \
                        list(folder.glob('*.jpeg'))
                
                all_images[category].extend(images)
                print(f"  {folder_name} -> {category}: {len(images)} images")
    
    return all_images

def prepare_balanced_dataset(all_images, output_path, max_per_category=2000):
    """Create balanced dataset from collected images"""
    
    output_path = Path(output_path)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for category in TARGET_CATEGORIES:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    for category, images in all_images.items():
        if not images:
            print(f"\n[WARNING] No images for {category}")
            continue
            
        print(f"\nProcessing {category}: {len(images)} total images")
        
        # Shuffle for random selection
        random.shuffle(images)
        
        # Limit to max_per_category
        if len(images) > max_per_category:
            images = images[:max_per_category]
            print(f"  Limited to {max_per_category} images")
        
        # Split: 70% train, 15% val, 15% test
        n_total = len(images)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to output
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            split_path = output_path / split_name / category
            
            for img_path in split_images:
                # Create unique filename to avoid conflicts
                source_dir = img_path.parent.name
                new_name = f"{source_dir}_{img_path.name}"
                dest_path = split_path / new_name
                
                try:
                    # Verify it's a valid image
                    img = Image.open(img_path)
                    img.verify()
                    
                    # Copy the file
                    shutil.copy2(img_path, dest_path)
                    
                except Exception as e:
                    print(f"  [ERROR] Failed to process {img_path}: {e}")
        
        stats[category] = {
            'total': n_total,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
    
    return stats

def main():
    """Main execution"""
    
    print("=" * 70)
    print("MULTI-DATASET PREPARATION FOR ROBUST PLANT DISEASE DETECTION")
    print("=" * 70)
    
    # Define data sources (update these paths based on downloads)
    data_sources = [
        'PlantVillage/PlantVillage',  # Existing
        # Add these after downloading:
        # 'PlantDisease',  # Kaggle Plant Disease dataset
        # 'PlantNet',      # PlantNet field images
    ]
    
    # Check what's available
    available_sources = []
    for source in data_sources:
        if Path(source).exists():
            available_sources.append(source)
            print(f"[OK] Found: {source}")
        else:
            print(f"[INFO] Not found: {source}")
    
    if not available_sources:
        print("\n[ERROR] No data sources found!")
        return
    
    # Collect all images
    print("\n" + "-" * 70)
    print("Collecting images from all sources...")
    all_images = collect_images_from_sources(available_sources)
    
    # Summary of collected images
    print("\n" + "-" * 70)
    print("Summary of collected images:")
    total = 0
    for category, images in all_images.items():
        count = len(images)
        total += count
        print(f"  {category:20s}: {count:6d} images")
    print(f"  {'TOTAL':20s}: {total:6d} images")
    
    # Prepare balanced dataset
    print("\n" + "-" * 70)
    print("Creating balanced dataset...")
    output_path = 'datasets/multi_source_processed'
    stats = prepare_balanced_dataset(all_images, output_path, max_per_category=2000)
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'sources': available_sources,
        'category_mapping': CATEGORY_MAPPING,
        'statistics': stats,
        'total_images': sum(s['total'] for s in stats.values())
    }
    
    metadata_path = Path(output_path) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print("\nFinal distribution:")
    
    for category, stat in stats.items():
        print(f"\n{category}:")
        print(f"  Train: {stat['train']}")
        print(f"  Val:   {stat['val']}")
        print(f"  Test:  {stat['test']}")
        print(f"  Total: {stat['total']}")
    
    print(f"\nTotal images processed: {sum(s['total'] for s in stats.values())}")
    print("\nNext steps:")
    print("1. Download PlantDisease and PlantNet datasets")
    print("2. Place in rgb_model/ directory")
    print("3. Run this script again to include new data")
    print("4. Train with: python train_robust_model.py")
    print("   (update data_path to 'datasets/multi_source_processed')")

if __name__ == "__main__":
    main()