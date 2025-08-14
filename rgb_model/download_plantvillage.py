#!/usr/bin/env python3
"""
Download PlantVillage dataset samples for better training
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

print("=" * 60)
print("PLANTVILLAGE DATASET DOWNLOAD")
print("=" * 60)

# Check if PlantVillage already exists locally
possible_paths = [
    Path("C:/Users/aayan/Downloads/PlantVillage"),
    Path("D:/Datasets/PlantVillage"),
    Path("~/Downloads/PlantVillage").expanduser(),
    Path("../PlantVillage"),
    Path("../../PlantVillage")
]

plantvillage_path = None
for path in possible_paths:
    if path.exists():
        plantvillage_path = path
        print(f"Found existing PlantVillage at: {plantvillage_path}")
        break

if not plantvillage_path:
    print("\nPlantVillage dataset not found locally.")
    print("\nTo download PlantVillage dataset:")
    print("1. Visit: https://github.com/spMohanty/PlantVillage-Dataset")
    print("2. Or: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("3. Download and extract to: C:/Users/aayan/Downloads/PlantVillage")
    print("\nThis dataset contains 54,000+ labeled images of plant diseases")
    print("It's essential for training a robust model.")
    
    # Create a sample structure for now
    print("\nCreating sample structure for demonstration...")
    sample_dir = Path("datasets/plantvillage_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Download a few sample images from public sources
    samples = {
        'Tomato___Early_blight': [
            'https://plantix.net/en/library/assets/custom/crop-diseases/tomato/early-blight-of-tomato/early-blight-of-tomato-3.jpeg',
        ],
        'Tomato___healthy': [
            'https://plantix.net/en/library/assets/custom/crop-stages/Tomato.jpeg',
        ],
        'Potato___Late_blight': [
            'https://plantix.net/en/library/assets/custom/crop-diseases/potato/late-blight-of-potato/late-blight-of-potato-4.jpeg',
        ]
    }
    
    for category, urls in samples.items():
        cat_dir = sample_dir / category
        cat_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                print(f"Downloading sample for {category}...")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(cat_dir / f"sample_{i}.jpg", 'wb') as f:
                        f.write(response.content)
            except:
                pass
    
    print(f"\nCreated sample structure at: {sample_dir}")
    print("\n⚠ NOTE: These are just samples!")
    print("For real training, download the full PlantVillage dataset")
    
else:
    # Copy some PlantVillage data to our training directory
    print("\nCopying PlantVillage data to training directory...")
    
    dest_dir = Path("datasets/plantvillage_real")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Map PlantVillage categories to our 7 categories
    mappings = {
        'Tomato___Early_blight': 'Blight',
        'Tomato___Late_blight': 'Blight',
        'Potato___Early_blight': 'Blight',
        'Potato___Late_blight': 'Blight',
        'Tomato___healthy': 'Healthy',
        'Potato___healthy': 'Healthy',
        'Pepper,_bell___healthy': 'Healthy',
        'Tomato___Septoria_leaf_spot': 'Leaf_Spot',
        'Tomato___Bacterial_spot': 'Leaf_Spot',
        'Tomato___Tomato_mosaic_virus': 'Mosaic_Virus',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf_Spot',
        'Strawberry___Leaf_scorch': 'Leaf_Spot',
        'Apple___Cedar_apple_rust': 'Rust',
        'Corn_(maize)___Common_rust_': 'Rust',
    }
    
    copied = 0
    for pv_folder, our_category in mappings.items():
        source = plantvillage_path / pv_folder
        if source.exists():
            dest = dest_dir / our_category
            dest.mkdir(exist_ok=True)
            
            # Copy first 50 images
            images = list(source.glob('*.JPG'))[:50]
            for img in images:
                shutil.copy2(img, dest / f"pv_{img.name}")
                copied += 1
    
    print(f"Copied {copied} real PlantVillage images")
    print(f"Data saved to: {dest_dir}")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("1. If PlantVillage not found, download it manually")
print("2. Re-run this script after downloading")
print("3. Train with real data: python train_with_real_data.py")
print("4. This will significantly improve model accuracy")