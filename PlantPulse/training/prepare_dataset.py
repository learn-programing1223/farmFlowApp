#!/usr/bin/env python3
"""
Dataset preparation helper for PlantPulse model training
"""

import os
import sys
import json
import numpy as np
import requests
from typing import Dict, List
import zipfile
import tarfile

def download_sample_thermal_data():
    """Download sample thermal images for testing"""
    print("\n📥 Downloading sample thermal data...")
    
    # Create sample dataset structure
    os.makedirs('./sample_dataset/thermal', exist_ok=True)
    os.makedirs('./sample_dataset/rgb', exist_ok=True)
    
    # Generate sample thermal data (since we can't download real data without auth)
    print("   Generating sample thermal images...")
    
    for plant_id in range(1, 6):  # 5 sample plants
        for day in range(1, 11):  # 10 days
            # Generate synthetic thermal data
            thermal_data = generate_sample_thermal(plant_id, day)
            
            # Save as numpy array
            filename = f'plant{plant_id}_day{day}_0800.npy'
            filepath = os.path.join('./sample_dataset/thermal', filename)
            np.save(filepath, thermal_data)
    
    # Create sample labels
    labels = generate_sample_labels()
    with open('./sample_dataset/labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print("✅ Sample dataset created at ./sample_dataset/")
    print("   - 5 plants × 10 days = 50 thermal images")
    print("   - Labels include water stress progression")

def generate_sample_thermal(plant_id: int, day: int) -> np.ndarray:
    """Generate realistic thermal pattern for a plant"""
    # Base parameters
    ambient_temp = 25.0 + np.random.normal(0, 2)
    plant_base_temp = ambient_temp - 6.0  # Healthy plant is cooler
    
    # Create thermal image (256x192)
    thermal = np.ones((192, 256)) * ambient_temp
    
    # Add plant shape (elliptical)
    center_x, center_y = 128, 96
    for y in range(192):
        for x in range(256):
            # Elliptical plant shape
            dist = np.sqrt(((x - center_x) / 80)**2 + ((y - center_y) / 60)**2)
            if dist < 1.0:
                # Plant area
                leaf_temp = plant_base_temp + np.random.normal(0, 0.5)
                
                # Add water stress as days progress
                if day > 5:
                    stress_increase = (day - 5) * 0.8
                    leaf_temp += stress_increase
                
                thermal[y, x] = leaf_temp
    
    # Add some realistic noise
    thermal += np.random.normal(0, 0.3, thermal.shape)
    
    return thermal.astype(np.float32)

def generate_sample_labels() -> Dict:
    """Generate sample labels for the dataset"""
    labels = {}
    
    for plant_id in range(1, 6):
        for day in range(1, 11):
            key = f"plant{plant_id}_day{day}"
            
            # Simulate water stress progression
            water_stress = min(1.0, max(0.0, (day - 3) / 10.0))
            
            # Simulate occasional disease
            disease = "healthy"
            if plant_id == 2 and day > 6:
                disease = "fungal"
            elif plant_id == 4 and day > 7:
                disease = "bacterial"
            
            # Simulate nutrient levels
            nutrients = {
                "nitrogen": max(0.1, 0.8 - day * 0.05),
                "phosphorus": 0.6,
                "potassium": max(0.2, 0.7 - day * 0.03)
            }
            
            labels[key] = {
                "water_stress": water_stress,
                "disease": disease,
                "nutrients": nutrients,
                "last_watered_hours": day * 24  # Assume no watering
            }
    
    return labels

def prepare_ieee_dataset():
    """Instructions for preparing IEEE DataPort dataset"""
    print("\n📚 IEEE DataPort Hydroponic Dataset Setup")
    print("=" * 50)
    print("\n1. Visit: https://ieee-dataport.org/open-access/lettuce-dataset")
    print("2. Create a free IEEE DataPort account")
    print("3. Download the dataset ZIP file")
    print("4. Extract the contents to: ./hydroponic_dataset/")
    print("\nExpected structure after extraction:")
    print("   hydroponic_dataset/")
    print("   ├── thermal/         # Thermal images")
    print("   ├── rgb/            # RGB images")
    print("   ├── metadata.csv    # Plant metadata")
    print("   └── sensors.csv     # Environmental data")

def create_custom_dataset_template():
    """Create template for custom dataset"""
    print("\n📁 Creating custom dataset template...")
    
    os.makedirs('./custom_dataset/thermal', exist_ok=True)
    os.makedirs('./custom_dataset/rgb', exist_ok=True)
    
    # Create example labels file
    example_labels = {
        "plant1_day1": {
            "water_stress": 0.0,
            "disease": "healthy",
            "nutrients": {
                "nitrogen": 0.5,
                "phosphorus": 0.5,
                "potassium": 0.5
            },
            "last_watered_hours": 0
        },
        "plant1_day2": {
            "water_stress": 0.1,
            "disease": "healthy",
            "nutrients": {
                "nitrogen": 0.5,
                "phosphorus": 0.5,
                "potassium": 0.5
            },
            "last_watered_hours": 24
        }
    }
    
    with open('./custom_dataset/labels_template.json', 'w') as f:
        json.dump(example_labels, f, indent=2)
    
    # Create README
    readme_content = """# Custom Dataset Format

## Directory Structure
- thermal/: Place your thermal images here
  - Supported formats: .npy (temperature arrays), .png/.tiff (16-bit thermal)
  - Naming: plant{ID}_day{DAY}_{TIME}.ext
  
- rgb/: Optional RGB images
  
- labels.json: Ground truth labels (see labels_template.json)

## Thermal Image Requirements
- Resolution: Any (will be resized to 224x224)
- Temperature data in Celsius
- For .npy files: Direct temperature values
- For image files: 16-bit encoding recommended

## Label Format
See labels_template.json for the expected format.
Each sample needs:
- water_stress: 0.0 to 1.0
- disease: "healthy", "bacterial", "fungal", or "viral"
- nutrients: N/P/K levels (0.0 to 1.0)
"""
    
    with open('./custom_dataset/README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Custom dataset template created at ./custom_dataset/")

def main():
    print("🌱 PlantPulse Dataset Preparation Tool")
    print("=" * 40)
    
    while True:
        print("\nSelect an option:")
        print("1. Download sample dataset (for testing)")
        print("2. Setup IEEE DataPort dataset")
        print("3. Create custom dataset template")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            download_sample_thermal_data()
        elif choice == '2':
            prepare_ieee_dataset()
        elif choice == '3':
            create_custom_dataset_template()
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()