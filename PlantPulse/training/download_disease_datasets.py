#!/usr/bin/env python3
"""
Download plant disease datasets and convert to thermal-like images
Includes PlantVillage, OLID, and other disease datasets
"""

import os
import sys
import subprocess
import requests
import zipfile
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm
import urllib.request

class DiseaseDatasetDownloader:
    """Download and prepare disease datasets for thermal training"""
    
    def __init__(self, base_dir: str = "data/disease_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease to thermal signature mapping
        self.disease_thermal_signatures = {
            # Bacterial diseases - initial cooling then heating
            "bacterial": {
                "temp_offset": 3.0,  # +3°C average
                "pattern": "spot",
                "variance": 2.0
            },
            "bacterial_spot": {
                "temp_offset": 4.0,
                "pattern": "spot",
                "variance": 2.5
            },
            "bacterial_blight": {
                "temp_offset": 5.0,
                "pattern": "spreading",
                "variance": 3.0
            },
            
            # Fungal diseases - variable patterns
            "fungal": {
                "temp_offset": -2.0,  # -2°C for biotrophic
                "pattern": "patch",
                "variance": 4.0
            },
            "early_blight": {
                "temp_offset": 6.0,  # +6°C for necrotrophic
                "pattern": "concentric",
                "variance": 3.0
            },
            "late_blight": {
                "temp_offset": 8.0,
                "pattern": "spreading",
                "variance": 4.0
            },
            "powdery_mildew": {
                "temp_offset": -3.0,
                "pattern": "surface",
                "variance": 2.0
            },
            "blast": {
                "temp_offset": 7.0,
                "pattern": "lesion",
                "variance": 3.5
            },
            
            # Viral diseases - mosaic patterns
            "viral": {
                "temp_offset": 2.0,
                "pattern": "mosaic",
                "variance": 5.0
            },
            "mosaic_virus": {
                "temp_offset": 3.0,
                "pattern": "mosaic",
                "variance": 4.0
            },
            "yellow_curl_virus": {
                "temp_offset": 4.0,
                "pattern": "systemic",
                "variance": 3.0
            },
            
            # Healthy baseline
            "healthy": {
                "temp_offset": 0.0,
                "pattern": "uniform",
                "variance": 1.0
            }
        }
    
    def download_plantvillage(self):
        """Download PlantVillage dataset"""
        print("\n" + "="*60)
        print("DOWNLOADING PLANTVILLAGE DATASET")
        print("="*60)
        
        dataset_dir = self.base_dir / "plantvillage"
        
        # Clone from GitHub
        if not dataset_dir.exists():
            print("Cloning PlantVillage repository...")
            try:
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/spMohanty/PlantVillage-Dataset",
                    str(dataset_dir)
                ], check=True)
                print("✅ PlantVillage downloaded successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to clone PlantVillage. Trying alternative method...")
                self.download_plantvillage_kaggle()
        else:
            print("✅ PlantVillage already exists")
        
        return dataset_dir
    
    def download_plantvillage_kaggle(self):
        """Alternative: Download from Kaggle (requires kaggle API)"""
        print("\nTo download from Kaggle:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Setup API key: https://github.com/Kaggle/kaggle-api")
        print("3. Run: kaggle datasets download -d emmarex/plantdisease")
        print("\nOr download manually from:")
        print("https://www.kaggle.com/datasets/emmarex/plantdisease")
    
    def download_rice_diseases(self):
        """Download rice disease datasets"""
        print("\n" + "="*60)
        print("DOWNLOADING RICE DISEASE DATASETS")
        print("="*60)
        
        dataset_dir = self.base_dir / "rice_diseases"
        dataset_dir.mkdir(exist_ok=True)
        
        # Rice leaf diseases from Mendeley
        mendeley_url = "https://data.mendeley.com/public-files/datasets/fwcj7stb8r/files/2f775cb9-3189-42b0-834f-3f61d5b6dd5d/file_downloaded"
        
        print("Downloading rice disease dataset...")
        print("This contains: Bacterial blight, Blast, Brown Spot")
        
        # Note: Direct download from Mendeley requires manual intervention
        print("\nPlease download manually from:")
        print("https://data.mendeley.com/datasets/fwcj7stb8r/1")
        print(f"Extract to: {dataset_dir}")
        
        return dataset_dir
    
    def convert_rgb_to_thermal(self, rgb_image: np.ndarray, disease_type: str) -> np.ndarray:
        """Convert RGB disease image to thermal-like representation"""
        
        # Get disease thermal signature
        if disease_type.lower() in self.disease_thermal_signatures:
            signature = self.disease_thermal_signatures[disease_type.lower()]
        else:
            # Try to match partial disease names
            signature = self.disease_thermal_signatures["healthy"]
            for key, sig in self.disease_thermal_signatures.items():
                if key in disease_type.lower():
                    signature = sig
                    break
        
        # Convert to grayscale
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb_image
        
        # Normalize to 0-1
        gray_norm = gray.astype(np.float32) / 255.0
        
        # Create base temperature map (20-30°C for plants)
        base_temp = 25.0  # Base plant temperature
        thermal = np.ones_like(gray_norm) * base_temp
        
        # Apply disease-specific thermal patterns
        if signature["pattern"] == "spot":
            # Bacterial spots - localized hot spots
            # Detect dark regions (disease spots)
            _, disease_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            disease_regions = disease_mask > 0
            thermal[disease_regions] += signature["temp_offset"]
            
            # Add noise and blur for realism
            noise = np.random.normal(0, signature["variance"], thermal.shape)
            thermal += noise
            thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
            
        elif signature["pattern"] == "mosaic":
            # Viral mosaic - alternating hot/cold patches
            # Create mosaic pattern
            h, w = gray.shape
            block_size = 20
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if (i//block_size + j//block_size) % 2 == 0:
                        thermal[i:i+block_size, j:j+block_size] += signature["temp_offset"]
                    else:
                        thermal[i:i+block_size, j:j+block_size] -= signature["temp_offset"] * 0.5
            
            # Smooth transitions
            thermal = cv2.GaussianBlur(thermal, (11, 11), 0)
            
        elif signature["pattern"] == "spreading":
            # Blight - spreading from edges
            # Create gradient from edges
            edges = cv2.Canny(gray, 50, 150)
            dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            dist_norm = dist_transform / (dist_transform.max() + 1e-5)
            thermal += signature["temp_offset"] * (1 - dist_norm)
            
        elif signature["pattern"] == "concentric":
            # Early blight - concentric rings
            # Detect circular patterns
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=40)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    # Create concentric temperature rings
                    y, x = np.ogrid[:h, :w]
                    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                    thermal[mask] += signature["temp_offset"] * np.sin(
                        np.sqrt((x[mask] - center[0])**2 + (y[mask] - center[1])**2) * 0.5
                    )
            
        else:
            # Default: use image intensity as temperature variation
            thermal += (gray_norm - 0.5) * signature["temp_offset"] * 2
        
        # Add environmental variation
        thermal += np.random.normal(0, signature["variance"] * 0.5, thermal.shape)
        
        # Ensure realistic temperature range (15-35°C)
        thermal = np.clip(thermal, 15, 35)
        
        return thermal.astype(np.float32)
    
    def process_plantvillage_dataset(self, dataset_dir: Path, output_dir: Path):
        """Process PlantVillage dataset and convert to thermal"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease mapping for PlantVillage
        disease_mapping = {
            "bacterial_spot": "bacterial",
            "early_blight": "fungal",
            "late_blight": "fungal", 
            "leaf_mold": "fungal",
            "septoria_leaf_spot": "fungal",
            "spider_mites": "pest",
            "target_spot": "fungal",
            "yellow_leaf_curl_virus": "viral",
            "mosaic_virus": "viral",
            "healthy": "healthy",
            "powdery_mildew": "fungal",
            "rust": "fungal",
            "scab": "fungal",
            "black_rot": "fungal",
            "bacterial_leaf_blight": "bacterial",
            "blast": "fungal",
            "brown_spot": "fungal",
            "tungro": "viral"
        }
        
        processed_count = 0
        disease_counts = {"healthy": 0, "bacterial": 0, "fungal": 0, "viral": 0}
        
        # Find all image files
        image_files = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.JPG"))
        
        print(f"\nFound {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc="Converting to thermal"):
            # Extract disease type from path
            parts = img_path.parts
            disease_name = None
            
            # PlantVillage structure: .../Plant___Disease/image.jpg
            for part in parts:
                if "___" in part:
                    disease_name = part.split("___")[1].lower()
                    break
            
            if not disease_name:
                continue
            
            # Map to our disease categories
            disease_category = "unknown"
            for key, category in disease_mapping.items():
                if key in disease_name:
                    disease_category = category
                    break
            
            if disease_category == "unknown" or disease_category == "pest":
                continue
            
            # Load and convert image
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize to standard size
                img_resized = cv2.resize(img, (224, 224))
                
                # Convert to thermal
                thermal = self.convert_rgb_to_thermal(img_resized, disease_name)
                
                # Save thermal image
                output_subdir = output_dir / disease_category
                output_subdir.mkdir(exist_ok=True)
                
                output_path = output_subdir / f"thermal_{processed_count:05d}.png"
                
                # Normalize to 0-255 for saving
                thermal_norm = ((thermal - 15) / 20 * 255).astype(np.uint8)
                cv2.imwrite(str(output_path), thermal_norm)
                
                # Also save temperature data as .npy
                np.save(str(output_path.with_suffix('.npy')), thermal)
                
                disease_counts[disease_category] += 1
                processed_count += 1
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        # Save dataset info
        info = {
            "total_images": processed_count,
            "disease_distribution": disease_counts,
            "temperature_range": [15, 35],
            "image_size": [224, 224],
            "source": "PlantVillage converted to thermal"
        }
        
        with open(output_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Processed {processed_count} images")
        print(f"Disease distribution: {disease_counts}")
        
        return processed_count
    
    def download_and_prepare_all(self):
        """Download and prepare all disease datasets"""
        
        print("\n" + "="*60)
        print("PLANT DISEASE DATASET DOWNLOAD AND THERMAL CONVERSION")
        print("="*60)
        
        # Download PlantVillage
        plantvillage_dir = self.download_plantvillage()
        
        # Convert to thermal
        thermal_output = self.base_dir / "thermal_diseases"
        
        if plantvillage_dir.exists():
            print("\nConverting PlantVillage to thermal images...")
            self.process_plantvillage_dataset(plantvillage_dir, thermal_output)
        
        # Create combined dataset with all thermal images
        self.create_combined_thermal_dataset(thermal_output)
        
        print("\n✅ Dataset preparation complete!")
        print(f"Thermal disease images saved to: {thermal_output}")
        
        return thermal_output
    
    def create_combined_thermal_dataset(self, thermal_dir: Path):
        """Combine thermal disease data with existing datasets"""
        
        combined_dir = Path("data/all_thermal_datasets/combined_with_diseases")
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCreating combined dataset with diseases...")
        
        # Copy existing thermal data
        existing_thermal = Path("data/all_thermal_datasets/combined")
        if existing_thermal.exists():
            print("Copying existing thermal data...")
            for img in existing_thermal.glob("*.png"):
                shutil.copy(img, combined_dir / f"existing_{img.name}")
        
        # Copy new disease thermal data
        if thermal_dir.exists():
            print("Adding disease thermal data...")
            for disease_type in ["healthy", "bacterial", "fungal", "viral"]:
                disease_dir = thermal_dir / disease_type
                if disease_dir.exists():
                    for i, img in enumerate(disease_dir.glob("*.png")):
                        shutil.copy(img, combined_dir / f"disease_{disease_type}_{i:05d}.png")
        
        print(f"✅ Combined dataset created at: {combined_dir}")

def main():
    """Main download and conversion pipeline"""
    
    downloader = DiseaseDatasetDownloader()
    
    # Check if user wants to proceed
    print("This script will:")
    print("1. Download PlantVillage dataset (>1GB)")
    print("2. Convert disease images to thermal representations")
    print("3. Create a combined dataset for training")
    print("\nThis may take 30-60 minutes depending on internet speed.")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Download and prepare
    thermal_output = downloader.download_and_prepare_all()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Train with the new disease data:")
    print("   python train_with_real_thermal.py data/all_thermal_datasets/combined_with_diseases combined")
    print("\n2. The model should now properly learn disease patterns!")
    print("\n3. Expected improvement: Disease classification should no longer be 100%")

if __name__ == "__main__":
    main()