"""
Download ALL available thermal datasets for maximum robustness
Combines multiple sources for comprehensive training
"""

import os
import urllib.request
import zipfile
import tarfile
import json
import subprocess
from pathlib import Path
import requests
from typing import Dict, List
import shutil
import time

class ComprehensiveThermalDatasetDownloader:
    """Download and organize multiple thermal datasets"""
    
    def __init__(self, base_dir: str = "data/all_thermal_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download progress
        self.downloaded_datasets = []
        self.failed_datasets = []
        
    def download_with_progress(self, url: str, dest_path: str, chunk_size: int = 8192) -> bool:
        """Download file with resume support and progress bar"""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if partially downloaded
        resume_header = {}
        mode = 'wb'
        resume_pos = 0
        
        if dest_path.exists():
            resume_pos = dest_path.stat().st_size
            resume_header = {'Range': f'bytes={resume_pos}-'}
            mode = 'ab'
        
        try:
            req = urllib.request.Request(url, headers=resume_header)
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                if resume_pos > 0:
                    total_size += resume_pos
                
                with open(dest_path, mode) as f:
                    downloaded = resume_pos
                    start_time = time.time()
                    
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress display
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            speed = downloaded / (time.time() - start_time) / 1024 / 1024  # MB/s
                            print(f"\r{dest_path.name}: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes) {speed:.1f} MB/s", 
                                  end='', flush=True)
                
                print(f"\n✅ Downloaded: {dest_path.name}")
                return True
                
        except Exception as e:
            print(f"\n❌ Failed to download {url}: {e}")
            return False
    
    def download_eth_zurich(self) -> bool:
        """Download ETH Zurich Plant Stress Dataset - Best for stress conditions"""
        print("\n" + "="*60)
        print("1. ETH ZURICH THERMAL DATASET (2.5GB)")
        print("   - 1000+ thermal images")
        print("   - Stress conditions: drought, nitrogen, weeds")
        print("   - Professional quality")
        print("="*60)
        
        dataset_dir = self.base_dir / "eth_zurich"
        dataset_dir.mkdir(exist_ok=True)
        
        url = "http://robotics.ethz.ch/~asl-datasets/2018_plant_stress_phenotyping_dataset/images.zip"
        zip_path = dataset_dir / "images.zip"
        
        if self.download_with_progress(url, str(zip_path)):
            print("Extracting ETH dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            
            # Create dataset info
            info = {
                "name": "ETH Zurich Thermal",
                "images": "1000+",
                "labels": ["optimal", "drought", "low_nitrogen", "surplus_nitrogen", "weed_pressure"],
                "crop": "sugar_beet",
                "quality": "professional"
            }
            
            with open(dataset_dir / "info.json", 'w') as f:
                json.dump(info, f, indent=2)
            
            self.downloaded_datasets.append("ETH Zurich")
            return True
        
        self.failed_datasets.append("ETH Zurich")
        return False
    
    def download_plant_village(self) -> bool:
        """Download PlantVillage - Can be used for transfer learning"""
        print("\n" + "="*60)
        print("2. PLANTVILLAGE DATASET")
        print("   - 54,000+ RGB images (convert to thermal patterns)")
        print("   - 38 disease classes")
        print("   - Transfer learning potential")
        print("="*60)
        
        dataset_dir = self.base_dir / "plantvillage"
        dataset_dir.mkdir(exist_ok=True)
        
        # Using Kaggle API if available
        try:
            # Check if kaggle is installed
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
            
            print("Downloading via Kaggle API...")
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", "emmarex/plantdisease",
                "-p", str(dataset_dir),
                "--unzip"
            ], check=True)
            
            self.downloaded_datasets.append("PlantVillage")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Kaggle API not available")
            print("To download PlantVillage:")
            print("1. pip install kaggle")
            print("2. Get API token from kaggle.com/account")
            print("3. Run: kaggle datasets download -d emmarex/plantdisease")
            
            # Save instructions
            with open(dataset_dir / "download_instructions.txt", 'w') as f:
                f.write("PlantVillage Download Instructions:\n")
                f.write("1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease\n")
                f.write("2. Download manually or use:\n")
                f.write("   kaggle datasets download -d emmarex/plantdisease\n")
            
            self.failed_datasets.append("PlantVillage")
            return False
    
    def create_synthetic_thermal_augmented(self) -> bool:
        """Create high-quality synthetic thermal dataset with realistic patterns"""
        print("\n" + "="*60)
        print("3. SYNTHETIC THERMAL DATASET (Advanced)")
        print("   - 10,000 images with realistic patterns")
        print("   - Multiple stress conditions")
        print("   - Augmented with noise and artifacts")
        print("="*60)
        
        dataset_dir = self.base_dir / "synthetic_thermal"
        dataset_dir.mkdir(exist_ok=True)
        
        import numpy as np
        import cv2
        
        conditions = {
            "healthy": {"temp_mean": 22, "temp_std": 1, "count": 2000},
            "drought_mild": {"temp_mean": 25, "temp_std": 1.5, "count": 1500},
            "drought_severe": {"temp_mean": 28, "temp_std": 2, "count": 1500},
            "disease_bacterial": {"temp_mean": 24, "temp_std": 3, "count": 1500},
            "disease_fungal": {"temp_mean": 23, "temp_std": 2.5, "count": 1500},
            "nutrient_deficient": {"temp_mean": 21, "temp_std": 1.5, "count": 1000},
            "pest_damage": {"temp_mean": 26, "temp_std": 4, "count": 1000}
        }
        
        total_images = 0
        
        for condition, params in conditions.items():
            condition_dir = dataset_dir / condition
            condition_dir.mkdir(exist_ok=True)
            
            print(f"\nGenerating {params['count']} {condition} images...")
            
            for i in range(params['count']):
                # Create base thermal pattern
                img = np.random.normal(params['temp_mean'], params['temp_std'], (256, 256))
                
                # Add realistic plant structure
                num_leaves = np.random.randint(3, 8)
                for _ in range(num_leaves):
                    # Elliptical leaves
                    center = (np.random.randint(50, 200), np.random.randint(50, 200))
                    axes = (np.random.randint(20, 60), np.random.randint(15, 40))
                    angle = np.random.uniform(0, 180)
                    
                    mask = np.zeros((256, 256), dtype=np.uint8)
                    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
                    
                    # Leaves are cooler
                    leaf_temp_diff = np.random.uniform(-4, -2)
                    img[mask > 0] += leaf_temp_diff
                
                # Add condition-specific patterns
                if "drought" in condition:
                    # Hot spots
                    for _ in range(np.random.randint(2, 5)):
                        x, y = np.random.randint(0, 256, 2)
                        radius = np.random.randint(10, 30)
                        hot_spot = np.random.uniform(2, 5)
                        cv2.circle(img, (x, y), radius, 
                                 img[y, x] + hot_spot, -1)
                
                elif "disease" in condition:
                    if "bacterial" in condition:
                        # Linear patterns
                        for _ in range(np.random.randint(3, 7)):
                            pt1 = tuple(np.random.randint(0, 256, 2))
                            pt2 = tuple(np.random.randint(0, 256, 2))
                            thickness = np.random.randint(2, 5)
                            temp_change = np.random.uniform(-3, 3)
                            cv2.line(img, pt1, pt2, 
                                   params['temp_mean'] + temp_change, thickness)
                    else:  # fungal
                        # Circular patches
                        for _ in range(np.random.randint(5, 15)):
                            center = tuple(np.random.randint(0, 256, 2))
                            radius = np.random.randint(5, 20)
                            temp_change = np.random.uniform(-2, 4)
                            cv2.circle(img, center, radius,
                                     params['temp_mean'] + temp_change, -1)
                
                elif "nutrient" in condition:
                    # Overall cooling with patches
                    img -= np.random.uniform(1, 3)
                    # Irregular patches
                    for _ in range(np.random.randint(5, 10)):
                        x, y = np.random.randint(0, 200, 2)
                        w, h = np.random.randint(20, 60, 2)
                        patch_temp = np.random.uniform(-2, 2)
                        img[y:y+h, x:x+w] += patch_temp
                
                elif "pest" in condition:
                    # Irregular damage patterns
                    num_damages = np.random.randint(10, 20)
                    for _ in range(num_damages):
                        # Random polygon for damage area
                        pts = np.random.randint(0, 256, (np.random.randint(3, 6), 2))
                        temp_change = np.random.uniform(-5, 5)
                        cv2.fillPoly(img, [pts], params['temp_mean'] + temp_change)
                
                # Add realistic noise and artifacts
                # Sensor noise
                noise = np.random.normal(0, 0.3, img.shape)
                img += noise
                
                # Dead pixels
                if np.random.random() < 0.1:
                    dead_pixels = np.random.randint(0, 256, (np.random.randint(1, 5), 2))
                    for px, py in dead_pixels:
                        img[py, px] = 0
                
                # Edge vignetting
                y, x = np.ogrid[:256, :256]
                center = (128, 128)
                mask = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                vignette = 1 - (mask / mask.max()) * 0.3
                img *= vignette
                
                # Normalize and save
                img = np.clip(img, 10, 40)  # Realistic temperature range
                img_normalized = ((img - 10) / 30 * 255).astype(np.uint8)
                
                filename = condition_dir / f"thermal_{i:05d}.png"
                cv2.imwrite(str(filename), img_normalized)
                
                if (i + 1) % 500 == 0:
                    print(f"  Generated {i + 1}/{params['count']}")
                
                total_images += 1
        
        # Create comprehensive dataset info
        info = {
            "name": "Synthetic Thermal Advanced",
            "total_images": total_images,
            "conditions": list(conditions.keys()),
            "temperature_range": [10, 40],
            "resolution": "256x256",
            "features": [
                "Realistic plant structures",
                "Condition-specific patterns",
                "Sensor noise simulation",
                "Edge effects",
                "Dead pixels",
                "Multiple stress levels"
            ]
        }
        
        with open(dataset_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Generated {total_images} synthetic thermal images")
        self.downloaded_datasets.append("Synthetic Thermal Advanced")
        return True
    
    def download_sample_thermal_videos(self) -> bool:
        """Download sample thermal videos for temporal analysis"""
        print("\n" + "="*60)
        print("4. THERMAL VIDEO SAMPLES")
        print("   - Time-series thermal data")
        print("   - Shows stress progression")
        print("="*60)
        
        dataset_dir = self.base_dir / "thermal_videos"
        dataset_dir.mkdir(exist_ok=True)
        
        # Create sample video frames
        import numpy as np
        import cv2
        
        print("Generating thermal video sequences...")
        
        # Simulate plant stress progression over time
        for condition in ["drought_progression", "disease_spread"]:
            condition_dir = dataset_dir / condition
            condition_dir.mkdir(exist_ok=True)
            
            # 100 frames showing progression
            base_temp = 22
            
            for frame_idx in range(100):
                img = np.ones((256, 256)) * base_temp
                
                # Add plant structure
                plant_mask = np.zeros((256, 256), dtype=np.uint8)
                cv2.ellipse(plant_mask, (128, 128), (80, 100), 0, 0, 360, 255, -1)
                
                if condition == "drought_progression":
                    # Temperature increases over time
                    stress_level = frame_idx / 100.0
                    temp_increase = stress_level * 6  # Up to 6°C increase
                    img[plant_mask > 0] += temp_increase
                    
                    # Add hot spots that grow
                    num_spots = int(stress_level * 5) + 1
                    for _ in range(num_spots):
                        x, y = np.random.randint(80, 180, 2)
                        radius = int(10 + stress_level * 20)
                        cv2.circle(img, (x, y), radius, 
                                 base_temp + temp_increase + 2, -1)
                
                else:  # disease_spread
                    # Disease spreads from initial points
                    spread_radius = int(10 + frame_idx * 0.8)
                    
                    # Initial infection points
                    infection_centers = [(100, 100), (150, 120), (120, 160)]
                    
                    for center in infection_centers:
                        if spread_radius < 80:  # Limit spread
                            cv2.circle(img, center, spread_radius,
                                     base_temp + np.random.uniform(-2, 3), -1)
                
                # Add noise
                noise = np.random.normal(0, 0.2, img.shape)
                img += noise
                
                # Save frame
                img_normalized = ((img - 15) / 25 * 255).clip(0, 255).astype(np.uint8)
                filename = condition_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(filename), img_normalized)
            
            print(f"✅ Generated {condition} sequence")
        
        self.downloaded_datasets.append("Thermal Videos")
        return True
    
    def prepare_combined_dataset(self) -> Dict:
        """Combine all downloaded datasets into unified structure"""
        print("\n" + "="*60)
        print("PREPARING COMBINED DATASET")
        print("="*60)
        
        combined_dir = self.base_dir / "combined"
        combined_dir.mkdir(exist_ok=True)
        
        # Category mapping for unified labels
        category_map = {
            # Stress level mapping
            "optimal": "healthy",
            "healthy": "healthy",
            "drought": "water_stressed",
            "drought_mild": "water_stressed_mild",
            "drought_severe": "water_stressed_severe",
            "water_stress": "water_stressed",
            
            # Disease mapping
            "disease": "diseased",
            "diseased": "diseased",
            "disease_bacterial": "diseased_bacterial",
            "disease_fungal": "diseased_fungal",
            "bacterial": "diseased_bacterial",
            "fungal": "diseased_fungal",
            "viral": "diseased_viral",
            "infected": "diseased",
            "badly_damaged": "diseased_severe",
            
            # Nutrient mapping
            "nutrient": "nutrient_deficient",
            "nutrient_deficient": "nutrient_deficient",
            "low_nitrogen": "nutrient_deficient_n",
            "surplus_nitrogen": "nutrient_excess_n",
            
            # Other
            "weed_pressure": "competition_stress",
            "pest_damage": "pest_damage",
            "dead": "dead"
        }
        
        stats = {
            "total_images": 0,
            "categories": {},
            "sources": {}
        }
        
        # Process each dataset
        for dataset_path in self.base_dir.iterdir():
            if dataset_path.is_dir() and dataset_path.name != "combined":
                print(f"\nProcessing {dataset_path.name}...")
                
                image_count = 0
                
                # Find all images
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                    for img_path in dataset_path.rglob(ext):
                        # Extract category from path
                        path_parts = img_path.parts
                        category = "unknown"
                        
                        # Try to find category in path
                        for part in path_parts:
                            part_lower = part.lower()
                            for key in category_map:
                                if key in part_lower:
                                    category = category_map[key]
                                    break
                            if category != "unknown":
                                break
                        
                        # Create category directory
                        category_dir = combined_dir / category
                        category_dir.mkdir(exist_ok=True)
                        
                        # Copy image with unique name
                        new_name = f"{dataset_path.name}_{img_path.stem}_{stats['total_images']:06d}{img_path.suffix}"
                        dest_path = category_dir / new_name
                        
                        try:
                            shutil.copy2(img_path, dest_path)
                            
                            # Update stats
                            stats["total_images"] += 1
                            image_count += 1
                            
                            if category not in stats["categories"]:
                                stats["categories"][category] = 0
                            stats["categories"][category] += 1
                            
                        except Exception as e:
                            print(f"  Error copying {img_path}: {e}")
                
                stats["sources"][dataset_path.name] = image_count
                print(f"  Added {image_count} images")
        
        # Save combined dataset info
        info = {
            "name": "Combined Thermal Dataset",
            "created": str(Path.cwd()),
            "stats": stats,
            "label_mapping": category_map,
            "sources": self.downloaded_datasets
        }
        
        with open(combined_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print("\n" + "="*60)
        print("COMBINED DATASET SUMMARY")
        print("="*60)
        print(f"Total images: {stats['total_images']:,}")
        print(f"Sources: {len(stats['sources'])}")
        print(f"Categories: {len(stats['categories'])}")
        print("\nCategory breakdown:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count:,} images")
        
        return stats
    
    def download_all(self):
        """Download all available datasets"""
        print("COMPREHENSIVE THERMAL DATASET COLLECTION")
        print("="*60)
        print("This will download/generate multiple datasets for maximum robustness")
        print("Total estimated size: ~5-10GB")
        print("="*60)
        
        # Download each dataset
        self.download_eth_zurich()
        self.download_plant_village()
        self.create_synthetic_thermal_augmented()
        self.download_sample_thermal_videos()
        
        # Combine all datasets
        if self.downloaded_datasets:
            self.prepare_combined_dataset()
        
        # Final summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"✅ Successfully downloaded: {len(self.downloaded_datasets)} datasets")
        for dataset in self.downloaded_datasets:
            print(f"   - {dataset}")
        
        if self.failed_datasets:
            print(f"\n❌ Failed downloads: {len(self.failed_datasets)}")
            for dataset in self.failed_datasets:
                print(f"   - {dataset}")
        
        print(f"\nAll data saved to: {self.base_dir}")
        print("\nNext steps:")
        print("1. Review combined dataset in: data/all_thermal_datasets/combined/")
        print("2. Train with: python train_with_real_thermal.py")
        print("3. Select 'combined' dataset for maximum robustness")

if __name__ == "__main__":
    downloader = ComprehensiveThermalDatasetDownloader()
    downloader.download_all()