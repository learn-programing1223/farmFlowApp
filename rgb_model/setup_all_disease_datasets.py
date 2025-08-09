#!/usr/bin/env python3
"""
Complete setup script for ALL disease detection datasets
Maximizes disease training data for 85%+ accuracy
"""

import os
import sys
import json
import subprocess
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import shutil


class CompleteDiseaseDatasetSetup:
    """Sets up all available disease datasets for maximum training data"""
    
    def __init__(self, base_dir='./data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.dataset_status = {}
        
    def check_disk_space(self):
        """Check available disk space"""
        stat = shutil.disk_usage(self.base_dir)
        free_gb = stat.free / (1024**3)
        print(f"Available disk space: {free_gb:.2f} GB")
        
        if free_gb < 50:
            print("⚠️  Warning: Less than 50GB available. Some datasets may not fit.")
            print("    PlantVillage: ~3GB")
            print("    PlantDoc: ~200MB")
            print("    Kaggle datasets: ~2GB")
            print("    PlantNet (optional): 31.7GB")
        
        return free_gb
    
    def download_plantvillage_full(self):
        """Download FULL PlantVillage dataset (54,306 images)"""
        print("\n" + "="*60)
        print("1. PlantVillage Dataset (54,306 disease images)")
        print("="*60)
        
        dataset_dir = self.base_dir / 'PlantVillage'
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            image_count = sum(1 for _ in dataset_dir.rglob('*.jpg'))
            print(f"✓ PlantVillage already exists with {image_count} images")
            self.dataset_status['PlantVillage'] = image_count
            return True
        
        print("Downloading PlantVillage from GitHub...")
        
        try:
            # Try git clone first (fastest)
            subprocess.run([
                'git', 'clone', '--depth', '1',
                'https://github.com/spMohanty/PlantVillage-Dataset.git',
                str(dataset_dir)
            ], check=True)
            
            print("✓ PlantVillage downloaded successfully!")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Git not available, downloading as ZIP...")
            
            # Download as ZIP
            zip_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
            zip_path = self.base_dir / 'plantvillage.zip'
            
            response = requests.get(zip_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)
            
            # Rename
            extracted = self.base_dir / 'PlantVillage-Dataset-master'
            if extracted.exists():
                extracted.rename(dataset_dir)
            
            # Cleanup
            zip_path.unlink()
            print("✓ PlantVillage extracted!")
        
        # Count images
        image_count = sum(1 for _ in dataset_dir.rglob('*.jpg'))
        print(f"Total PlantVillage images: {image_count}")
        self.dataset_status['PlantVillage'] = image_count
        
        return True
    
    def download_plantdoc_full(self):
        """Download PlantDoc dataset (2,598 disease images)"""
        print("\n" + "="*60)
        print("2. PlantDoc Dataset (2,598 annotated disease images)")
        print("="*60)
        
        dataset_dir = self.base_dir / 'PlantDoc'
        
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            image_count = sum(1 for _ in dataset_dir.rglob('*.jpg'))
            print(f"✓ PlantDoc already exists with {image_count} images")
            self.dataset_status['PlantDoc'] = image_count
            return True
        
        print("Downloading PlantDoc from GitHub...")
        
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1',
                'https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git',
                str(dataset_dir)
            ], check=True)
            
            print("✓ PlantDoc downloaded successfully!")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Could not download PlantDoc automatically")
            print("   Please download manually from:")
            print("   https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset")
            self.dataset_status['PlantDoc'] = 0
            return False
        
        # Count images
        image_count = sum(1 for _ in dataset_dir.rglob('*.jpg'))
        print(f"Total PlantDoc images: {image_count}")
        self.dataset_status['PlantDoc'] = image_count
        
        return True
    
    def setup_kaggle_datasets(self):
        """Setup instructions for Kaggle datasets"""
        print("\n" + "="*60)
        print("3. Kaggle Plant Pathology (18,632 apple disease images)")
        print("="*60)
        
        dataset_dir = self.base_dir / 'KagglePlantPathology'
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if (dataset_dir / 'train_images').exists():
            image_count = sum(1 for _ in dataset_dir.rglob('*.jpg'))
            print(f"✓ Kaggle datasets exist with {image_count} images")
            self.dataset_status['Kaggle'] = image_count
            return True
        
        # Create download instructions
        instructions = """
To download Kaggle datasets:

1. Create account at https://www.kaggle.com
2. Visit these competitions and download:
   - https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data
   - https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data
3. Extract ZIP files to: {}

Or use Kaggle CLI:
   pip install kaggle
   kaggle competitions download -c plant-pathology-2021-fgvc8
""".format(dataset_dir)
        
        print(instructions)
        
        with open(dataset_dir / 'DOWNLOAD_INSTRUCTIONS.txt', 'w') as f:
            f.write(instructions)
        
        self.dataset_status['Kaggle'] = 0
        return False
    
    def download_additional_datasets(self):
        """Download additional disease datasets from various sources"""
        print("\n" + "="*60)
        print("4. Additional Disease Datasets")
        print("="*60)
        
        additional_sources = {
            'Rice Disease': {
                'url': 'https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset',
                'size': '~500MB',
                'images': 10407,
                'diseases': ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
            },
            'Tomato Disease': {
                'url': 'https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf',
                'size': '~100MB', 
                'images': 10000,
                'diseases': ['Early blight', 'Late blight', 'Leaf mold', 'Septoria']
            },
            'Potato Disease': {
                'url': 'https://www.kaggle.com/datasets/arjuntejaswi/plant-village',
                'size': '~250MB',
                'images': 2152,
                'diseases': ['Early blight', 'Late blight', 'Healthy']
            }
        }
        
        print("Additional datasets available from Kaggle:")
        for name, info in additional_sources.items():
            print(f"\n{name}:")
            print(f"  Images: {info['images']}")
            print(f"  Diseases: {', '.join(info['diseases'])}")
            print(f"  Download: {info['url']}")
        
        return additional_sources
    
    def verify_and_report(self):
        """Verify all datasets and report statistics"""
        print("\n" + "="*60)
        print("Dataset Summary")
        print("="*60)
        
        total_images = 0
        disease_distribution = {
            'Blight': 0,
            'Leaf_Spot': 0,
            'Powdery_Mildew': 0,
            'Rust': 0,
            'Mosaic_Virus': 0,
            'Nutrient_Deficiency': 0,
            'Healthy': 0
        }
        
        # Check each dataset
        datasets_found = []
        
        # PlantVillage
        pv_dir = self.base_dir / 'PlantVillage'
        if pv_dir.exists():
            pv_count = sum(1 for _ in pv_dir.rglob('*.jpg'))
            if pv_count > 0:
                datasets_found.append(('PlantVillage', pv_count))
                total_images += pv_count
                # Estimate disease distribution
                disease_distribution['Blight'] += 10000
                disease_distribution['Leaf_Spot'] += 15000
                disease_distribution['Rust'] += 5000
                disease_distribution['Healthy'] += 10000
                disease_distribution['Mosaic_Virus'] += 5000
                disease_distribution['Powdery_Mildew'] += 3000
                disease_distribution['Nutrient_Deficiency'] += 6306
        
        # PlantDoc
        pd_dir = self.base_dir / 'PlantDoc'
        if pd_dir.exists():
            pd_count = sum(1 for _ in pd_dir.rglob('*.jpg'))
            if pd_count > 0:
                datasets_found.append(('PlantDoc', pd_count))
                total_images += pd_count
                disease_distribution['Leaf_Spot'] += 1000
                disease_distribution['Healthy'] += 800
                disease_distribution['Blight'] += 798
        
        # Kaggle
        kg_dir = self.base_dir / 'KagglePlantPathology'
        if (kg_dir / 'train_images').exists():
            kg_count = sum(1 for _ in kg_dir.rglob('*.jpg'))
            if kg_count > 0:
                datasets_found.append(('Kaggle', kg_count))
                total_images += kg_count
                disease_distribution['Rust'] += 5000
                disease_distribution['Leaf_Spot'] += 5000
                disease_distribution['Powdery_Mildew'] += 3000
                disease_distribution['Healthy'] += 5632
        
        # PlantNet (if exists)
        pn_path = Path(__file__).parent / 'src' / 'data' / 'plantnet_300K.zip'
        if pn_path.exists():
            datasets_found.append(('PlantNet', 306146))
            total_images += 306146
            disease_distribution['Healthy'] += 306146
        
        print("\nDatasets Available:")
        for name, count in datasets_found:
            print(f"  ✓ {name}: {count:,} images")
        
        if not datasets_found:
            print("  ✗ No datasets found!")
        
        print(f"\nTotal Images Available: {total_images:,}")
        
        print("\nEstimated Universal Category Distribution:")
        for category, count in disease_distribution.items():
            if count > 0:
                percentage = (count / max(total_images, 1)) * 100
                print(f"  {category}: ~{count:,} images ({percentage:.1f}%)")
        
        # Training recommendations
        print("\n" + "="*60)
        print("Training Recommendations")
        print("="*60)
        
        if total_images < 10000:
            print("⚠️  Limited data available. Download more datasets for better accuracy.")
            print("   Minimum recommended: PlantVillage dataset")
        elif total_images < 50000:
            print("✓ Sufficient data for 75-80% accuracy")
            print("  Consider adding Kaggle datasets for better performance")
        else:
            print("✅ Excellent! Sufficient data for 85%+ accuracy target")
            print("  You have enough disease images for robust training")
        
        # Save summary
        summary = {
            'datasets': dict(datasets_found),
            'total_images': total_images,
            'distribution': disease_distribution,
            'ready_for_training': total_images >= 50000
        }
        
        with open(self.base_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    print("Complete Disease Dataset Setup for RGB Model")
    print("Target: 85% Universal Disease Detection Accuracy")
    print("="*60)
    
    setup = CompleteDiseaseDatasetSetup(base_dir='./data')
    
    # Check disk space
    free_space = setup.check_disk_space()
    
    # Download all available datasets
    print("\nDownloading disease datasets...")
    
    # Essential datasets
    setup.download_plantvillage_full()
    setup.download_plantdoc_full()
    
    # Optional but recommended
    setup.setup_kaggle_datasets()
    
    # Show additional options
    setup.download_additional_datasets()
    
    # Verify and report
    summary = setup.verify_and_report()
    
    # Final instructions
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    
    if summary['ready_for_training']:
        print("✅ Ready to train! Run:")
        print("   python train_rgb_model.py --use-all-data")
    else:
        print("📥 Download more datasets:")
        print("   1. Complete PlantVillage download if not done")
        print("   2. Add Kaggle datasets (see instructions in data/KagglePlantPathology/)")
        print("   3. Consider additional disease datasets listed above")
    
    print("\nFor maximum accuracy with all available images:")
    print("   python train_rgb_model.py --samples-per-class 5000 --batch-size 32")


if __name__ == "__main__":
    # Change to rgb_model directory if needed
    if not Path('data').exists() and Path('rgb_model').exists():
        os.chdir('rgb_model')
    
    main()