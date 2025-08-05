#!/usr/bin/env python3
"""
Download plant disease datasets for RGB model training
"""

import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path
import subprocess
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, base_dir='./data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_file(self, url, filename, chunk_size=8192):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def download_plantvillage_github(self):
        """Download PlantVillage from GitHub (easiest, no registration)"""
        print("\n=== Downloading PlantVillage Dataset ===")
        print("Source: GitHub (no registration required)")
        
        dataset_dir = self.base_dir / 'PlantVillage'
        
        if dataset_dir.exists():
            print(f"PlantVillage already exists at {dataset_dir}")
            return
        
        # Clone the repository
        print("Cloning PlantVillage repository...")
        try:
            subprocess.run([
                'git', 'clone', 
                'https://github.com/spMohanty/PlantVillage-Dataset.git',
                str(dataset_dir)
            ], check=True)
            print("✓ PlantVillage downloaded successfully!")
        except subprocess.CalledProcessError:
            print("Git clone failed. Trying direct download...")
            # Fallback to ZIP download
            zip_url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
            zip_path = self.base_dir / 'plantvillage.zip'
            
            print("Downloading ZIP file...")
            self.download_file(zip_url, zip_path)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)
            
            # Rename extracted folder
            extracted = self.base_dir / 'PlantVillage-Dataset-master'
            if extracted.exists():
                extracted.rename(dataset_dir)
            
            # Clean up
            zip_path.unlink()
            print("✓ PlantVillage downloaded and extracted!")
    
    def download_plantdoc_github(self):
        """Download PlantDoc from GitHub"""
        print("\n=== Downloading PlantDoc Dataset ===")
        print("Source: GitHub (no registration required)")
        
        dataset_dir = self.base_dir / 'PlantDoc'
        
        if dataset_dir.exists():
            print(f"PlantDoc already exists at {dataset_dir}")
            return
        
        # Clone the repository
        print("Cloning PlantDoc repository...")
        try:
            subprocess.run([
                'git', 'clone',
                'https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git',
                str(dataset_dir)
            ], check=True)
            print("✓ PlantDoc downloaded successfully!")
        except subprocess.CalledProcessError:
            print("Git clone failed. Please install git or download manually.")
    
    def download_plantnet_sample(self):
        """Download a sample of PlantNet (full dataset is 31.7GB)"""
        print("\n=== PlantNet-300K Dataset ===")
        print("Full dataset: 31.7GB from https://zenodo.org/records/5645731")
        print("This would take a long time to download.")
        print("\nFor testing, we'll create a sample structure instead.")
        
        dataset_dir = self.base_dir / 'PlantNet'
        dataset_dir.mkdir(exist_ok=True)
        
        # Create a README with download instructions
        readme_content = """# PlantNet-300K Dataset

The full PlantNet-300K dataset is 31.7GB and contains 306,146 images.

## Download Instructions:

1. Visit: https://zenodo.org/records/5645731
2. Click "Download" to get plantnet_300K.zip (31.7GB)
3. Extract to this directory

## Quick Alternative:

For testing the RGB model, you can use just PlantVillage and PlantDoc datasets,
which are much smaller and sufficient for achieving 80%+ accuracy.
"""
        
        with open(dataset_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"✓ Created PlantNet directory with download instructions at {dataset_dir}")
    
    def create_kaggle_download_script(self):
        """Create script for Kaggle dataset download"""
        print("\n=== Kaggle Plant Pathology Datasets ===")
        print("Kaggle datasets require account registration and API key.")
        
        dataset_dir = self.base_dir / 'KagglePlantPathology'
        dataset_dir.mkdir(exist_ok=True)
        
        # Create download script
        script_content = """#!/bin/bash
# Kaggle Plant Pathology Dataset Download Script

echo "=== Kaggle Dataset Download ==="
echo "Prerequisites:"
echo "1. Create a Kaggle account at https://www.kaggle.com"
echo "2. Go to Account Settings > API > Create New API Token"
echo "3. Save kaggle.json to ~/.kaggle/"
echo ""

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

# Check for credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: kaggle.json not found in ~/.kaggle/"
    echo "Please follow the prerequisites above."
    exit 1
fi

# Download Plant Pathology 2020
echo "Downloading Plant Pathology 2020..."
kaggle competitions download -c plant-pathology-2020-fgvc7 -p .

# Download Plant Pathology 2021
echo "Downloading Plant Pathology 2021..."
kaggle competitions download -c plant-pathology-2021-fgvc8 -p .

# Extract datasets
echo "Extracting datasets..."
unzip -q plant-pathology-2020-fgvc7.zip -d plant-pathology-2020
unzip -q plant-pathology-2021-fgvc8.zip -d plant-pathology-2021

# Clean up
rm *.zip

echo "✓ Kaggle datasets downloaded successfully!"
"""
        
        script_path = dataset_dir / 'download_kaggle.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        # Create README
        readme_content = """# Kaggle Plant Pathology Datasets

## Quick Download (Manual):

1. Visit and download:
   - 2020: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data
   - 2021: https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data

2. Extract ZIP files to this directory

## Automated Download:

1. Install Kaggle API: `pip install kaggle`
2. Get API key from https://www.kaggle.com/settings
3. Run: `./download_kaggle.sh`
"""
        
        with open(dataset_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"✓ Created Kaggle download script at {script_path}")
    
    def download_all(self):
        """Download all datasets"""
        print("Plant Disease Dataset Downloader")
        print("="*50)
        
        # Download datasets that don't require registration
        self.download_plantvillage_github()
        self.download_plantdoc_github()
        
        # Provide instructions for large/restricted datasets
        self.download_plantnet_sample()
        self.create_kaggle_download_script()
        
        print("\n" + "="*50)
        print("Summary:")
        print("✓ PlantVillage - Downloaded (or ready to download)")
        print("✓ PlantDoc - Downloaded (or ready to download)")
        print("! PlantNet - Manual download required (31.7GB)")
        print("! Kaggle - Account required (see instructions)")
        
        print("\nYou can start training with just PlantVillage dataset!")
        print("Run: python rgb_model/train_rgb_model.py")


def main():
    downloader = DatasetDownloader(base_dir='rgb_model/data')
    downloader.download_all()


if __name__ == "__main__":
    main()