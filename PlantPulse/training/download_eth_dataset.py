"""
Download and prepare ETH Zurich Plant Stress Phenotyping Dataset
Best thermal dataset with labeled stress conditions
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path
import shutil

# Dataset information
DATASET_INFO = {
    "name": "ETH Zurich Plant Stress Phenotyping Dataset",
    "url_base": "http://robotics.ethz.ch/~asl-datasets/2018_plant_stress_phenotyping_dataset/",
    "files": {
        "images": "images.zip",  # 2.5GB - Contains thermal infrared stereo pairs
        "pointclouds": "pointclouds.zip",  # 2.5GB - Additional 3D data
        "metadata": "metadata.json"  # Field layout and stress conditions
    },
    "stress_types": ["drought", "low_nitrogen", "surplus_nitrogen", "weed_pressure", "optimal"],
    "crop": "sugar_beet"
}

def download_file(url: str, dest_path: str, chunk_size: int = 8192):
    """Download a file with progress reporting"""
    print(f"Downloading {url}")
    print(f"Destination: {dest_path}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(dest_path, 'wb') as out_file:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='', flush=True)
                
                print("\n✅ Download complete!")
                return True
                
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file with progress"""
    print(f"\nExtracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        
        for i, member in enumerate(zip_ref.namelist()):
            zip_ref.extract(member, extract_to)
            if (i + 1) % 100 == 0:
                print(f"\rExtracted {i+1}/{total_files} files", end='', flush=True)
        
        print(f"\n✅ Extracted {total_files} files")

def prepare_dataset():
    """Download and prepare the ETH Zurich thermal dataset"""
    
    # Create data directory
    data_dir = Path("data/eth_thermal")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ETH ZURICH THERMAL DATASET DOWNLOAD")
    print("=" * 60)
    print(f"Dataset: {DATASET_INFO['name']}")
    print(f"Crop: {DATASET_INFO['crop']}")
    print(f"Stress types: {', '.join(DATASET_INFO['stress_types'])}")
    print(f"Download location: {data_dir}")
    print("=" * 60)
    
    # Download images.zip (thermal data)
    images_url = DATASET_INFO["url_base"] + DATASET_INFO["files"]["images"]
    images_path = data_dir / "images.zip"
    
    if not images_path.exists():
        print("\n1. Downloading thermal images (2.5GB)...")
        print("⚠️  This is a large download and may take several minutes")
        
        success = download_file(images_url, str(images_path))
        if not success:
            print("\n❌ Failed to download images. Please check your internet connection.")
            print("You can manually download from:")
            print(f"  {images_url}")
            return False
    else:
        print("\n✅ Images already downloaded")
    
    # Extract if not already extracted
    extracted_dir = data_dir / "images"
    if not extracted_dir.exists():
        print("\n2. Extracting thermal images...")
        extract_zip(str(images_path), str(data_dir))
    else:
        print("\n✅ Images already extracted")
    
    # Create dataset info file
    info_path = data_dir / "dataset_info.json"
    dataset_info = {
        "name": DATASET_INFO["name"],
        "crop": DATASET_INFO["crop"],
        "stress_types": DATASET_INFO["stress_types"],
        "image_format": "thermal_infrared_stereo_pairs",
        "labels": {
            "0": "optimal",
            "1": "drought", 
            "2": "low_nitrogen",
            "3": "surplus_nitrogen",
            "4": "weed_pressure"
        },
        "notes": [
            "Thermal infrared images showing plant temperature patterns",
            "Labels indicate different stress conditions",
            "Images collected from sugar beet fields",
            "Temperature differences indicate plant health status"
        ]
    }
    
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Dataset info saved to: {info_path}")
    
    # Alternative smaller dataset option
    print("\n" + "=" * 60)
    print("ALTERNATIVE: Date Palm Thermal Dataset")
    print("If ETH download is too large, consider the Date Palm dataset:")
    print("- 800+ thermal images")
    print("- 4 categories: Non-infected, Infected, Badly Damaged, Dead")
    print("- Smaller size (<1GB)")
    print("- Good for pest damage detection")
    print("=" * 60)
    
    return True

def download_alternative_datasets():
    """Download information for alternative thermal datasets"""
    
    alt_dir = Path("data/alternative_datasets")
    alt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save information about alternative datasets
    alternatives = {
        "date_palm_thermal": {
            "name": "Date Palm Trees Thermal Dataset",
            "platform": "Figshare",
            "doi": "10.6084/m9.figshare.25974295",
            "size": "<1GB",
            "images": "800+",
            "labels": ["Non-infected", "Infected", "Badly Damaged", "Dead"],
            "structure": "Paired RGB and thermal folders",
            "application": "Pest damage detection"
        },
        "harvard_uav_thermal": {
            "name": "Harvard Dataverse UAV Multispectral & Thermal",
            "doi": "10.7910/DVN/RYA2ZQ",
            "url": "https://doi.org/10.7910/DVN/RYA2ZQ",
            "sensors": ["senseFly Thermo-map", "Parrot Sequoia", "MSP4C"],
            "crops": ["wheat", "barley", "potato"],
            "collection_date": "2017-06-21",
            "resolution": "cm-level"
        },
        "plantvillage_for_thermal": {
            "name": "PlantVillage (RGB - can simulate thermal patterns)",
            "github": "https://github.com/spMohanty/PlantVillage-Dataset",
            "kaggle": "https://www.kaggle.com/datasets/emmarex/plantdisease",
            "images": "54,303",
            "classes": "38 (14 plant species)",
            "note": "Can be used to train disease patterns, then transfer to thermal"
        }
    }
    
    info_path = alt_dir / "alternative_datasets_info.json"
    with open(info_path, 'w') as f:
        json.dump(alternatives, f, indent=2)
    
    print(f"\n📄 Alternative dataset information saved to: {info_path}")
    
    # Create download instructions
    instructions_path = alt_dir / "download_instructions.txt"
    with open(instructions_path, 'w') as f:
        f.write("THERMAL DATASET DOWNLOAD INSTRUCTIONS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("1. ETH Zurich Dataset (Primary - Best for thermal stress):\n")
        f.write("   wget http://robotics.ethz.ch/~asl-datasets/2018_plant_stress_phenotyping_dataset/images.zip\n")
        f.write("   unzip images.zip\n\n")
        
        f.write("2. Date Palm Thermal (Smaller alternative):\n")
        f.write("   - Visit: https://figshare.com/\n")
        f.write("   - Search DOI: 10.6084/m9.figshare.25974295\n")
        f.write("   - Download thermal folders\n\n")
        
        f.write("3. Harvard UAV Thermal:\n")
        f.write("   - Visit: https://doi.org/10.7910/DVN/RYA2ZQ\n")
        f.write("   - Download thermal sensor data\n\n")
        
        f.write("4. PlantVillage (RGB baseline):\n")
        f.write("   git clone https://github.com/spMohanty/PlantVillage-Dataset\n")
        f.write("   OR\n")
        f.write("   kaggle datasets download -d emmarex/plantdisease\n")
    
    print(f"📝 Download instructions saved to: {instructions_path}")

if __name__ == "__main__":
    print("THERMAL DATASET PREPARATION")
    print("=" * 60)
    print("This script will help you download real thermal datasets")
    print("for training improved plant health models.\n")
    
    # Try to download ETH dataset
    success = prepare_dataset()
    
    # Save alternative dataset information
    download_alternative_datasets()
    
    if success:
        print("\n✅ Dataset preparation complete!")
        print("Next steps:")
        print("1. Run the thermal data loader script")
        print("2. Train the model with real thermal data")
    else:
        print("\n⚠️  ETH dataset download failed")
        print("Please check the alternative datasets in:")
        print("  data/alternative_datasets/")