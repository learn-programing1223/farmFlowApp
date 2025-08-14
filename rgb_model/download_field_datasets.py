#!/usr/bin/env python3
"""
Download diverse field image datasets for robust plant disease detection
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import json
import shutil
import subprocess

def download_plantdoc():
    """Download PlantDoc dataset - real field images"""
    print("\n" + "="*60)
    print("DOWNLOADING PLANTDOC DATASET (Field Images)")
    print("="*60)
    
    # PlantDoc has real-world field images
    base_url = "https://github.com/pratikkayal/PlantDoc-Dataset"
    
    print("PlantDoc contains:")
    print("- 2,598 images of plant diseases")
    print("- Taken in REAL FIELD CONDITIONS")
    print("- 13 plant species, 27 classes")
    print("- Natural backgrounds, variable lighting")
    
    # Create directory
    plantdoc_dir = Path("datasets/PlantDoc")
    plantdoc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTo download PlantDoc:")
    print("1. Visit: https://github.com/pratikkayal/PlantDoc-Dataset")
    print("2. Download the dataset")
    print(f"3. Extract to: {plantdoc_dir.absolute()}")
    
    return plantdoc_dir

def download_plant_pathology():
    """Download Plant Pathology 2021 dataset from Kaggle"""
    print("\n" + "="*60)
    print("DOWNLOADING PLANT PATHOLOGY 2021 (Kaggle)")
    print("="*60)
    
    print("Plant Pathology 2021 contains:")
    print("- 23,000+ apple leaf images")
    print("- REAL ORCHARD CONDITIONS")
    print("- Multiple diseases per image")
    print("- High quality field photography")
    
    kaggle_dir = Path("datasets/PlantPathology2021")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("\nDownloading from Kaggle...")
        # Requires kaggle.json API key in ~/.kaggle/
        subprocess.run([
            "kaggle", "competitions", "download", 
            "-c", "plant-pathology-2021-fgvc8",
            "-p", str(kaggle_dir)
        ], check=False)
    except (ImportError, OSError) as e:
        print(f"\nKaggle download skipped: {str(e)[:100]}")
        print("\nWe'll proceed without Kaggle data for now.")
        print("To enable Kaggle downloads later:")
        print("1. Get API key from kaggle.com/account")
        print("2. Save to ~/.kaggle/kaggle.json")
    
    return kaggle_dir

def setup_data_structure():
    """Organize all datasets into unified structure"""
    print("\n" + "="*60)
    print("SETTING UP UNIFIED DATA STRUCTURE")
    print("="*60)
    
    # Create master dataset directory
    master_dir = Path("datasets/master_field_dataset")
    master_dir.mkdir(parents=True, exist_ok=True)
    
    # Disease categories we care about
    categories = [
        'Blight',
        'Healthy', 
        'Leaf_Spot',
        'Mosaic_Virus',
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    for category in categories:
        (master_dir / category).mkdir(exist_ok=True)
    
    print(f"Created unified structure at: {master_dir.absolute()}")
    print(f"Categories: {categories}")
    
    return master_dir, categories

def map_plantdoc_to_categories():
    """Map PlantDoc classes to our 7 categories"""
    
    mapping = {
        # Blight mappings
        'Tomato Early blight leaf': 'Blight',
        'Tomato Late blight leaf': 'Blight',
        'Potato Early blight leaf': 'Blight',
        'Potato Late blight leaf': 'Blight',
        
        # Healthy mappings
        'Tomato healthy leaf': 'Healthy',
        'Potato healthy leaf': 'Healthy',
        'Corn healthy leaf': 'Healthy',
        'Apple healthy leaf': 'Healthy',
        'Grape healthy leaf': 'Healthy',
        'Peach healthy leaf': 'Healthy',
        'Strawberry healthy leaf': 'Healthy',
        'Cherry healthy leaf': 'Healthy',
        'Pepper bell healthy leaf': 'Healthy',
        
        # Leaf spot mappings
        'Tomato Septoria leaf spot': 'Leaf_Spot',
        'Tomato Bacterial spot leaf': 'Leaf_Spot',
        'Apple Black spot leaf': 'Leaf_Spot',
        'Cherry Powdery mildew leaf': 'Powdery_Mildew',
        
        # Mosaic virus mappings
        'Tomato Mosaic virus leaf': 'Mosaic_Virus',
        
        # Nutrient deficiency (yellowing)
        'Tomato Yellow leaf curl virus': 'Nutrient_Deficiency',
        
        # Rust mappings
        'Corn Common rust leaf': 'Rust',
        'Apple rust leaf': 'Rust',
        
        # Powdery mildew
        'Grape Powdery mildew leaf': 'Powdery_Mildew',
    }
    
    return mapping

def download_additional_sources():
    """Provide links to additional data sources"""
    print("\n" + "="*60)
    print("ADDITIONAL DATA SOURCES")
    print("="*60)
    
    sources = {
        'PlantNet API': {
            'url': 'https://my.plantnet.org/api',
            'description': 'Crowdsourced field images',
            'images': '10,000+',
            'quality': 'Variable but real-world'
        },
        'iNaturalist': {
            'url': 'https://www.inaturalist.org/observations?taxon_id=47126&term_id=17&term_value_id=20',
            'description': 'Plant diseases in nature',
            'images': '5,000+',
            'quality': 'High quality field photos'
        },
        'Google Images Dataset': {
            'url': 'Use image scraping',
            'description': 'Search for each disease type',
            'images': 'Unlimited',
            'quality': 'Mixed'
        },
        'CGIAR Plant Disease': {
            'url': 'https://bigdata.cgiar.org/',
            'description': 'Agricultural research data',
            'images': '2,000+',
            'quality': 'Scientific grade'
        }
    }
    
    print("\nAdditional sources to explore:")
    for name, info in sources.items():
        print(f"\n{name}:")
        print(f"  URL: {info['url']}")
        print(f"  Description: {info['description']}")
        print(f"  Images: {info['images']}")
        print(f"  Quality: {info['quality']}")
    
    return sources

def create_download_script():
    """Create a script to download images from Google"""
    
    script_content = '''#!/usr/bin/env python3
"""
Download images from Google for each disease category
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import time
from pathlib import Path
import hashlib

def download_google_images(query, output_dir, num_images=100):
    """Download images from Google Images search"""
    
    # Setup Chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=options)
    
    # Search URL
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    driver.get(search_url)
    
    # Scroll to load more images
    for _ in range(5):
        driver.execute_script("window.scrollBy(0,1000)")
        time.sleep(2)
    
    # Find image elements
    images = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img in images[:num_images]:
        try:
            img.click()
            time.sleep(1)
            
            # Get full res image
            actual_images = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
            for actual_img in actual_images:
                src = actual_img.get_attribute('src')
                if src and src.startswith('http'):
                    # Download image
                    response = requests.get(src, timeout=5)
                    if response.status_code == 200:
                        # Save with hash as filename
                        file_hash = hashlib.md5(response.content).hexdigest()
                        file_path = Path(output_dir) / f"{file_hash}.jpg"
                        
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        count += 1
                        print(f"Downloaded {count}/{num_images}: {query}")
                        break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    driver.quit()
    return count

# Disease search queries for real-world images
queries = {
    'Blight': [
        'tomato early blight disease field',
        'potato late blight leaves garden',
        'plant blight disease outdoor'
    ],
    'Healthy': [
        'healthy tomato plant leaves',
        'healthy vegetable garden plants',
        'green healthy crop leaves'
    ],
    'Leaf_Spot': [
        'bacterial leaf spot disease plants',
        'fungal leaf spots garden',
        'septoria leaf spot tomato'
    ],
    'Mosaic_Virus': [
        'mosaic virus plant leaves',
        'viral mosaic disease crops',
        'cucumber mosaic virus symptoms'
    ],
    'Nutrient_Deficiency': [
        'nitrogen deficiency plant leaves',
        'yellowing leaves nutrient deficiency',
        'iron chlorosis plant symptoms'
    ],
    'Powdery_Mildew': [
        'powdery mildew disease leaves',
        'white fungus plant disease',
        'powdery mildew vegetables garden'
    ],
    'Rust': [
        'rust disease plant leaves',
        'orange rust fungus crops',
        'bean rust disease field'
    ]
}

if __name__ == "__main__":
    output_base = "datasets/google_images"
    
    for category, search_terms in queries.items():
        for term in search_terms:
            output_dir = Path(output_base) / category
            num_downloaded = download_google_images(term, output_dir, 50)
            print(f"Downloaded {num_downloaded} images for {category}")
'''
    
    script_path = Path("download_google_images.py")
    script_path.write_text(script_content)
    print(f"\nCreated Google Images download script: {script_path}")
    
    return script_path

def main():
    print("="*60)
    print("FIELD IMAGE DATASET ACQUISITION")
    print("="*60)
    
    # Download datasets
    plantdoc_dir = download_plantdoc()
    kaggle_dir = download_plant_pathology()
    
    # Setup unified structure
    master_dir, categories = setup_data_structure()
    
    # Get additional sources
    sources = download_additional_sources()
    
    # Create Google download script
    google_script = create_download_script()
    
    # Summary
    print("\n" + "="*60)
    print("DATASET ACQUISITION SUMMARY")
    print("="*60)
    
    print("\nKey Datasets to Acquire:")
    print("1. PlantDoc - 2,598 field images")
    print("2. Plant Pathology 2021 - 23,000+ field images")
    print("3. PlantNet API - 10,000+ crowdsourced")
    print("4. Google Images - Unlimited")
    
    print("\nWhy These Will Improve Performance:")
    print("- REAL field conditions vs lab")
    print("- Natural backgrounds")
    print("- Variable lighting")
    print("- Multiple angles")
    print("- Realistic disease presentation")
    
    print("\nNext Steps:")
    print("1. Download the datasets manually")
    print("2. Run the Google Images scraper")
    print("3. Organize into unified structure")
    print("4. Begin aggressive augmentation")
    print("5. Train with modern architecture")

if __name__ == "__main__":
    main()