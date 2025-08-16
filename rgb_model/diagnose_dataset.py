#!/usr/bin/env python3
"""
Dataset Diagnostic Script
Diagnoses why only 31 images are loading instead of 54,303
"""

import os
from pathlib import Path
import json

def diagnose_dataset(base_path='datasets/plantvillage_processed'):
    """Complete diagnostic of dataset directory structure and contents."""
    
    print("=" * 70)
    print("DATASET DIAGNOSTIC REPORT")
    print("=" * 70)
    
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"❌ ERROR: Path does not exist: {base_path.absolute()}")
        return
    
    print(f"📁 Base path: {base_path.absolute()}")
    
    # Check each split
    splits = ['train', 'val', 'test']
    total_images = 0
    dataset_info = {}
    
    for split in splits:
        split_path = base_path / split
        print(f"\n{'=' * 50}")
        print(f"📂 {split.upper()} Split:")
        print(f"{'=' * 50}")
        
        if not split_path.exists():
            print(f"   ❌ Split directory does not exist: {split_path}")
            continue
            
        # Get all subdirectories (class folders)
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        print(f"   Found {len(class_dirs)} class directories")
        
        split_info = {}
        split_total = 0
        
        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            
            # Count all image files with various extensions
            image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.bmp', '.BMP'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(f'*{ext}')))
            
            # Also check for nested directories
            nested_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
            if nested_dirs:
                print(f"   ⚠️  WARNING: Found {len(nested_dirs)} nested directories in {class_name}")
                for nested in nested_dirs:
                    for ext in image_extensions:
                        image_files.extend(list(nested.glob(f'*{ext}')))
            
            num_images = len(image_files)
            split_total += num_images
            split_info[class_name] = num_images
            
            # Show details
            if num_images > 0:
                print(f"   ✓ {class_name:30s}: {num_images:5d} images")
                # Show sample filenames
                if num_images <= 3:
                    for img in image_files:
                        print(f"      - {img.name}")
            else:
                print(f"   ✗ {class_name:30s}: NO IMAGES FOUND")
                # Check what's in the directory
                all_files = list(class_dir.iterdir())
                if all_files:
                    print(f"      Directory contains {len(all_files)} items:")
                    for item in all_files[:5]:  # Show first 5 items
                        if item.is_dir():
                            print(f"        📁 {item.name}/ (directory)")
                        else:
                            print(f"        📄 {item.name}")
        
        dataset_info[split] = split_info
        total_images += split_total
        print(f"\n   TOTAL {split.upper()}: {split_total} images")
    
    print(f"\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"{'=' * 70}")
    print(f"Total images found: {total_images}")
    
    # Save diagnostic report
    report_path = Path('dataset_diagnostic_report.json')
    with open(report_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\n📊 Detailed report saved to: {report_path.absolute()}")
    
    # Check for PlantVillage source
    print(f"\n{'=' * 70}")
    print("CHECKING FOR ORIGINAL PLANTVILLAGE DATA:")
    print(f"{'=' * 70}")
    
    possible_paths = [
        Path('PlantVillage'),
        Path('../PlantVillage'),
        Path('data/PlantVillage'),
        Path('datasets/PlantVillage'),
        base_path.parent / 'PlantVillage'
    ]
    
    for pv_path in possible_paths:
        if pv_path.exists():
            print(f"✓ Found PlantVillage at: {pv_path.absolute()}")
            # Count images in original dataset
            total_original = 0
            for root, dirs, files in os.walk(pv_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        total_original += 1
            print(f"  Contains {total_original} images")
            break
    else:
        print("✗ Original PlantVillage dataset not found")
        print("  You may need to run: python prepare_plantvillage_data.py")
    
    return dataset_info


def check_data_preparation_script():
    """Check if data preparation script exists and show how to use it."""
    print(f"\n{'=' * 70}")
    print("DATA PREPARATION SCRIPT CHECK:")
    print(f"{'=' * 70}")
    
    prep_scripts = [
        'prepare_plantvillage_data.py',
        'prepare_data.py',
        'process_plantvillage.py'
    ]
    
    for script in prep_scripts:
        if Path(script).exists():
            print(f"✓ Found: {script}")
            print(f"\n  To prepare the dataset, run:")
            print(f"  python {script} --source PlantVillage --output datasets/plantvillage_processed")
            return True
    
    print("✗ No data preparation script found")
    print("\nThe dataset needs to be prepared properly.")
    print("The processed dataset should have this structure:")
    print("""
    datasets/plantvillage_processed/
    ├── train/
    │   ├── Blight/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── Healthy/
    │   ├── Leaf_Spot/
    │   └── ... (all disease classes)
    ├── val/
    │   └── (same structure as train)
    └── test/
        └── (same structure as train)
    """)
    return False


if __name__ == "__main__":
    import sys
    
    # Allow custom path as argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = 'datasets/plantvillage_processed'
    
    print(f"Running diagnostic on: {dataset_path}\n")
    
    # Run diagnostic
    dataset_info = diagnose_dataset(dataset_path)
    
    # Check for data preparation script
    check_data_preparation_script()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)