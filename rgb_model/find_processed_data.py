#!/usr/bin/env python3
"""
Find all processed data files to avoid reprocessing
"""

import os
from pathlib import Path
import numpy as np
from datetime import datetime

def find_all_data_files():
    """Find all numpy data files in the project"""
    print("SEARCHING FOR PROCESSED DATA FILES")
    print("="*50)
    
    # Directories to search
    search_dirs = [
        Path('./data'),
        Path('./data/splits'),
        Path('./data/cache'),
        Path('./data/augmented'),
        Path('./models')
    ]
    
    all_files = []
    
    for base_dir in search_dirs:
        if base_dir.exists():
            # Find .npy files
            npy_files = list(base_dir.rglob('*.npy'))
            # Find .npz files (compressed)
            npz_files = list(base_dir.rglob('*.npz'))
            
            if npy_files or npz_files:
                print(f"\nIn {base_dir}:")
                for f in npy_files + npz_files:
                    size_mb = os.path.getsize(f) / (1024**2)
                    mtime = datetime.fromtimestamp(os.path.getmtime(f))
                    print(f"  {f.name}: {size_mb:.1f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
                    all_files.append(f)
    
    # Look for specific processed data patterns
    print("\n" + "-"*50)
    print("LOOKING FOR BALANCED DATASET (7000 samples per class):")
    
    # Check if there's a splits directory with the right data
    splits_dir = Path('./data/splits')
    if splits_dir.exists():
        print(f"\n✓ Found splits directory: {splits_dir}")
        
        # Check for train/val/test splits
        for split in ['train', 'val', 'test']:
            # Compressed format
            compressed = splits_dir / f'{split}_data.npz'
            if compressed.exists():
                data = np.load(compressed)
                X_shape = data['X'].shape
                y_shape = data['y'].shape
                print(f"  {split}: X{X_shape}, y{y_shape} (compressed)")
                data.close()
            else:
                # Regular format
                X_path = splits_dir / f'X_{split}.npy'
                y_path = splits_dir / f'y_{split}.npy'
                if X_path.exists() and y_path.exists():
                    X_shape = np.load(X_path, mmap_mode='r').shape
                    y_shape = np.load(y_path, mmap_mode='r').shape
                    print(f"  {split}: X{X_shape}, y{y_shape}")
    
    # Check for processed full dataset
    if (Path('./data/X_processed.npy').exists() and 
        Path('./data/y_processed.npy').exists()):
        print("\n✓ Found processed full dataset:")
        X_shape = np.load('./data/X_processed.npy', mmap_mode='r').shape
        y_shape = np.load('./data/y_processed.npy', mmap_mode='r').shape
        print(f"  X_processed: {X_shape}")
        print(f"  y_processed: {y_shape}")
    
    print("\n" + "="*50)
    print("TO USE EXISTING DATA:")
    print("="*50)
    
    if splits_dir.exists() and any(splits_dir.iterdir()):
        print("\n✅ Data splits found! You can train directly without reprocessing:")
        print("\npython rgb_model/train_with_existing_data.py --batch-size 32")
    else:
        print("\n⚠️  No splits found. You may need to:")
        print("1. Run data processing once to create splits")
        print("2. Or manually create splits from existing processed data")
    
    return all_files

def check_augmented_images():
    """Check if augmented images exist"""
    aug_dir = Path('./data/augmented')
    if aug_dir.exists():
        print("\n" + "-"*50)
        print("AUGMENTED IMAGES:")
        total_images = 0
        for class_dir in aug_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                if images:
                    print(f"  {class_dir.name}: {len(images)} images")
                    total_images += len(images)
        
        if total_images > 0:
            print(f"\nTotal augmented images: {total_images}")
            print("\nThese images are already created and can be used for training!")

if __name__ == "__main__":
    find_all_data_files()
    check_augmented_images()
    
    print("\n" + "-"*50)
    print("NEXT STEPS:")
    print("-"*50)
    print("1. If splits exist, run:")
    print("   python rgb_model/train_with_existing_data.py")
    print("\n2. To check disk space:")
    print("   python rgb_model/check_disk_space.py")
    print("\n3. To clean up old files:")
    print("   rm -rf ./data/cache")
    print("   find ./data -name '*.npy' -mtime +1 -delete")