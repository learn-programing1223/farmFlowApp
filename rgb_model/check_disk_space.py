#!/usr/bin/env python3
"""
Check disk space and clean up temporary files
"""

import os
import shutil
from pathlib import Path

def check_disk_space():
    """Check available disk space"""
    cwd = os.getcwd()
    stat = shutil.disk_usage(cwd)
    
    print("DISK SPACE ANALYSIS")
    print("="*50)
    print(f"Current directory: {cwd}")
    print(f"Total space: {stat.total / (1024**3):.2f} GB")
    print(f"Used space: {stat.used / (1024**3):.2f} GB")
    print(f"Free space: {stat.free / (1024**3):.2f} GB")
    print(f"Usage: {(stat.used / stat.total) * 100:.1f}%")
    
    # Check size of data directories
    print("\nLARGE DIRECTORIES:")
    data_dirs = [
        Path('./data'),
        Path('./models'),
        Path('./data/augmented'),
        Path('./data/splits'),
        Path('./data/cache')
    ]
    
    total_size = 0
    for dir_path in data_dirs:
        if dir_path.exists():
            size = get_dir_size(dir_path)
            total_size += size
            print(f"  {dir_path}: {size / (1024**3):.2f} GB")
    
    print(f"\nTotal data size: {total_size / (1024**3):.2f} GB")
    
    # Estimate space needed for training
    print("\nSPACE REQUIREMENTS:")
    print("  Training 7000 images (224x224x3):")
    print("    - Raw data: ~0.8 GB per 1000 images")
    print("    - Train/val/test splits: ~5.6 GB total")
    print("    - Model checkpoints: ~0.5 GB")
    print("    - Total needed: ~7-8 GB free space")
    
    if stat.free / (1024**3) < 8:
        print("\n⚠️  WARNING: Less than 8 GB free space!")
        print("\nRECOMMENDATIONS:")
        print("1. Clean up cache: rm -rf ./data/cache")
        print("2. Remove augmented images after loading: rm -rf ./data/augmented")
        print("3. Use compressed saves (already implemented)")
        print("4. Train with smaller batch size to reduce memory")
        print("5. Use the no-cache training script:")
        print("   python rgb_model/train_robust_model_no_cache.py")
    
    return stat.free / (1024**3)

def get_dir_size(path):
    """Get total size of directory"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except:
        pass
    return total

def cleanup_temp_files():
    """Clean up temporary files to free space"""
    print("\nCLEANUP OPTIONS:")
    
    cleanable = [
        ('Cache files', './data/cache', True),
        ('Old model checkpoints', './models/rgb_model/*/checkpoint*', False),
        ('TensorBoard logs', './models/rgb_model/*/tensorboard', False),
        ('Temporary NumPy files', './data/*.npy', False),
        ('Old splits', './data/splits', False)
    ]
    
    for name, pattern, safe in cleanable:
        if Path(pattern).parent.exists():
            size = get_dir_size(Path(pattern).parent) if not '*' in pattern else 0
            safety = "SAFE" if safe else "CHECK FIRST"
            print(f"  {name}: {pattern} ({size/(1024**2):.1f} MB) [{safety}]")
    
    print("\nTo clean cache (safe):")
    print("  rm -rf ./data/cache")
    
    print("\nTo clean all temporary files (check first):")
    print("  find ./data -name '*.npy' -type f -delete")
    print("  find ./models -name 'checkpoint*' -type f -delete")

if __name__ == "__main__":
    free_gb = check_disk_space()
    print()
    cleanup_temp_files()
    
    if free_gb < 5:
        print("\n🚨 CRITICAL: Very low disk space!")
        print("Training may fail. Please free up space or use external storage.")