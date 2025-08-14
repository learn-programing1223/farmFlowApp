#!/usr/bin/env python3
"""
Monitor robust model training progress
"""

import os
import time
from pathlib import Path
import json

def check_training_status():
    """Check current training status"""
    
    print("=" * 70)
    print("ROBUST MODEL TRAINING STATUS")
    print("=" * 70)
    
    # Check for saved models
    model_dir = Path('models')
    if model_dir.exists():
        models = list(model_dir.glob('plantvillage_robust*.h5')) + \
                list(model_dir.glob('plantvillage_robust*.tflite'))
        
        if models:
            print("\n[OK] Saved Models:")
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  - {model.name}: {size_mb:.2f} MB")
        else:
            print("\n[INFO] No robust models saved yet")
    
    # Check for history file
    history_file = Path('models/plantvillage_robust_history.json')
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print("\n[OK] Training Complete!")
        print(f"  Final Accuracy: {history['test_results']['accuracy']:.2%}")
        print(f"  Final Precision: {history['test_results']['precision']:.2%}")
        print(f"  Final Recall: {history['test_results']['recall']:.2%}")
        
        if history['test_results']['accuracy'] >= 0.85:
            print("\n[SUCCESS] TARGET ACHIEVED! >85% accuracy!")
        else:
            print(f"\n[WARNING] Accuracy {history['test_results']['accuracy']:.2%} < 85% target")
    else:
        print("\n[INFO] Training still in progress...")
        print("  Expected time on CPU: 30-60 minutes for 30 epochs")
        print("  Check back periodically for updates")
    
    # Check dataset
    print("\n" + "-" * 70)
    print("Dataset Information:")
    processed_dir = Path('datasets/plantvillage_processed')
    if processed_dir.exists():
        for split in ['train', 'val', 'test']:
            split_path = processed_dir / split
            if split_path.exists():
                classes = [d.name for d in split_path.iterdir() if d.is_dir()]
                total = sum(len(list((split_path / cls).glob('*.jpg'))) for cls in classes)
                print(f"  {split}: {total} images across {len(classes)} classes")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_training_status()