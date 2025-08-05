#!/usr/bin/env python3
"""
Test script to verify the RGB model works correctly
"""

import os
import sys

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import UniversalDiseaseDetector, FocalLoss, test_model

if __name__ == "__main__":
    print("Testing RGB Universal Disease Detection Model...")
    print("="*50)
    
    try:
        test_model()
        print("\n✓ Model test passed!")
        print("\nNext steps:")
        print("1. Download PlantVillage dataset from Kaggle")
        print("2. Extract to ./rgb_model/data/PlantVillage/")
        print("3. Run: python rgb_model/train_rgb_model.py --plantvillage-subset 0.1")
    except Exception as e:
        print(f"\n✗ Model test failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure TensorFlow is installed: pip install tensorflow")
        print("2. Check Python version (3.8-3.11 recommended)")
        print("3. Try reinstalling: pip install --upgrade tensorflow keras")