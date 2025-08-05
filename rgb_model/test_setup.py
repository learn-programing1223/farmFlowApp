#!/usr/bin/env python3
"""
Quick test to verify the RGB model setup is working
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Protobuf gencode.*')

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Testing RGB Model Setup...")
print("="*50)

# Test 1: Check data directory
print("\n1. Checking data directory...")
data_path = "rgb_model/data/PlantVillage/raw/color"
if os.path.exists(data_path):
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print(f"   ✓ Found {len(folders)} disease folders")
    print(f"   Sample folders: {folders[:3]}")
else:
    print(f"   ✗ Data not found at {data_path}")

# Test 2: Import modules
print("\n2. Testing imports...")
try:
    from model import UniversalDiseaseDetector
    from data_loader import MultiDatasetLoader
    from training import ProgressiveTrainer
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n3. Creating model...")
try:
    detector = UniversalDiseaseDetector(num_classes=7)
    detector.compile_model()
    print("   ✓ Model created and compiled")
    print(f"   Total parameters: {detector.model.count_params():,}")
except Exception as e:
    print(f"   ✗ Model error: {e}")

# Test 4: Test data loader
print("\n4. Testing data loader...")
try:
    loader = MultiDatasetLoader(base_data_dir='./data')
    # Just check if it can initialize
    print("   ✓ Data loader initialized")
except Exception as e:
    print(f"   ✗ Data loader error: {e}")

print("\n" + "="*50)
print("Setup test complete!")
print("\nReady to train? Run:")
print("python rgb_model/train_rgb_model.py --plantvillage-subset 0.1 --batch-size 16")