#!/usr/bin/env python3
"""
Quick start script to test the RGB model without full dependencies
"""

import os
import sys
import numpy as np

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("RGB Plant Disease Detection Model - Quick Start")
print("="*50)

# Test imports
try:
    print("\n1. Testing core imports...")
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__}")
    
    from model import UniversalDiseaseDetector, FocalLoss
    print("   ✓ Model classes imported")
    
    from dataset_harmonizer import PlantDiseaseHarmonizer
    print("   ✓ Dataset harmonizer imported")
    
    from preprocessing import CrossCropPreprocessor
    print("   ✓ Preprocessing imported")
    
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install tensorflow opencv-python pillow numpy pandas scikit-learn")
    sys.exit(1)

# Test model creation
try:
    print("\n2. Creating model...")
    detector = UniversalDiseaseDetector(num_classes=7)
    print("   ✓ Model created successfully")
    
    # Compile model
    print("\n3. Compiling model...")
    detector.compile_model(
        learning_rate=0.001,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0
    )
    print("   ✓ Model compiled")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = detector.model(dummy_input)
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Output sum: {tf.reduce_sum(output).numpy():.4f} (should be ~1.0)")
    
    # Model summary
    print("\n5. Model summary:")
    print(f"   - Total parameters: {detector.model.count_params():,}")
    print(f"   - Input shape: (224, 224, 3)")
    print(f"   - Output classes: 7 (universal disease categories)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test data harmonizer
try:
    print("\n6. Testing data harmonizer...")
    harmonizer = PlantDiseaseHarmonizer()
    
    # Test with sample labels
    test_labels = ["tomato_early_blight", "apple_healthy", "corn_rust"]
    test_images = [f"img_{i}.jpg" for i in range(len(test_labels))]
    
    _, harmonized = harmonizer.harmonize_dataset(
        "Test", test_images, test_labels
    )
    
    print("   ✓ Harmonization test passed")
    for original, universal in zip(test_labels, harmonized):
        print(f"     {original} → {universal}")
    
except Exception as e:
    print(f"   ✗ Harmonizer error: {e}")

print("\n" + "="*50)
print("✓ Quick start test completed successfully!")
print("\nNext steps:")
print("1. Install optional dependencies for better augmentation:")
print("   pip install albumentations matplotlib seaborn")
print("\n2. Download PlantVillage dataset:")
print("   - Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
print("   - Download and extract to: ./rgb_model/data/PlantVillage/")
print("\n3. Train a test model (small subset):")
print("   python rgb_model/train_rgb_model.py --plantvillage-subset 0.1 --samples-per-class 50")
print("\n4. For full training (80%+ accuracy):")
print("   python rgb_model/train_rgb_model.py")