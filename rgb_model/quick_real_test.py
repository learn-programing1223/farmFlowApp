#!/usr/bin/env python3
"""
Quick test showing model performance on real vs clean images
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def simulate_real_world(image):
    """Quick simulation of real-world conditions"""
    # Add noise
    noise = np.random.randn(*image.shape) * 0.03
    image = image + noise
    
    # Random brightness
    brightness = np.random.uniform(0.7, 1.3)
    image = image * brightness
    
    # Slight blur (out of focus)
    from scipy.ndimage import gaussian_filter
    if np.random.rand() > 0.5:
        image = gaussian_filter(image, sigma=0.8)
    
    return np.clip(image, 0, 1).astype(np.float32)

def quick_test():
    print("\n" + "="*70)
    print("QUICK REAL-WORLD PERFORMANCE TEST")
    print("="*70)
    
    # Load model
    model = tf.keras.models.load_model('models/best_working_model.h5')
    
    # Load small test batch
    data_dir = Path('./data/splits')
    X_test = np.load(data_dir / 'X_test.npy').astype(np.float32)[:100]
    y_test = np.load(data_dir / 'y_test.npy').astype(np.float32)[:100]
    
    # Test 1: Clean images
    print("\nTesting on CLEAN images (PlantVillage style)...")
    pred_clean = model.predict(X_test, verbose=0)
    acc_clean = np.mean(np.argmax(pred_clean, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy: {acc_clean:.1%}")
    
    # Test 2: Realistic images
    print("\nTesting on REALISTIC images (simulated real-world)...")
    X_realistic = np.array([simulate_real_world(img) for img in X_test])
    pred_realistic = model.predict(X_realistic, verbose=0)
    acc_realistic = np.mean(np.argmax(pred_realistic, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy: {acc_realistic:.1%}")
    
    # Show the problem
    drop = (acc_clean - acc_realistic) * 100
    print(f"\n⚠️  Performance drop: -{drop:.1f}%")
    
    print("\n" + "="*70)
    print("THE PROBLEM")
    print("="*70)
    print(f"✓ Model gets {acc_clean:.0%} on clean lab images")
    print(f"✗ Model gets {acc_realistic:.0%} on realistic images")
    print(f"This explains why internet images don't work well!")
    
    print("\n" + "="*70)
    print("THE SOLUTION")
    print("="*70)
    print("1. IMMEDIATE FIX: Test-Time Augmentation")
    print("   - Average multiple predictions")
    print("   - Reduces effect of noise/blur")
    print("   - Can add 5-10% accuracy")
    print("\n2. PROPER FIX: Retrain with augmentation")
    print("   - python train_robust_simple.py")
    print("   - Trains model to handle real conditions")
    print("   - Expected: 80-85% on real images")
    print("\n3. BEST FIX: Multi-dataset training")
    print("   - Add PlantDoc (real field images)")
    print("   - Add PlantNet (crowd-sourced images)")
    print("   - Expected: 85-90% on real images")

if __name__ == "__main__":
    quick_test()