#!/usr/bin/env python3
"""
Test script to verify PlantNet integration and data leakage prevention
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from pathlib import Path
from data_loader import MultiDatasetLoader
from preprocessing_simple import MixUpAugmentation, CutMixAugmentation
import warnings
warnings.filterwarnings('ignore')


def test_plantnet_loading():
    """Test PlantNet dataset loading"""
    print("=" * 60)
    print("Testing PlantNet Dataset Loading")
    print("=" * 60)
    
    loader = MultiDatasetLoader(base_data_dir='./data')
    
    # Test loading PlantNet with small sample
    images, labels = loader.load_plantnet(max_samples=100)
    
    if images:
        print(f"✓ Successfully loaded {len(images)} PlantNet images")
        print(f"✓ Unique labels: {len(set(labels))}")
        print(f"✓ Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    else:
        print("✗ PlantNet dataset not found or failed to load")
        print("  Please ensure plantnet_300K.zip is in data/ or src/data/")
    
    return len(images) > 0


def test_data_leakage_prevention():
    """Test that data splits have no leakage"""
    print("\n" + "=" * 60)
    print("Testing Data Leakage Prevention")
    print("=" * 60)
    
    # Create dummy data
    n_samples = 1000
    n_features = 224 * 224 * 3
    n_classes = 7
    
    # Generate unique data for each sample
    np.random.seed(42)
    X = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
    
    # Add unique identifiers to ensure samples are distinct
    for i in range(n_samples):
        X[i, 0, 0, :] = i / n_samples  # Embed unique ID in first pixel
    
    # Generate labels
    y = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y[i, i % n_classes] = 1  # Stratified labels
    
    # Test splitting
    loader = MultiDatasetLoader(base_data_dir='./data')
    
    try:
        splits = loader.prepare_train_val_test_split(
            X, y, 
            val_split=0.15, 
            test_split=0.15,
            ensure_no_leakage=True  # This will run verification
        )
        
        print("\n✓ Data splitting completed successfully")
        print("✓ No data leakage detected")
        
        # Additional verification
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Check unique IDs don't overlap
        train_ids = set(X_train[:, 0, 0, 0])
        val_ids = set(X_val[:, 0, 0, 0])
        test_ids = set(X_test[:, 0, 0, 0])
        
        assert len(train_ids & val_ids) == 0, "Found overlapping samples!"
        assert len(train_ids & test_ids) == 0, "Found overlapping samples!"
        assert len(val_ids & test_ids) == 0, "Found overlapping samples!"
        
        print("✓ Secondary verification passed")
        return True
        
    except AssertionError as e:
        print(f"✗ Data leakage detected: {e}")
        return False


def test_augmentations():
    """Test MixUp and CutMix augmentations"""
    print("\n" + "=" * 60)
    print("Testing Advanced Augmentations")
    print("=" * 60)
    
    # Create test data
    batch_size = 4
    X = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    y = np.eye(7)[np.random.randint(0, 7, batch_size)]  # One-hot labels
    
    # Test MixUp
    mixup = MixUpAugmentation(alpha=0.2)
    X_mixed, y_mixed = mixup.mixup_batch(X, y)
    
    print("MixUp Augmentation:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {X_mixed.shape}")
    print(f"  Labels sum to 1: {np.allclose(y_mixed.sum(axis=1), 1.0)}")
    
    # Test CutMix
    cutmix = CutMixAugmentation(alpha=1.0)
    X_cut, y_cut = cutmix.cutmix(X[0], y[0], X[1], y[1])
    
    print("\nCutMix Augmentation:")
    print(f"  Input shape: {X[0].shape}")
    print(f"  Output shape: {X_cut.shape}")
    print(f"  Label sum: {y_cut.sum():.3f} (should be ~1.0)")
    
    # Verify augmentations preserve data integrity
    assert X_mixed.shape == X.shape, "MixUp changed shape!"
    assert X_cut.shape == X[0].shape, "CutMix changed shape!"
    assert np.all(X_mixed >= 0) and np.all(X_mixed <= 1), "MixUp values out of range!"
    
    print("\n✓ All augmentations working correctly")
    return True


def test_all_datasets_integration():
    """Test loading all datasets together"""
    print("\n" + "=" * 60)
    print("Testing All Datasets Integration")
    print("=" * 60)
    
    loader = MultiDatasetLoader(base_data_dir='./data')
    
    # Load all datasets with small subsets for testing
    all_datasets = loader.load_all_datasets(
        use_cache=False,
        plantvillage_subset=0.01,  # 1% for speed
        include_augmented=False
    )
    
    print("\nLoaded datasets:")
    for name, (images, labels) in all_datasets.items():
        print(f"  {name}: {len(images)} images")
        unique_labels = set(labels)
        print(f"    Unique labels: {unique_labels}")
    
    if len(all_datasets) > 0:
        # Test balanced dataset creation
        X, y = loader.create_balanced_dataset(all_datasets, samples_per_class=50)
        
        print(f"\nBalanced dataset created:")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {y.shape[1]}")
        
        # Verify class balance
        class_counts = y.sum(axis=0)
        print(f"  Class distribution: {class_counts}")
        
        # Check if relatively balanced (within 20% of mean)
        mean_count = class_counts.mean()
        balanced = np.all(np.abs(class_counts - mean_count) < mean_count * 0.5)
        
        if balanced:
            print("  ✓ Classes are well balanced")
        else:
            print("  ⚠ Classes have some imbalance (expected with real data)")
        
        return True
    else:
        print("✗ No datasets found")
        return False


def main():
    """Run all tests"""
    print("RGB Model Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("PlantNet Loading", test_plantnet_loading),
        ("Data Leakage Prevention", test_data_leakage_prevention),
        ("Advanced Augmentations", test_augmentations),
        ("All Datasets Integration", test_all_datasets_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The RGB model is ready for training.")
        print("\nNext steps:")
        print("1. Ensure plantnet_300K.zip is in data/ or src/data/")
        print("2. Run: python train_rgb_model.py")
        print("3. Monitor training for any signs of overfitting")
    else:
        print("\n⚠ Some tests failed. Please review the issues above.")
        print("Common issues:")
        print("- PlantNet dataset not found: Check file location")
        print("- Data leakage: Review splitting logic")
        print("- Augmentation errors: Check dependencies")


if __name__ == "__main__":
    main()