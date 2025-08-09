#!/usr/bin/env python3
"""
Test script for the fixed model implementation
Verifies that the model converges properly with corrected Focal Loss
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_fixed import build_fixed_model, compile_fixed_model, FocalLoss


def test_model_convergence():
    """Test that the fixed model actually learns"""
    
    print("="*60)
    print("Testing Fixed Model Convergence")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create synthetic data (should converge quickly on synthetic data)
    n_samples = 1000
    n_classes = 7
    
    print("\n1. Creating synthetic dataset...")
    # Create simple synthetic data that should be easily learnable
    X_train = np.random.randn(n_samples, 224, 224, 3).astype(np.float32)
    
    # Create labels with some pattern (not random)
    y_train = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        # Simple pattern: class based on sum of first pixel values
        pixel_sum = X_train[i, 0, 0, :].sum()
        class_idx = int(abs(pixel_sum) * n_classes / 10) % n_classes
        y_train[i, class_idx] = 1
    
    # Add some noise to make it more realistic
    X_train += np.random.randn(*X_train.shape) * 0.1
    
    # Create validation data
    X_val = np.random.randn(200, 224, 224, 3).astype(np.float32)
    y_val = np.zeros((200, n_classes))
    for i in range(200):
        pixel_sum = X_val[i, 0, 0, :].sum()
        class_idx = int(abs(pixel_sum) * n_classes / 10) % n_classes
        y_val[i, class_idx] = 1
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Classes: {n_classes}")
    
    # Build and compile model
    print("\n2. Building fixed model...")
    model, base_model = build_fixed_model(num_classes=n_classes)
    
    print("\n3. Compiling with fixed Focal Loss...")
    model = compile_fixed_model(model, learning_rate=0.001, use_focal_loss=True)
    
    # Test initial predictions
    print("\n4. Testing initial predictions...")
    initial_pred = model.predict(X_val[:5], verbose=0)
    print(f"  Initial prediction shape: {initial_pred.shape}")
    print(f"  Initial max probability: {initial_pred.max():.3f}")
    
    # Train for a few epochs
    print("\n5. Training for 5 epochs...")
    print("-"*40)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    print("-"*40)
    
    # Check if model is learning
    print("\n6. Checking training results...")
    
    initial_loss = history.history['loss'][0]
    final_loss = history.history['loss'][-1]
    initial_acc = history.history['accuracy'][0]
    final_acc = history.history['accuracy'][-1]
    
    print(f"\n  Training Loss: {initial_loss:.4f} → {final_loss:.4f}")
    print(f"  Training Accuracy: {initial_acc:.3f} → {final_acc:.3f}")
    
    if len(history.history.get('val_loss', [])) > 0:
        val_initial_loss = history.history['val_loss'][0]
        val_final_loss = history.history['val_loss'][-1]
        val_initial_acc = history.history['val_accuracy'][0]
        val_final_acc = history.history['val_accuracy'][-1]
        print(f"  Validation Loss: {val_initial_loss:.4f} → {val_final_loss:.4f}")
        print(f"  Validation Accuracy: {val_initial_acc:.3f} → {val_final_acc:.3f}")
    
    # Test if model improved
    loss_improved = final_loss < initial_loss * 0.9  # At least 10% improvement
    acc_improved = final_acc > initial_acc + 0.1  # At least 0.1 improvement
    
    print("\n7. Model Assessment:")
    if loss_improved and acc_improved:
        print("  ✅ Model is learning properly!")
        print("  ✅ Loss decreased significantly")
        print("  ✅ Accuracy improved")
    elif loss_improved:
        print("  ⚠️ Loss decreased but accuracy didn't improve much")
    elif acc_improved:
        print("  ⚠️ Accuracy improved but loss didn't decrease much")
    else:
        print("  ❌ Model is NOT learning properly")
        print("  ❌ Consider checking the implementation")
    
    # Test with standard crossentropy for comparison
    print("\n8. Testing with standard CrossEntropy for comparison...")
    model2, _ = build_fixed_model(num_classes=n_classes)
    model2 = compile_fixed_model(model2, learning_rate=0.001, use_focal_loss=False)
    
    history2 = model2.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    ce_final_loss = history2.history['loss'][-1]
    ce_final_acc = history2.history['accuracy'][-1]
    
    print(f"  CrossEntropy Final Loss: {ce_final_loss:.4f}")
    print(f"  CrossEntropy Final Accuracy: {ce_final_acc:.3f}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    return history


def test_with_real_data():
    """Test with actual loaded data if available"""
    
    print("\n" + "="*60)
    print("Testing with Real Data (if available)")
    print("="*60)
    
    # Check if preprocessed data exists
    splits_dir = Path('./data/splits')
    if not splits_dir.exists():
        print("\n⚠️ No preprocessed data found.")
        print("Run train_disease_focused.py first to generate data.")
        return
    
    # Load preprocessed data
    try:
        print("\nLoading preprocessed data...")
        X_train = np.load(splits_dir / 'X_train.npy')
        y_train = np.load(splits_dir / 'y_train.npy')
        X_val = np.load(splits_dir / 'X_val.npy')
        y_val = np.load(splits_dir / 'y_val.npy')
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Classes: {y_train.shape[1]}")
        
        # Use subset for quick test
        subset_size = min(500, len(X_train))
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        
        print(f"\nUsing subset of {subset_size} samples for quick test")
        
        # Build model
        model, base_model = build_fixed_model(num_classes=y_train.shape[1])
        model = compile_fixed_model(model, learning_rate=0.001, use_focal_loss=True)
        
        # Train for a few epochs
        print("\nTraining for 3 epochs on real data...")
        history = model.fit(
            X_train_subset, y_train_subset,
            validation_data=(X_val[:100], y_val[:100]),
            epochs=3,
            batch_size=16,
            verbose=1
        )
        
        # Check results
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        initial_acc = history.history['accuracy'][0]
        final_acc = history.history['accuracy'][-1]
        
        print(f"\nResults on Real Data:")
        print(f"  Loss: {initial_loss:.4f} → {final_loss:.4f}")
        print(f"  Accuracy: {initial_acc:.3f} → {final_acc:.3f}")
        
        if final_loss < initial_loss and final_acc > initial_acc:
            print("  ✅ Model learns on real data!")
        else:
            print("  ⚠️ Model may need more epochs or tuning")
            
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print("Please run setup and training scripts first.")


if __name__ == "__main__":
    # Test synthetic data convergence
    test_model_convergence()
    
    # Test with real data if available
    test_with_real_data()
    
    print("\n✨ All tests complete!")