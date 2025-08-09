#!/usr/bin/env python3
"""
Diagnostic script to identify why the model is not learning
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def diagnose_data():
    """Check the data for issues"""
    print("=" * 60)
    print("DATA DIAGNOSTICS")
    print("=" * 60)
    
    # Load data
    data_dir = Path('./data/splits')
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"\n1. DATA SHAPES:")
    print(f"   X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"   y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"   X_val: {X_val.shape}, dtype: {X_val.dtype}")
    print(f"   y_val: {y_val.shape}, dtype: {y_val.dtype}")
    
    print(f"\n2. PIXEL VALUE RANGES:")
    print(f"   X_train min: {X_train.min():.3f}, max: {X_train.max():.3f}")
    print(f"   X_train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
    
    print(f"\n3. LABEL DISTRIBUTION:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"   Training labels: {dict(zip(unique_train, counts_train))}")
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print(f"   Validation labels: {dict(zip(unique_val, counts_val))}")
    
    # Check if labels are one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"\n   Labels are one-hot encoded: shape {y_train.shape}")
        # Convert to class indices for analysis
        y_train_idx = np.argmax(y_train, axis=1)
        y_val_idx = np.argmax(y_val, axis=1)
        unique_train, counts_train = np.unique(y_train_idx, return_counts=True)
        print(f"   Actual class distribution (train): {dict(zip(unique_train, counts_train))}")
        unique_val, counts_val = np.unique(y_val_idx, return_counts=True)
        print(f"   Actual class distribution (val): {dict(zip(unique_val, counts_val))}")
    
    print(f"\n4. CLASS BALANCE:")
    if len(counts_train) > 0:
        imbalance_ratio = max(counts_train) / min(counts_train)
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 10:
            print("   WARNING: Severe class imbalance detected!")
    
    # Check for NaN or Inf values
    print(f"\n5. DATA INTEGRITY:")
    print(f"   NaN in X_train: {np.isnan(X_train).any()}")
    print(f"   Inf in X_train: {np.isinf(X_train).any()}")
    print(f"   NaN in y_train: {np.isnan(y_train).any()}")
    
    # Sample some images to check
    print(f"\n6. SAMPLE IMAGE CHECK:")
    for i in range(min(3, len(X_train))):
        img = X_train[i]
        print(f"   Image {i}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}, mean={img.mean():.3f}")
    
    return X_train, y_train, X_val, y_val

def test_simple_model(X_train, y_train, X_val, y_val):
    """Test with a very simple model to isolate issues"""
    print("\n" + "=" * 60)
    print("SIMPLE MODEL TEST")
    print("=" * 60)
    
    # Determine number of classes
    if len(y_train.shape) > 1:
        num_classes = y_train.shape[1]
        y_train_labels = np.argmax(y_train, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
    else:
        num_classes = len(np.unique(y_train))
        y_train_labels = y_train
        y_val_labels = y_val
    
    print(f"\nNumber of classes: {num_classes}")
    
    # Create a very simple CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if len(y_train.shape) == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining simple model for 5 epochs...")
    history = model.fit(
        X_train[:100], y_train[:100],  # Use small subset for quick test
        validation_data=(X_val[:50], y_val[:50]),
        epochs=5,
        batch_size=8,
        verbose=1
    )
    
    print(f"\nSimple model results:")
    print(f"  Final train accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.3f}")
    
    # Check predictions
    predictions = model.predict(X_val[:10])
    pred_classes = np.argmax(predictions, axis=1)
    print(f"\nSample predictions: {pred_classes}")
    print(f"Unique predicted classes: {np.unique(pred_classes)}")
    
    return model, history

def check_model_outputs():
    """Check what the current model is actually outputting"""
    print("\n" + "=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    from model_fixed import build_fixed_model
    
    # Build model
    model = build_fixed_model(num_classes=7, input_shape=(224, 224, 3))
    
    # Create random input
    dummy_input = np.random.randn(10, 224, 224, 3).astype(np.float32)
    
    # Get raw outputs (before softmax)
    outputs = model(dummy_input, training=False)
    
    print(f"\nRaw model outputs (logits):")
    print(f"  Shape: {outputs.shape}")
    print(f"  Sample output[0]: {outputs[0].numpy()}")
    print(f"  Output range: [{outputs.numpy().min():.3f}, {outputs.numpy().max():.3f}]")
    
    # Apply softmax to get probabilities
    probs = tf.nn.softmax(outputs)
    print(f"\nAfter softmax:")
    print(f"  Sample probs[0]: {probs[0].numpy()}")
    print(f"  Sum of probs[0]: {probs[0].numpy().sum():.3f}")
    
    # Check which class is being predicted
    pred_classes = np.argmax(probs, axis=1)
    print(f"\nPredicted classes: {pred_classes}")
    print(f"Unique predictions: {np.unique(pred_classes)}")

def suggest_fixes():
    """Suggest potential fixes based on diagnostics"""
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES")
    print("=" * 60)
    
    print("""
1. DATA PREPROCESSING:
   - Ensure images are normalized to [0, 1] range
   - Check that augmentation is not too aggressive
   - Verify labels are correctly mapped

2. MODEL ARCHITECTURE:
   - Start with unfrozen base model (fine-tuning)
   - Use lower learning rate (1e-4 or 1e-5)
   - Remove focal loss initially, use standard categorical_crossentropy

3. TRAINING STRATEGY:
   - Use class weights for imbalanced data
   - Start with smaller subset to verify learning
   - Monitor gradient flow

4. DEBUGGING STEPS:
   - Train on just 2 classes first
   - Use pre-trained ImageNet weights properly
   - Check that data generator is shuffling

5. ALTERNATIVE APPROACHES:
   - Try ResNet50 or MobileNetV2 instead of EfficientNet
   - Use transfer learning from PlantNet or similar plant models
   - Consider using PyTorch with proven implementations
    """)

if __name__ == "__main__":
    print("PLANT DISEASE MODEL DIAGNOSTIC TOOL")
    print("====================================\n")
    
    # Run diagnostics
    X_train, y_train, X_val, y_val = diagnose_data()
    
    # Test simple model
    model, history = test_simple_model(X_train, y_train, X_val, y_val)
    
    # Check model outputs
    check_model_outputs()
    
    # Suggest fixes
    suggest_fixes()
    
    print("\nDiagnostics complete!")