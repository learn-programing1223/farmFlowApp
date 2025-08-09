#!/usr/bin/env python3
"""
Quick test to verify training is working correctly
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quick_test():
    """Quick test with 3 epochs"""
    
    print("="*60)
    print("QUICK TRAINING TEST (3 epochs)")
    print("="*60)
    
    # Load data
    data_dir = Path('./data/splits')
    X_train = np.load(data_dir / 'X_train.npy')[:1000]  # Use subset
    y_train = np.load(data_dir / 'y_train.npy')[:1000]
    X_val = np.load(data_dir / 'X_val.npy')[:500]
    y_val = np.load(data_dir / 'y_val.npy')[:500]
    
    # Normalize
    if X_train.max() > 1.0:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
    
    print(f"Using subset - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Train for 3 epochs
    print("\nTraining for 3 epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    # Results
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print("="*60)
    
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final training accuracy: {train_acc:.2%}")
    print(f"Final validation accuracy: {val_acc:.2%}")
    
    if val_acc > 0.15:  # Better than random (1/7 = 14.3%)
        print("✓ Model is learning!")
    else:
        print("⚠ Model may have issues")
    
    # Check if accuracy is improving
    if len(history.history['val_accuracy']) > 1:
        improvement = history.history['val_accuracy'][-1] - history.history['val_accuracy'][0]
        print(f"Validation accuracy improvement: {improvement:.2%}")
        
        if improvement > 0:
            print("✓ Model is improving over epochs!")
    
    return history

if __name__ == "__main__":
    try:
        history = quick_test()
        print("\n✓ Training test completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()