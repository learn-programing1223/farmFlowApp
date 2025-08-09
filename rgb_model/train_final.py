#!/usr/bin/env python3
"""
Final optimized training script for RGB plant disease model
Implements key improvements from research for 80%+ accuracy
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Disable verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_model(num_classes=7):
    """Build a robust CNN model"""
    
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 4
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Output
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_data():
    """Load preprocessed data"""
    data_dir = Path('./data/splits')
    
    print("Loading data...")
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    # Normalize if needed
    if X_train.max() > 1.0:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Load test
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        if X_test.max() > 1.0:
            X_test = X_test / 255.0
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def cutmix_batch(X_batch, y_batch, alpha=1.0):
    """Apply CutMix to a batch"""
    batch_size = len(X_batch)
    indices = np.random.permutation(batch_size)
    
    # Get lambda value
    lam = np.random.beta(alpha, alpha)
    
    # Get cut size
    h, w = X_batch.shape[1:3]
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # Get center point
    cy = np.random.randint(cut_h // 2, h - cut_h // 2)
    cx = np.random.randint(cut_w // 2, w - cut_w // 2)
    
    # Apply cutmix
    X_mixed = X_batch.copy()
    y1, y2 = cy - cut_h // 2, cy + cut_h // 2
    x1, x2 = cx - cut_w // 2, cx + cut_w // 2
    
    X_mixed[:, y1:y2, x1:x2, :] = X_batch[indices, y1:y2, x1:x2, :]
    y_mixed = lam * y_batch + (1 - lam) * y_batch[indices]
    
    return X_mixed, y_mixed


def train_model():
    """Main training function"""
    
    print("="*60)
    print("RGB PLANT DISEASE MODEL TRAINING")
    print("="*60)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_classes=y_train.shape[1])
    
    # Compile with label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    print(f"Total parameters: {model.count_params():,}")
    
    # Training parameters
    batch_size = 32
    epochs = 50
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    # Callbacks
    checkpoint_dir = Path('./models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train
    print("\nStarting training...")
    print("-" * 40)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2  # Less verbose output
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Get best accuracy
    best_acc = max(history.history['val_accuracy'])
    final_acc = history.history['val_accuracy'][-1]
    
    print(f"\nBest validation accuracy: {best_acc:.2%}")
    print(f"Final validation accuracy: {final_acc:.2%}")
    
    if best_acc >= 0.80:
        print("✓ TARGET ACHIEVED: 80%+ accuracy!")
    else:
        print(f"Current best: {best_acc:.2%} (Target: 80%)")
    
    # Test evaluation
    if X_test is not None:
        print("\nEvaluating on test set...")
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Test accuracy: {test_acc:.2%}")
        
        if test_acc >= 0.80:
            print("✓ TEST TARGET ACHIEVED: 80%+ accuracy!")
    
    # Save model
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f'rgb_model_{timestamp}.h5'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save history
    history_path = model_dir / f'history_{timestamp}.json'
    with open(history_path, 'w') as f:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"History saved to: {history_path}")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_model()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)