#!/usr/bin/env python3
"""
Enhanced CNN for Plant Disease Detection
Building on the simple CNN that achieved 63% accuracy
Goal: Achieve 85%+ accuracy with proper architecture and training
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard
)
from pathlib import Path
import json
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def build_enhanced_cnn(num_classes, input_shape=(224, 224, 3)):
    """
    Enhanced CNN architecture based on the simple model that worked
    Gradually increases complexity while maintaining trainability
    """
    print("[BUILD] Creating Enhanced CNN Model...")
    
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Data Augmentation (critical for generalization)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        
        # Block 1: 32 filters
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 2: 64 filters
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 3: 128 filters
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 4: 256 filters (deeper feature extraction)
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # Better than Flatten
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"[BUILD] Model created with {model.count_params():,} parameters")
    return model

def create_lightweight_cnn(num_classes, input_shape=(224, 224, 3)):
    """
    Lightweight version if enhanced is too heavy
    """
    print("[BUILD] Creating Lightweight CNN Model...")
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Light augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        # Simpler architecture
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"[BUILD] Lightweight model with {model.count_params():,} parameters")
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile with proven optimizer settings
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def create_data_pipeline(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create efficient data pipeline with caching and prefetching
    """
    print("[DATA] Creating optimized data pipeline...")
    
    # Convert to TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Optimize pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = (train_dataset
        .shuffle(buffer_size=1000)
        .batch(batch_size)
        .cache()  # Cache in memory after first epoch
        .prefetch(buffer_size=AUTOTUNE))
    
    val_dataset = (val_dataset
        .batch(batch_size)
        .cache()
        .prefetch(buffer_size=AUTOTUNE))
    
    return train_dataset, val_dataset

def train_model(args):
    """
    Main training function with smart strategies
    """
    print("\n" + "="*60)
    print("ENHANCED CNN TRAINING FOR PLANT DISEASE DETECTION")
    print("Building on successful simple CNN approach")
    print("="*60)
    
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[CPU] No GPU found, using CPU")
        print("[WARNING] Training will be slower on CPU")
    
    # Load data
    print("\n[DATA] Loading preprocessed data...")
    data_dir = Path('./data/splits')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    # Load test data
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
    
    num_classes = y_train.shape[1]
    
    print(f"[DATA] Training: {len(X_train)} samples")
    print(f"[DATA] Validation: {len(X_val)} samples")
    print(f"[DATA] Classes: {num_classes}")
    
    # Choose model based on dataset size
    if len(X_train) < 5000:
        model = create_lightweight_cnn(num_classes)
    else:
        model = build_enhanced_cnn(num_classes)
    
    # Compile
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Create data pipeline
    train_dataset, val_dataset = create_data_pipeline(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size
    )
    
    # Setup callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping with patience
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard for monitoring
        TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=1
        )
    ]
    
    # Training strategy
    print("\n[TRAIN] Starting training with smart strategy...")
    print("-"*60)
    print("Strategy:")
    print("  1. Train with augmentation")
    print("  2. Early stopping when validation plateaus")
    print("  3. Reduce LR when stuck")
    print("  4. Save best model automatically")
    print("-"*60)
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[EVAL] Evaluating model...")
    
    # Validation metrics
    val_results = model.evaluate(val_dataset, verbose=0)
    val_loss, val_acc, val_prec, val_rec, val_auc = val_results
    
    print(f"\n[VALIDATION] Final Metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  AUC:       {val_auc:.4f}")
    
    # Test metrics if available
    if X_test is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(args.batch_size)
        
        test_results = model.evaluate(test_dataset, verbose=0)
        test_loss, test_acc, test_prec, test_rec, test_auc = test_results
        
        print(f"\n[TEST] Test Set Metrics:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")
        print(f"  AUC:       {test_auc:.4f}")
    
    # Save final model
    final_path = output_dir / 'final_model.h5'
    model.save(final_path)
    print(f"\n[SAVE] Model saved to: {final_path}")
    
    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    
    # Performance analysis
    print("\n" + "="*60)
    if val_acc >= 0.85:
        print("[SUCCESS] EXCELLENT! Target accuracy achieved!")
        if val_acc >= 0.95:
            print("[OUTSTANDING] Near-perfect accuracy!")
        elif val_acc >= 0.90:
            print("[GREAT] Very high accuracy achieved!")
    elif val_acc >= 0.75:
        print("[GOOD] Good accuracy - consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Adding more augmentation")
    elif val_acc >= 0.60:
        print("[PROGRESS] Model is learning well")
        print("Continue training or try adjustments")
    else:
        print("[INFO] Accuracy needs improvement")
        print("Consider architectural changes")
    print("="*60)
    
    # Training insights
    if len(history.history['loss']) < args.epochs:
        print(f"\n[INFO] Training stopped early at epoch {len(history.history['loss'])}")
        print("Model converged or validation stopped improving")
    
    # Check for overfitting
    if val_acc < history.history['accuracy'][-1] - 0.1:
        print("\n[WARNING] Possible overfitting detected")
        print("Consider more dropout or regularization")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced CNN')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--output-dir', type=str, default='./models/enhanced_cnn',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("[CONFIG] Training Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Train
    model, history = train_model(args)

if __name__ == "__main__":
    main()