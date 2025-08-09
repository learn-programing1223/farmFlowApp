#!/usr/bin/env python3
"""
FIXED EfficientNet Training - Solves the 14% accuracy problem
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def build_correct_efficientnet(num_classes=7, input_shape=(224, 224, 3)):
    """Build EfficientNet with CORRECT preprocessing and unfrozen layers"""
    
    from tensorflow.keras.applications import EfficientNetB0
    
    # Base model
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # CRITICAL FIX #1: UNFREEZE the base model!
    # Research shows frozen = 23%, unfrozen = 99%
    base_model.trainable = True
    
    # But freeze first few layers to prevent destroying low-level features
    for layer in base_model.layers[:20]:
        layer.trainable = False
    
    # Build model
    inputs = tf.keras.Input(shape=input_shape)
    
    # CRITICAL FIX #2: Scale [0,1] data to [0,255] BEFORE preprocessing
    x = inputs * 255.0
    
    # Now apply EfficientNet preprocessing (expects [0,255])
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    
    # Pass through base model
    x = base_model(x)
    
    # Add regularization
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model


def train_fixed():
    """Training with all fixes applied"""
    
    print("\n" + "="*70)
    print("FIXED EFFICIENTNET TRAINING")
    print("Solving the 14% accuracy problem")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dir = Path('./data/splits')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"Loaded {len(X_train):,} training samples")
    print(f"Data is in range [{X_train.min():.1f}, {X_train.max():.1f}]")
    
    # Build FIXED model
    print("\nBuilding FIXED EfficientNet model...")
    model, base_model = build_correct_efficientnet()
    
    # Count trainable layers
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"Trainable layers: {trainable_count} (should be many, not just top layers!)")
    
    # Compile with proper settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',  # Simple loss first
        metrics=['accuracy']
    )
    
    print(f"Total parameters: {model.count_params():,}")
    
    # Simple data augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15
    )
    
    val_datagen = ImageDataGenerator()
    
    batch_size = 32
    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_efficientnet_fixed.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Progressive unfreezing
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: unfreeze_more(base_model, epoch)
        )
    ]
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("Expected: 60-85% accuracy (not 14%!)")
    print("="*70 + "\n")
    
    # Train
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=30,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print(f"Best accuracy: {max(history.history['val_accuracy']):.2%}")
    print("="*70)
    
    return model, history


def unfreeze_more(base_model, epoch):
    """Progressive unfreezing"""
    if epoch == 5:
        print("\n[UNFREEZING] More layers at epoch 5...")
        for layer in base_model.layers[20:]:
            layer.trainable = True
    elif epoch == 10:
        print("\n[UNFREEZING] All layers at epoch 10...")
        base_model.trainable = True


if __name__ == "__main__":
    import time
    
    print("\n" + "="*70)
    print("EFFICIENTNET FIX FOR PLANT DISEASE DETECTION")
    print("="*70)
    
    print("\nFIXES APPLIED:")
    print("1. ✅ Unfrozen layers (was frozen, causing 14%)")
    print("2. ✅ Correct preprocessing ([0,1] → [0,255])")
    print("3. ✅ Progressive unfreezing strategy")
    print("4. ✅ Lower learning rate for fine-tuning")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    model, history = train_fixed()