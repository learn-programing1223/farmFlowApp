#!/usr/bin/env python3
"""
Improved training script - fixes the learning rate issue
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json

print("=" * 60)
print("IMPROVED TRAINING - FIXED VERSION")
print("=" * 60)

# Configuration
config = {
    'batch_size': 16,
    'epochs': 20,  # Reduced for faster iteration
    'learning_rate': 0.001,  # Fixed learning rate
    'input_shape': (224, 224, 3),
    'num_classes': 7
}

# Check for existing training data
train_dir = Path('datasets/master_field_dataset/train')
val_dir = Path('datasets/master_field_dataset/val')

if not train_dir.exists():
    print("Error: No training data found!")
    print("Please generate data first")
    exit(1)

# Count images
train_count = sum(len(list(d.glob('*.jpg'))) for d in train_dir.iterdir() if d.is_dir())
val_count = sum(len(list(d.glob('*.jpg'))) for d in val_dir.iterdir() if d.is_dir())
print(f"\nDataset: {train_count} train, {val_count} val images")

# Create data generators with heavy augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.3, 1.7],
    channel_shift_range=50,
    fill_mode='reflect'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=False
)

print(f"Classes: {list(train_generator.class_indices.keys())}")

# Build a simpler, more trainable model
def build_simple_cnn():
    """Build a simple CNN that can actually learn from synthetic data"""
    model = keras.Sequential([
        # Input normalization
        layers.Input(shape=config['input_shape']),
        
        # Conv Block 1 - fewer filters
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        
        # Conv Block 2
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        
        # Conv Block 3
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        # Conv Block 4
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(config['num_classes'], activation='softmax')
    ])
    
    return model

print("\nBuilding model...")
model = build_simple_cnn()
print(f"Total parameters: {model.count_params():,}")

# Compile with fixed learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/improved_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Train
print("\nStarting training...")
print("-" * 60)

history = model.fit(
    train_generator,
    epochs=config['epochs'],
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
print("\nSaving models...")
model.save('models/improved_final.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/improved_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model: {len(tflite_model)/1024/1024:.2f} MB")

# Get results
best_acc = max(history.history['val_accuracy'])
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best validation accuracy: {best_acc:.2%}")

if best_acc < 0.5:
    print("\nModel performance is low.")
    print("Synthetic data alone is insufficient.")
    print("\nNext steps:")
    print("1. Download real PlantVillage dataset")
    print("2. Use transfer learning from pre-trained model")
    print("3. Train for more epochs with real data")
else:
    print("\nModel shows promise!")
    print("Test with: python test_real_world_simple.py")