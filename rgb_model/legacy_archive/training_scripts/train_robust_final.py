#!/usr/bin/env python3
"""
Final robust training script - trains on synthetic + real data
Uses transfer learning for best results
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("="*60)
print("ROBUST PLANT DISEASE MODEL TRAINING")
print("="*60)

# Configuration
config = {
    'model_type': 'efficientnet',  # Using EfficientNet for better performance
    'batch_size': 16,  # Smaller batch size to avoid memory issues
    'epochs': 30,  # Reduced epochs for faster training
    'learning_rate': 1e-3,
    'input_shape': (224, 224, 3),
    'num_classes': 7
}

print("\nConfiguration:")
print(json.dumps(config, indent=2))

# Check GPU
print("\nChecking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found, using CPU")

# Data directories
train_dir = Path('datasets/master_field_dataset/train')
val_dir = Path('datasets/master_field_dataset/val')

if not train_dir.exists():
    print(f"\nError: Training data not found at {train_dir}")
    print("Please run generate_training_data.py first")
    exit(1)

# Count images
train_count = sum(len(list(d.glob('*.jpg'))) for d in train_dir.iterdir() if d.is_dir())
val_count = sum(len(list(d.glob('*.jpg'))) for d in val_dir.iterdir() if d.is_dir())
print(f"\nDataset size:")
print(f"  Training: {train_count} images")
print(f"  Validation: {val_count} images")

# Create data augmentation
print("\nSetting up data augmentation...")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='reflect'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Load data
print("\nLoading training data...")
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

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"\nClasses: {class_names}")

# Build model
print("\nBuilding model...")

def build_efficientnet_model():
    """Build EfficientNet-based model"""
    
    # Use EfficientNetB0 as base (smaller and faster)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=config['input_shape'],
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=config['input_shape'])
    
    # Preprocess for EfficientNet
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(config['num_classes'], activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def build_custom_cnn():
    """Build custom CNN (fallback if EfficientNet fails)"""
    
    model = keras.Sequential([
        # Input normalization
        layers.Lambda(lambda x: x * 2.0 - 1.0, input_shape=config['input_shape']),
        
        # Conv Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Conv Block 4
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(config['num_classes'], activation='softmax')
    ])
    
    return model, None

# Try to build EfficientNet, fallback to custom CNN
try:
    model, base_model = build_efficientnet_model()
    print("Using EfficientNet architecture")
except Exception as e:
    print(f"EfficientNet failed: {e}")
    print("Using custom CNN architecture")
    model, base_model = build_custom_cnn()

print(f"Total parameters: {model.count_params():,}")

# Compile model
print("\nCompiling model...")

# Learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=config['learning_rate'],
    decay_steps=100,
    decay_rate=0.96
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/robust_final_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

# Train model
print("\nStarting training...")
print("-"*60)

history = model.fit(
    train_generator,
    epochs=config['epochs'],
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Fine-tune if using transfer learning
if base_model and config['epochs'] > 10:
    print("\nFine-tuning base model layers...")
    
    # Unfreeze last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Continue training
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

# Save final model
print("\nSaving models...")
model.save('models/robust_final_model.h5')

# Convert to TFLite
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'models/robust_final_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")

# Get final metrics
final_val_acc = max(history.history['val_accuracy'])
final_val_loss = min(history.history['val_loss'])

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best validation accuracy: {final_val_acc:.2%}")
print(f"Best validation loss: {final_val_loss:.4f}")
print("\nModels saved:")
print("  - models/robust_final_best.h5 (best checkpoint)")
print("  - models/robust_final_model.h5 (final model)")
print("  - models/robust_final_model.tflite (mobile deployment)")

# Save training history
history_data = {
    'config': config,
    'class_names': class_names,
    'final_accuracy': float(final_val_acc),
    'final_loss': float(final_val_loss),
    'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
}

with open('models/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)

print("\nNext steps:")
print("1. Test on real images: python evaluate_real_world_performance.py")
print("2. Deploy to app: cp models/robust_final_model.tflite ../PlantPulse/assets/models/")
print("3. Update web app to use new model")

print("\n✓ Training pipeline complete!")