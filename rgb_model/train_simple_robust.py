#!/usr/bin/env python3
"""
Simplified robust training script for PlantVillage dataset
Fixed version without shape mismatches
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("=" * 70)
print("ROBUST PLANTVILLAGE MODEL TRAINING")
print("=" * 70)

# Configuration
config = {
    'input_shape': (224, 224, 3),
    'num_classes': 6,  # 6 categories (no Rust in PlantVillage)
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'data_path': 'datasets/plantvillage_processed'
}

print("\nConfiguration:")
print(json.dumps(config, indent=2))

# Check GPU
print("\n" + "-" * 70)
print("Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[OK] GPU Available: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("[WARNING] No GPU found, using CPU (training will be slower)")

# Setup data paths
data_path = Path(config['data_path'])
train_path = data_path / 'train'
val_path = data_path / 'val'
test_path = data_path / 'test'

if not train_path.exists():
    print(f"\n[ERROR] Training data not found at {train_path}")
    print("Please run prepare_plantvillage_data.py first")
    exit(1)

# Calculate class weights for imbalanced data
print("\n" + "-" * 70)
print("Calculating class weights...")

class_counts = {}
for class_dir in train_path.iterdir():
    if class_dir.is_dir():
        count = len(list(class_dir.glob('*.jpg')))
        class_counts[class_dir.name] = count
        print(f"  {class_dir.name}: {count} images")

# Calculate weights (inverse of frequency)
# Filter out classes with zero samples
non_zero_classes = {k: v for k, v in class_counts.items() if v > 0}
total_samples = sum(non_zero_classes.values())
num_classes = len(non_zero_classes)
class_weights = {}

for idx, (class_name, count) in enumerate(sorted(non_zero_classes.items())):
    weight = total_samples / (num_classes * count)
    class_weights[idx] = weight
    print(f"  Weight for {class_name}: {weight:.3f}")

# Data generators with augmentation
print("\n" + "-" * 70)
print("Setting up data generators...")

# Training data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Validation/Test data - only rescaling
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=False
)

print(f"\n[OK] Training samples: {train_generator.samples}")
print(f"[OK] Validation samples: {val_generator.samples}")
print(f"[OK] Test samples: {test_generator.samples}")
print(f"[OK] Classes: {list(train_generator.class_indices.keys())}")

# Build model with transfer learning
print("\n" + "-" * 70)
print("Building EfficientNetB0 model...")

def create_model():
    """Create EfficientNetB0 model with proper input shape"""
    
    # Base model - EfficientNetB0 for mobile deployment
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=config['input_shape'],
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=config['input_shape'])
    
    # Preprocessing for EfficientNet (normalize to [-1, 1])
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

# Create model
model, base_model = create_model()
print(f"[OK] Model created with {model.count_params():,} parameters")
print(f"  Trainable: {sum(1 for w in model.trainable_weights):,} parameters")

# Compile model
print("\n" + "-" * 70)
print("Compiling model...")

# Use Adam optimizer with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
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
    # Save best model
    keras.callbacks.ModelCheckpoint(
        'models/plantvillage_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Tensorboard logs
    keras.callbacks.TensorBoard(
        log_dir=f'logs/plantvillage_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=1
    )
]

# Training Phase 1: Train with frozen base
print("\n" + "=" * 70)
print("PHASE 1: Training with frozen base model...")
print("=" * 70)

history_frozen = model.fit(
    train_generator,
    epochs=10,  # First 10 epochs with frozen base
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tuning
print("\n" + "=" * 70)
print("PHASE 2: Fine-tuning the model...")
print("=" * 70)

# Unfreeze the top layers of base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - 20

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"[OK] Unfreezing top {len(base_model.layers) - fine_tune_at} layers for fine-tuning")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=config['epochs'] - 10,  # Remaining epochs
    initial_epoch=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

test_results = model.evaluate(test_generator, verbose=1)
print(f"\nTest Results:")
print(f"  Loss: {test_results[0]:.4f}")
print(f"  Accuracy: {test_results[1]:.2%}")
print(f"  Precision: {test_results[2]:.2%}")
print(f"  Recall: {test_results[3]:.2%}")

# Save final model
print("\n" + "-" * 70)
print("Saving models...")

# Save Keras model
model.save('models/plantvillage_final.h5')
print("[OK] Saved Keras model: models/plantvillage_final.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'models/plantvillage_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"[OK] Saved TFLite model: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")

# Save training history
history_data = {
    'config': config,
    'class_weights': class_weights,
    'test_results': {
        'loss': float(test_results[0]),
        'accuracy': float(test_results[1]),
        'precision': float(test_results[2]),
        'recall': float(test_results[3])
    },
    'training_history': {
        'frozen_phase': history_frozen.history,
        'fine_tuning_phase': history_fine.history if 'history_fine' in locals() else {}
    }
}

with open('models/plantvillage_history.json', 'w') as f:
    json.dump(history_data, f, indent=2, default=str)

print("[OK] Saved training history: models/plantvillage_history.json")

# Final summary
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\n>>> Final Test Accuracy: {test_results[1]:.2%}")

if test_results[1] >= 0.85:
    print("[SUCCESS] TARGET ACHIEVED! Model exceeds 85% accuracy!")
else:
    print(f"[WARNING] Model accuracy is {test_results[1]:.2%}, target is 85%")
    print("Consider: More epochs, different augmentation, or additional data")

print("\n>>> Model files created:")
print("  - models/plantvillage_best.h5 (best checkpoint)")
print("  - models/plantvillage_final.h5 (final model)")
print(f"  - models/plantvillage_model.tflite ({len(tflite_model)/1024/1024:.2f} MB)")

print("\n>>> Next steps:")
print("  1. Test on real-world images: python comprehensive_real_world_test.py")
print("  2. Deploy to app: python convert_and_deploy.py")
print("  3. Verify deployment: python verify_deployed_model.py")