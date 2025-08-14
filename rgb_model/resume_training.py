#!/usr/bin/env python3
"""
RESUME TRAINING FROM CHECKPOINT
Continue training from where you left off
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

print("="*70)
print("🔄 RESUMING CYCLEGAN MODEL TRAINING")
print("="*70)

# Load the saved model
print("\n📂 Loading checkpoint...")
model_path = 'models/cyclegan_best.h5'

if not Path(model_path).exists():
    print(f"❌ No checkpoint found at {model_path}")
    print("Please ensure you've run training at least once")
    exit(1)

model = keras.models.load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# Load metadata to get last epoch
metadata_path = 'models/cyclegan_metadata.json'
last_epoch = 0
if Path(metadata_path).exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        if 'training_history' in metadata and 'loss' in metadata['training_history']:
            last_epoch = len(metadata['training_history']['loss'])
            print(f"📊 Last completed epoch: {last_epoch}")

# Configuration (same as original)
config = {
    'input_shape': (224, 224, 3),
    'num_classes': 7,
    'batch_size': 32,
    'epochs': 50,  # Total epochs
    'initial_epoch': last_epoch,  # Start from here
    'learning_rate': 0.001,
    'data_path': 'datasets/ultimate_cyclegan',
}

print(f"\n⏭️ Resuming from epoch {config['initial_epoch']} to {config['epochs']}")

# Setup data generators (same as original)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    Path(config['data_path']) / 'train',
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    Path(config['data_path']) / 'val',
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights (same as original)
print("\n⚖️ Recalculating class weights...")
train_path = Path(config['data_path']) / 'train'
class_counts = {}
total_images = 0

for class_dir in train_path.iterdir():
    if class_dir.is_dir():
        count = len(list(class_dir.glob('*.jpg')))
        if count > 0:
            class_counts[class_dir.name] = count
            total_images += count

num_classes = len(class_counts)
class_weights = {}
for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
    weight = total_images / (num_classes * count)
    class_weights[idx] = weight
    print(f"  {class_name}: {weight:.3f}")

# Learning rate schedule with warm restart
def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing schedule"""
    epochs = config['epochs']
    lr_min = 1e-6
    lr_max = config['learning_rate']
    
    # Adjust for resumed training
    current_epoch = epoch + config['initial_epoch']
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(current_epoch / epochs * np.pi))
    return lr

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/cyclegan_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    keras.callbacks.LearningRateScheduler(
        cosine_annealing_schedule,
        verbose=1
    ),
    
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Add CSV logger to track progress
    keras.callbacks.CSVLogger(
        'models/training_log.csv',
        append=True  # Append to existing log
    )
]

# RESUME TRAINING
print("\n" + "="*70)
print("🚀 RESUMING TRAINING")
print("="*70)
print(f"\n⏰ Continuing from epoch {config['initial_epoch']+1}")
print("💡 Press Ctrl+C to pause again (will save checkpoint)")
print("📊 Progress saved to: models/training_log.csv")

try:
    history = model.fit(
        train_generator,
        initial_epoch=config['initial_epoch'],  # Resume from here
        epochs=config['epochs'],
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed successfully!")
    
except KeyboardInterrupt:
    print("\n\n⏸️ Training paused by user")
    print("✅ Model checkpoint saved to: models/cyclegan_best.h5")
    print("📝 To resume: python resume_training.py")
    print("📊 Check progress: models/training_log.csv")

# Save final model
print("\n💾 Saving final model...")
model.save('models/cyclegan_final.h5')

# Convert to TFLite
print("\n🔄 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('models/cyclegan_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved: {len(tflite_model)/1024/1024:.2f} MB")
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)