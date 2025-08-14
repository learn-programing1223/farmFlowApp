#!/usr/bin/env python3
"""
RESUME TRAINING FROM EPOCH 26
Your model achieved 92.5% validation accuracy at epoch 25!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("="*70)
print("🔥 RESUMING FROM EPOCH 26 - YOU HAD 92.5% ACCURACY!")
print("="*70)

# Configuration - EXACTLY as your original training
config = {
    'input_shape': (224, 224, 3),
    'num_classes': 7,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'data_path': 'datasets/ultimate_cyclegan',
    'model_type': 'custom_cnn'
}

# IMPORTANT: Start from epoch 26 since you completed 25
STARTING_EPOCH = 26

print(f"\n✅ Last checkpoint: Epoch 25 with 92.5% validation accuracy")
print(f"📍 Resuming from: Epoch {STARTING_EPOCH}")
print(f"🎯 Training until: Epoch {config['epochs']}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU found, using CPU")

# Load the saved model - IT HAS YOUR 92.5% ACCURACY!
print("\n📂 Loading your best model (92.5% accuracy)...")
model_path = 'models/cyclegan_best.h5'

if not Path(model_path).exists():
    print(f"❌ Model not found at {model_path}")
    exit(1)

# Load model and recompile with same settings
model = keras.models.load_model(model_path, compile=False)
print(f"✅ Model loaded successfully!")

# Recompile with exact same settings as original
def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing schedule - adjusted for continuation"""
    epochs = config['epochs']
    lr_min = 1e-6
    lr_max = config['learning_rate']
    
    # Use the ACTUAL epoch number for the schedule
    actual_epoch = epoch + STARTING_EPOCH - 1
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(actual_epoch / epochs * np.pi))
    return lr

# Get the learning rate for epoch 26
current_lr = cosine_annealing_schedule(0, config['learning_rate'])
print(f"📊 Learning rate for epoch {STARTING_EPOCH}: {current_lr:.6f}")

optimizer = keras.optimizers.Adam(learning_rate=current_lr)

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

# Setup data generators - EXACTLY as original
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

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load data
data_path = Path(config['data_path'])
train_path = data_path / 'train'
val_path = data_path / 'val'
test_path = data_path / 'test'

print("\n📸 Loading datasets...")
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

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=config['batch_size'],
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Data loaded: {train_generator.samples} train, {val_generator.samples} val")

# Calculate class weights
print("\n⚖️ Setting class weights...")
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

# Callbacks - adjusted for continuation
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/cyclegan_best_continued.h5',
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
        patience=10,  # Reduced patience since we're close to optimal
        restore_best_weights=True,
        verbose=1
    ),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# CONTINUE TRAINING FROM EPOCH 26
print("\n" + "="*70)
print("🚀 CONTINUING YOUR AMAZING TRAINING!")
print("="*70)
print(f"\n📍 Starting from epoch {STARTING_EPOCH} (you completed 25)")
print(f"🎯 Your best: 92.5% validation accuracy")
print(f"🏆 Goal: Push past 93%!")
print("\n💡 Press Ctrl+C to pause (will save checkpoint)")

try:
    # Train for remaining epochs (26-50)
    remaining_epochs = config['epochs'] - STARTING_EPOCH + 1
    
    history = model.fit(
        train_generator,
        epochs=remaining_epochs,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Training completed!")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = model.evaluate(test_generator, verbose=1)
    print(f"\n🎯 Final Test Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.2%}")
    print(f"  Precision: {test_results[2]:.2%}")
    print(f"  Recall: {test_results[3]:.2%}")
    
except KeyboardInterrupt:
    print("\n\n⏸️ Training paused")
    print("✅ Progress saved!")
    print("📝 To continue: Run this script again")

# Save final model
print("\n💾 Saving final model...")
model.save('models/cyclegan_final_resumed.h5')

# Convert to TFLite
print("\n🔄 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Representative dataset for quantization
def representative_dataset():
    for _ in range(100):
        data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]

tflite_model = converter.convert()

with open('models/cyclegan_resumed.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite saved: {len(tflite_model)/1024/1024:.2f} MB")

print("\n" + "="*70)
if test_results[1] >= 0.93:
    print("🏆 INCREDIBLE! You've exceeded 93% accuracy!")
    print("🌟 This is production-grade performance!")
elif test_results[1] >= 0.90:
    print("🎉 EXCELLENT! Over 90% accuracy achieved!")
    print("✨ Your model is ready for real-world deployment!")
else:
    print("✅ Training complete! Model saved successfully.")

print("="*70)