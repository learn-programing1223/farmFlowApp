#!/usr/bin/env python3
"""
Diagnostic script to identify training issues
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("=" * 70)
print("TRAINING DIAGNOSTICS")
print("=" * 70)

# 1. Check GPU
print("\n1. GPU Check:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ GPU found: {gpus[0].name}")
    print(f"   Memory growth enabled: {tf.config.experimental.get_memory_growth(gpus[0])}")
else:
    print("   ✗ No GPU found - using CPU")

# 2. Check data distribution
print("\n2. Data Distribution Check:")
data_dir = 'datasets/plantvillage_processed'

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(data_dir, split)
    if os.path.exists(split_dir):
        print(f"\n   {split.upper()}:")
        total = 0
        class_counts = {}
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
                total += count
                print(f"      {class_name}: {count} images")
        print(f"      TOTAL: {total} images")
        
        # Check for empty classes
        empty_classes = [c for c, count in class_counts.items() if count == 0]
        if empty_classes:
            print(f"      ⚠️  EMPTY CLASSES: {empty_classes}")

# 3. Test data loading
print("\n3. Data Loading Test:")
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"   Train generator: {train_generator.samples} samples, {train_generator.num_classes} classes")
print(f"   Val generator: {val_generator.samples} samples, {val_generator.num_classes} classes")
print(f"   Class indices: {train_generator.class_indices}")

# 4. Check first batch
print("\n4. First Batch Analysis:")
X_batch, y_batch = next(train_generator)
print(f"   Batch shape: {X_batch.shape}")
print(f"   Label shape: {y_batch.shape}")
print(f"   Min pixel value: {X_batch.min():.3f}")
print(f"   Max pixel value: {X_batch.max():.3f}")
print(f"   Label distribution in batch: {np.sum(y_batch, axis=0).astype(int)}")

# 5. Check class distribution in validation
print("\n5. Validation Set Class Distribution:")
val_labels = []
val_generator.reset()
for i in range(len(val_generator)):
    _, y = val_generator[i]
    val_labels.append(y)
val_labels = np.vstack(val_labels)
val_class_counts = np.sum(val_labels, axis=0).astype(int)
print(f"   Validation class distribution: {val_class_counts}")
print(f"   Most common class: Class {np.argmax(val_class_counts)} with {np.max(val_class_counts)} samples")

# 6. Test a simple model prediction
print("\n6. Simple Model Test:")
from tensorflow.keras import layers, models

# Create a tiny test model
test_model = models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

test_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Test one batch
X_test, y_test = next(val_generator)
preds = test_model.predict(X_test, verbose=0)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(f"   Predicted classes: {np.unique(pred_classes, return_counts=True)}")
print(f"   True classes: {np.unique(true_classes, return_counts=True)}")

# 7. Check for data leakage
print("\n7. Data Leakage Check:")
train_files = set()
for root, dirs, files in os.walk(os.path.join(data_dir, 'train')):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            train_files.add(file)

val_files = set()
for root, dirs, files in os.walk(os.path.join(data_dir, 'val')):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            val_files.add(file)

overlap = train_files.intersection(val_files)
if overlap:
    print(f"   ⚠️  WARNING: {len(overlap)} files appear in both train and val!")
else:
    print(f"   ✓ No overlap between train and val sets")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)