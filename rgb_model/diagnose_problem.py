#!/usr/bin/env python3
"""
Diagnose why models are getting stuck at random chance (14.3%)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
X_train = np.load('data/splits/X_train.npy')
y_train = np.load('data/splits/y_train.npy')
X_val = np.load('data/splits/X_val.npy')
y_val = np.load('data/splits/y_val.npy')

print(f"Train shape: {X_train.shape}")
print(f"Val shape: {X_val.shape}")
print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# Test 1: Check if data has meaningful variation
print("\n" + "="*50)
print("TEST 1: Data Variation")
print("="*50)
sample_std = X_train[:100].std(axis=(1,2,3))
print(f"Standard deviation of first 100 samples: {sample_std.mean():.4f}")
if sample_std.mean() < 0.05:
    print("WARNING: Very low variation in data!")

# Test 2: Check class separability with simple model
print("\n" + "="*50)
print("TEST 2: Simple Model Test")
print("="*50)

# Create simplest possible model
simple_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(7, activation='softmax')
])

simple_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train for just 5 epochs
print("Training simple linear model...")
history = simple_model.fit(
    X_train[:1000], y_train[:1000],
    validation_data=(X_val[:200], y_val[:200]),
    epochs=5,
    batch_size=32,
    verbose=0
)

print(f"Simple model accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"Simple model val accuracy: {history.history['val_accuracy'][-1]:.2%}")

if history.history['accuracy'][-1] < 0.20:
    print("ERROR: Even simple model can't learn! Data might be corrupted.")

# Test 3: Check actual image content
print("\n" + "="*50)
print("TEST 3: Image Content Check")
print("="*50)

# Check if images are all the same
first_image = X_train[0]
different_count = 0
for i in range(1, 100):
    if not np.allclose(X_train[i], first_image, atol=0.01):
        different_count += 1

print(f"Different images in first 100: {different_count}/99")
if different_count < 50:
    print("WARNING: Many duplicate or near-duplicate images!")

# Test 4: Check predictions distribution
print("\n" + "="*50)
print("TEST 4: Prediction Distribution")
print("="*50)

# Get predictions
preds = simple_model.predict(X_val[:100], verbose=0)
pred_classes = np.argmax(preds, axis=1)
unique, counts = np.unique(pred_classes, return_counts=True)

print("Predicted class distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count}")

if len(unique) == 1:
    print("ERROR: Model predicting only one class!")

# Test 5: Check if problem is with preprocessing
print("\n" + "="*50)
print("TEST 5: Testing Different Preprocessing")
print("="*50)

# Try with different scaling
X_train_scaled = X_train[:1000] * 2.0 - 1.0  # Scale to [-1, 1]

simple_model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(7, activation='softmax')
])

simple_model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = simple_model2.fit(
    X_train_scaled, y_train[:1000],
    epochs=5,
    batch_size=32,
    verbose=0
)

print(f"Model with [-1,1] scaling accuracy: {history2.history['accuracy'][-1]:.2%}")

# Save diagnostic results
print("\n" + "="*50)
print("DIAGNOSIS COMPLETE")
print("="*50)

if history.history['accuracy'][-1] > 0.30:
    print("✓ Data is learnable - issue is with complex model architecture")
    print("Solution: Use simpler model or fix preprocessing for complex models")
else:
    print("✗ Fundamental data issue detected")
    print("Solution: Need to reload and reprocess the dataset")