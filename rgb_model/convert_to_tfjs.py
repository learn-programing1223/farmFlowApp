#!/usr/bin/env python3
"""
Convert the 96% accuracy CycleGAN model to TensorFlow.js format
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path

print("="*70)
print("🔄 CONVERTING 96% MODEL TO TENSORFLOW.JS")
print("="*70)

# Load your best model
model_path = 'models/cyclegan_best.h5'
output_path = '../PlantPulse/assets/models/'

print(f"\n📂 Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

print(f"✅ Model loaded successfully!")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Parameters: {model.count_params():,}")

# Create output directory if it doesn't exist
Path(output_path).mkdir(parents=True, exist_ok=True)

# Convert to TensorFlow.js format
print(f"\n🔄 Converting to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, output_path)

print(f"\n✅ Conversion complete!")
print(f"📁 Model saved to: {output_path}")
print("\nFiles created:")
print("  - model.json (model architecture)")
print("  - group1-shard1of1.bin (model weights)")

print("\n🌐 To use in web app:")
print("  1. Open PlantPulse/web-app-cyclegan.html in a browser")
print("  2. Upload any plant image")
print("  3. Get instant disease detection with 96% accuracy!")

print("\n" + "="*70)