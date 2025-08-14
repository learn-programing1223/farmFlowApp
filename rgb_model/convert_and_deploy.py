#!/usr/bin/env python3
"""
Convert best model to TFLite and deploy to web app
"""

import tensorflow as tf
from pathlib import Path
import shutil

print("=" * 60)
print("MODEL DEPLOYMENT")
print("=" * 60)

# Find best model
model_files = [
    ('models/improved_best.h5', 'Improved model'),
    ('models/robust_final_best.h5', 'Robust model'),
    ('models/best_cyclegan_model.h5', 'CycleGAN model')
]

best_model_path = None
for path, name in model_files:
    if Path(path).exists():
        print(f"Found: {name} at {path}")
        best_model_path = path
        break

if not best_model_path:
    print("No model found!")
    exit(1)

print(f"\nUsing: {best_model_path}")

# Load model
print("Loading model...")
model = tf.keras.models.load_model(best_model_path, compile=False)

# Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save TFLite model
tflite_path = 'models/plant_disease_final.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved: {len(tflite_model)/1024/1024:.2f} MB")

# Deploy to web app
web_model_path = Path('../PlantPulse/assets/models/plant_disease_model.tflite')
web_model_path.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(tflite_path, web_model_path)
print(f"Deployed to: {web_model_path}")

# Update web app to use new model
web_app_path = Path('../PlantPulse/web-app-final.html')
if web_app_path.exists():
    print(f"\nWeb app found at: {web_app_path}")
    print("Model has been deployed!")
    print("\nTo test:")
    print("1. Open web-app-final.html in browser")
    print("2. Upload plant disease images")
    print("3. Check predictions")
else:
    print("\nWeb app not found")
    print("Copy the model manually to your web app")

print("\n" + "=" * 60)
print("DEPLOYMENT COMPLETE")
print("=" * 60)
print("\nNote: The model was trained on synthetic data")
print("Performance on real images may be limited")
print("\nFor production use:")
print("1. Download real PlantVillage dataset (54,000 images)")
print("2. Train with transfer learning from ImageNet")
print("3. Use data augmentation specifically for field conditions")
print("4. Target 85%+ accuracy on validation set")