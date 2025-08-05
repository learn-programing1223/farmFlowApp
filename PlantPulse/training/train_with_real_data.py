#!/usr/bin/env python3
"""
Complete training script that uses real thermal data if available,
falls back to synthetic data if not
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Import our modules
from train_plant_health_model import (
    build_multi_task_model, 
    train_model, 
    convert_to_tflite,
    ThermalDataGenerator,
    create_dataset
)
from dataset_loader import create_training_dataset

def main():
    parser = argparse.ArgumentParser(description='Train PlantPulse thermal analysis model')
    parser.add_argument('--dataset', type=str, default='./hydroponic_dataset',
                        help='Path to real dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use only synthetic data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 60)
    print("PlantPulse Model Training")
    print("=" * 60)
    
    # Load or generate data
    if args.synthetic or not os.path.exists(args.dataset):
        print("\n📊 Using synthetic thermal data...")
        generator = ThermalDataGenerator()
        train_data = create_dataset(generator, num_samples=10000)
        val_data = create_dataset(generator, num_samples=2000)
    else:
        print(f"\n📊 Loading real dataset from {args.dataset}...")
        train_data, val_data = create_training_dataset(
            args.dataset, 
            augment=True,
            train_split=0.8
        )
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Training samples: {len(train_images)}")
    print(f"   Validation samples: {len(val_images)}")
    print(f"   Image shape: {train_images[0].shape}")
    print(f"   Temperature range: {train_images.min():.1f}°C - {train_images.max():.1f}°C")
    
    # Build model
    print("\n🏗️  Building MobileNetV3-based model...")
    model = build_multi_task_model()
    
    # Model summary
    print("\n📊 Model architecture:")
    model.summary()
    
    # Calculate model parameters
    total_params = model.count_params()
    print(f"\n📈 Total parameters: {total_params:,}")
    print(f"   Estimated size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"   Target size: <50 MB (INT8 quantized)")
    
    # Train model
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: 0.001 (with decay)")
    
    model, history = train_model(model, train_data, val_data)
    
    # Save models
    print("\n💾 Saving models...")
    
    # Save Keras model
    keras_path = os.path.join(args.output_dir, f'plant_health_model_{timestamp}.h5')
    model.save(keras_path)
    print(f"   ✅ Keras model saved: {keras_path}")
    
    # Convert to TFLite
    print("\n🔄 Converting to TensorFlow Lite...")
    tflite_model = convert_to_tflite(model, quantize=True)
    
    # Save TFLite model
    tflite_path = os.path.join(args.output_dir, 'plant_health_v1.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"   ✅ TFLite model saved: {tflite_path}")
    print(f"   📦 Size: {len(tflite_model) / 1024 / 1024:.1f} MB")
    
    # Save training history
    import json
    history_path = os.path.join(args.output_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"   ✅ Training history saved: {history_path}")
    
    # Print final metrics
    print("\n📊 Final training metrics:")
    final_metrics = {k: v[-1] for k, v in history.history.items() if not k.startswith('val_')}
    for metric, value in final_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n📊 Final validation metrics:")
    val_metrics = {k: v[-1] for k, v in history.history.items() if k.startswith('val_')}
    for metric, value in val_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Copy to app directory if it exists
    app_model_dir = '../src/ml/models'
    if os.path.exists(app_model_dir):
        import shutil
        app_model_path = os.path.join(app_model_dir, 'plant_health_v1.tflite')
        shutil.copy(tflite_path, app_model_path)
        print(f"\n🚀 Model copied to app: {app_model_path}")
    
    print("\n✅ Training complete!")
    print(f"\n📱 To use in PlantPulse app:")
    print(f"   1. Copy {tflite_path} to PlantPulse/src/ml/models/")
    print(f"   2. The app will automatically load the new model")
    
    # Performance summary
    print("\n🎯 Model Performance Summary:")
    if 'val_water_stress_mae' in val_metrics:
        print(f"   Water Stress MAE: {val_metrics['val_water_stress_mae']:.3f}")
    if 'val_disease_accuracy' in val_metrics:
        print(f"   Disease Accuracy: {val_metrics['val_disease_accuracy']:.1%}")
    if 'val_nutrients_mae' in val_metrics:
        print(f"   Nutrient MAE: {val_metrics['val_nutrients_mae']:.3f}")
    if 'val_segmentation_accuracy' in val_metrics:
        print(f"   Segmentation Accuracy: {val_metrics['val_segmentation_accuracy']:.1%}")

if __name__ == "__main__":
    main()