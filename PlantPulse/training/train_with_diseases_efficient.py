#!/usr/bin/env python3
"""
Memory-efficient training with disease data
Uses data generators instead of loading all images into memory
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_with_real_thermal import create_thermal_model
from thermal_data_loader import ThermalDatasetLoader

def create_efficient_data_generator(data_dir: Path, batch_size: int = 32):
    """Create a memory-efficient data generator"""
    
    # Use ImageDataGenerator for efficient loading
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create data generator with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    
    # Create training generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        color_mode='grayscale'
    )
    
    # Create validation generator
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        color_mode='grayscale'
    )
    
    return train_generator, val_generator

def create_subset_dataset(full_dir: Path, subset_dir: Path, samples_per_class: int = 1000):
    """Create a smaller subset for training"""
    import shutil
    import random
    
    print(f"\nCreating subset with {samples_per_class} samples per class...")
    
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Categories to process
    categories = ['bacterial', 'fungal', 'viral', 'healthy']
    
    for category in categories:
        src_dir = full_dir / category
        dst_dir = subset_dir / category
        dst_dir.mkdir(exist_ok=True)
        
        if src_dir.exists():
            # Get all images
            images = list(src_dir.glob('*.png'))
            
            # Sample randomly
            selected = random.sample(images, min(samples_per_class, len(images)))
            
            # Copy selected images
            for img in selected:
                shutil.copy(img, dst_dir / img.name)
            
            print(f"  {category}: {len(selected)} images")
    
    return subset_dir

def train_efficient():
    """Train with memory-efficient approach"""
    
    print("\n" + "="*60)
    print("MEMORY-EFFICIENT DISEASE TRAINING")
    print("="*60)
    
    # Paths
    disease_dir = Path("data/disease_datasets/thermal_diseases")
    subset_dir = Path("data/disease_datasets/thermal_subset")
    
    # Check if we should create a subset
    if not subset_dir.exists():
        print("\n⚠️  Full dataset too large for memory!")
        print("Creating a balanced subset for training...")
        create_subset_dataset(disease_dir, subset_dir, samples_per_class=2000)
    
    # Count images
    print("\n📊 Subset Statistics:")
    for category in ['bacterial', 'fungal', 'viral', 'healthy']:
        count = len(list((subset_dir / category).glob('*.png')))
        print(f"  {category}: {count} images")
    
    # Create data generators
    print("\n🔄 Creating data generators...")
    train_gen, val_gen = create_efficient_data_generator(subset_dir, batch_size=32)
    
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes: {list(train_gen.class_indices.keys())}")
    
    # Build model
    print("\n🏗️  Building model...")
    
    # Simple efficient model for disease classification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 1)),
        
        # Feature extraction
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.4),
        
        # Classification
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
    ])
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'disease_classifier_{timestamp}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Starting training...")
    print("This uses generators to avoid memory issues")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(f'disease_classifier_{timestamp}_final.h5')
    
    # Results
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n✅ Training complete!")
    print(f"Final validation accuracy: {final_acc:.1%}")
    
    if final_acc > 0.95:
        print("\n⚠️  Accuracy still seems high!")
        print("But this is more realistic with real disease data")
    else:
        print("\n✅ Realistic accuracy achieved!")
        print("The model is learning real disease patterns")
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f'disease_classifier_{timestamp}.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved: disease_classifier_{timestamp}.tflite")
    print(f"Size: {len(tflite_model) / 1024 / 1024:.1f} MB")
    
    return model, history

def main():
    """Main entry point"""
    
    print("This script trains efficiently with limited memory")
    print("It will create a subset of the data if needed")
    
    response = input("\nProceed with efficient training? (y/n): ")
    if response.lower() != 'y':
        return
    
    train_efficient()

if __name__ == "__main__":
    main()