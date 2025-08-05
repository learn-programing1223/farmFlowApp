#!/usr/bin/env python3
"""
Resume training from checkpoint
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Import from existing modules
from train_plant_health_model import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    ThermalDataGenerator, create_dataset
)
from dataset_loader import create_training_dataset

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    checkpoints = [
        'plant_health_best.h5',
        'plant_health_improved_best.h5',
        'plant_health_improved.h5'
    ]
    
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            print(f"Found checkpoint: {checkpoint}")
            return checkpoint
    
    return None

def load_training_history():
    """Try to load previous training history"""
    history_files = ['training_history.json']
    
    for hist_file in history_files:
        if os.path.exists(hist_file):
            with open(hist_file, 'r') as f:
                return json.load(f)
    
    return None

def resume_training(checkpoint_path, initial_epoch=18, total_epochs=100):
    """Resume training from checkpoint"""
    
    print("=" * 60)
    print("Resuming PlantPulse Model Training")
    print("=" * 60)
    
    # Load model with custom objects
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    # Define custom loss functions if needed
    custom_objects = {
        'mse': keras.losses.MeanSquaredError(),
        'binary_crossentropy': keras.losses.BinaryCrossentropy(),
        'categorical_crossentropy': keras.losses.CategoricalCrossentropy()
    }
    
    try:
        model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    except:
        # Try loading without compilation
        model = keras.models.load_model(checkpoint_path, compile=False)
        
        # Recompile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                'water_stress': 'mse',
                'disease': 'categorical_crossentropy',
                'nutrients': 'binary_crossentropy',
                'segmentation': 'binary_crossentropy'
            },
            metrics={
                'water_stress': ['mae'],
                'disease': ['accuracy'],
                'nutrients': ['accuracy'],
                'segmentation': ['accuracy']
            }
        )
    
    # Load or generate data
    dataset_path = './hydroponic_dataset'
    if os.path.exists(dataset_path):
        print(f"\n📊 Loading real dataset from {dataset_path}...")
        train_data, val_data = create_training_dataset(
            dataset_path, 
            augment=True,
            train_split=0.8
        )
    else:
        print("\n📊 Using synthetic thermal data...")
        generator = ThermalDataGenerator()
        train_data = create_dataset(generator, num_samples=10000)
        val_data = create_dataset(generator, num_samples=2000)
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Training samples: {len(train_images)}")
    print(f"   Validation samples: {len(val_images)}")
    
    # Set up callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'plant_health_resumed_{timestamp}_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(
            f'training_log_resumed_{timestamp}.csv',
            append=True
        )
    ]
    
    # Resume training
    print(f"\n🚀 Resuming from epoch {initial_epoch}/{total_epochs}...")
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=BATCH_SIZE,
        initial_epoch=initial_epoch - 1,  # 0-indexed
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = f'plant_health_resumed_{timestamp}_final.h5'
    model.save(final_path)
    print(f"\n✅ Final model saved: {final_path}")
    
    # Save updated history
    history_path = f'training_history_resumed_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"✅ Training history saved: {history_path}")
    
    return model, history

if __name__ == "__main__":
    # Find checkpoint
    checkpoint = find_latest_checkpoint()
    
    if not checkpoint:
        print("❌ No checkpoint found!")
        print("Available files:")
        os.system("ls -la *.h5")
        sys.exit(1)
    
    # Resume from epoch 18 (as shown in your screenshot)
    model, history = resume_training(checkpoint, initial_epoch=18, total_epochs=100)
    
    print("\n✅ Training resumed successfully!")