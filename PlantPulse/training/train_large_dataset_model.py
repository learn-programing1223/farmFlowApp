#!/usr/bin/env python3
"""
Train disease classification model on larger dataset for 85%+ accuracy
Uses memory-efficient techniques to handle more data
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
from datetime import datetime
import json
import gc

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import realistic thermal converter
from train_realistic_disease_model import RealisticThermalConverter

class EfficientLargeDataGenerator(keras.utils.Sequence):
    """Memory-efficient generator for large datasets"""
    
    def __init__(self, data_dir, subset_size_per_class=10000, 
                 batch_size=32, augment=True, validation_split=0.2, 
                 is_training=True, random_seed=42):
        
        self.batch_size = batch_size
        self.augment = augment and is_training
        self.converter = RealisticThermalConverter()
        
        # Categories
        self.categories = ['bacterial', 'fungal', 'viral', 'healthy']
        self.label_map = {cat: i for i, cat in enumerate(self.categories)}
        
        # Collect file paths
        all_paths = []
        all_labels = []
        
        print(f"\nLoading dataset (up to {subset_size_per_class} per class)...")
        
        np.random.seed(random_seed)
        
        for category in self.categories:
            cat_dir = data_dir / category
            if cat_dir.exists():
                # Get all image paths
                paths = list(cat_dir.glob('*.png'))
                
                # Shuffle to ensure random selection
                np.random.shuffle(paths)
                
                # Take subset
                selected_paths = paths[:subset_size_per_class]
                
                print(f"  {category}: {len(selected_paths)} images")
                
                all_paths.extend(selected_paths)
                all_labels.extend([category] * len(selected_paths))
        
        # Convert to arrays and shuffle
        combined = list(zip(all_paths, all_labels))
        np.random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        
        # Split train/validation
        n_samples = len(all_paths)
        n_train = int(n_samples * (1 - validation_split))
        
        if is_training:
            self.paths = all_paths[:n_train]
            self.labels = all_labels[:n_train]
        else:
            self.paths = all_paths[n_train:]
            self.labels = all_labels[n_train:]
        
        self.indices = np.arange(len(self.paths))
        
        print(f"\n{'Training' if is_training else 'Validation'} set: {len(self.paths)} images")
    
    def __len__(self):
        return len(self.paths) // self.batch_size
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), 224, 224, 1), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices), 4), dtype=np.float32)
        
        # Process each image in batch
        for i, img_idx in enumerate(batch_indices):
            # Load image
            img_path = self.paths[img_idx]
            label = self.labels[img_idx]
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("Failed to load image")
                
                # Resize
                img = cv2.resize(img, (224, 224))
                
            except Exception:
                # Create random image if loading fails
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Convert to thermal
            thermal = self.converter.convert_to_thermal(img, label, add_artifacts=True)
            
            # Data augmentation
            if self.augment:
                thermal = self._augment(thermal)
            
            # Normalize
            thermal = (thermal - 22) / 5
            thermal = np.expand_dims(thermal, -1)
            
            # Add to batch
            batch_x[i] = thermal
            batch_y[i] = keras.utils.to_categorical(self.label_map[label], 4)
        
        return batch_x, batch_y
    
    def _augment(self, thermal):
        """Apply augmentation to thermal image"""
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1)
            thermal = cv2.warpAffine(thermal, M, (224, 224))
        
        # Random shift
        if np.random.random() > 0.5:
            dx = np.random.randint(-15, 15)
            dy = np.random.randint(-15, 15)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            thermal = cv2.warpAffine(thermal, M, (224, 224))
        
        # Random zoom
        if np.random.random() > 0.5:
            zoom = np.random.uniform(0.85, 1.15)
            M = cv2.getRotationMatrix2D((112, 112), 0, zoom)
            thermal = cv2.warpAffine(thermal, M, (224, 224))
        
        # Random flip
        if np.random.random() > 0.5:
            thermal = np.fliplr(thermal)
        
        # Random brightness/contrast
        if np.random.random() > 0.5:
            thermal = thermal * np.random.uniform(0.9, 1.1) + np.random.uniform(-1, 1)
        
        # Additional noise
        thermal += np.random.normal(0, 0.3, thermal.shape)
        
        return thermal
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        np.random.shuffle(self.indices)

def create_optimized_model(input_shape=(224, 224, 1), num_classes=4):
    """Optimized model for larger dataset - slightly deeper than simple model"""
    
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_on_large_dataset():
    """Train model on larger dataset for improved accuracy"""
    
    print("\n" + "="*60)
    print("LARGE DATASET TRAINING FOR 85%+ ACCURACY")
    print("="*60)
    print("Strategy:")
    print("✓ Use 10K images per class (40K total)")
    print("✓ Slightly deeper architecture")
    print("✓ Advanced augmentation")
    print("✓ Mixed precision training for efficiency")
    print("="*60)
    
    # Enable mixed precision for faster training (only on GPU)
    try:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled (GPU detected)")
        else:
            print("Running on CPU - mixed precision disabled")
    except Exception as e:
        print(f"Mixed precision setup skipped: {e}")
    
    # Paths
    data_dir = Path("data/disease_datasets/thermal_diseases")
    
    # Create data generators
    print("\nCreating data generators...")
    
    train_gen = EfficientLargeDataGenerator(
        data_dir,
        subset_size_per_class=10000,  # 10K per class = 40K total
        batch_size=64,  # Larger batch for efficiency
        augment=True,
        validation_split=0.2,
        is_training=True
    )
    
    val_gen = EfficientLargeDataGenerator(
        data_dir,
        subset_size_per_class=10000,
        batch_size=64,
        augment=False,
        validation_split=0.2,
        is_training=False
    )
    
    # Build model
    print("\nBuilding optimized model...")
    model = create_optimized_model()
    
    # Compile with optimizations
    optimizer = keras.optimizers.Adam(0.001)
    # Add loss scaling for mixed precision only if enabled
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'large_dataset_model_{timestamp}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training on large dataset...")
    print("This may take 1-2 hours depending on GPU...")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(f'large_dataset_model_{timestamp}_final.h5')
    
    # Save history
    with open(f'large_dataset_model_{timestamp}_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Get best accuracy
    best_acc = max(history.history['val_accuracy'])
    final_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_acc:.1%}")
    print(f"Final validation accuracy: {final_acc:.1%}")
    
    if best_acc >= 0.85:
        print("\n✅ TARGET ACHIEVED! Model reached 85%+ accuracy")
        print("Ready for deployment!")
    elif best_acc >= 0.82:
        print("\n⚠️  Close to target! Consider:")
        print("- Training on even more data (20K per class)")
        print("- Fine-tuning hyperparameters")
        print("- Ensemble with previous models")
    else:
        print("\n⚠️  Below target. Try:")
        print("- Using the full dataset")
        print("- More complex architecture")
        print("- Different augmentation strategies")
    
    # Clean up memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    return model, history

if __name__ == "__main__":
    print("This script trains on a larger dataset (40K images)")
    print("Expected improvements:")
    print("- 5x more training data than before")
    print("- Better generalization")
    print("- Target: 85%+ accuracy")
    
    response = input("\nStart large dataset training? (y/n): ")
    if response.lower() == 'y':
        train_on_large_dataset()