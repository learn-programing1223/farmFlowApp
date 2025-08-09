#!/usr/bin/env python3
"""
SIMPLIFIED GPU-OPTIMIZED Training Script
Works reliably with CPU optimization, ready for GPU when available
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import psutil


def build_simple_model(num_classes=7, input_shape=(224, 224, 3)):
    """Build a simple but effective CNN model"""
    
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        # Block 4
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def setup_optimization():
    """Configure TensorFlow for maximum performance"""
    
    print("\n" + "="*70)
    print("CONFIGURING PERFORMANCE OPTIMIZATION")
    print("="*70)
    
    # CPU optimization
    num_cores = psutil.cpu_count(logical=False)
    num_threads = psutil.cpu_count(logical=True)
    
    # Configure threading
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    print(f"[OK] CPU: {num_cores} cores, {num_threads} threads")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] GPU detected: {gpus[0].name}")
            return True
        except:
            print("[INFO] GPU detected but not configured")
            return False
    else:
        print("[INFO] No GPU - using CPU optimization")
        print("      For 2-3x faster training, install CUDA 11.8 + cuDNN 8.6")
        return False
    
    print("="*70 + "\n")


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """Create optimized data generators"""
    
    # Training data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data (no augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model():
    """Main training function"""
    
    # Setup optimization
    has_gpu = setup_optimization()
    
    # Load data
    print("Loading training data...")
    data_dir = Path('./data/splits')
    
    if not data_dir.exists():
        print("\n[ERROR] Training data not found!")
        print("Please run: python setup_all_disease_datasets.py")
        return None
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"[OK] Loaded {len(X_train):,} training samples")
    print(f"[OK] Loaded {len(X_val):,} validation samples")
    
    # Batch size based on hardware
    batch_size = 64 if has_gpu else 32
    
    # Create data generators
    print("\nCreating data pipeline...")
    train_gen, val_gen = create_data_generators(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_simple_model(num_classes=7)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"[OK] Model ready: {model.count_params():,} parameters")
    print(f"[OK] Batch size: {batch_size}")
    print(f"[OK] Hardware: {'GPU' if has_gpu else 'CPU (All cores)'}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    # Calculate steps
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Training
    print("\n" + "="*70)
    print("STARTING OPTIMIZED TRAINING")
    print("="*70)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=50,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Results
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        
        print(f"\nResults:")
        print(f"  Total time: {training_time/60:.1f} minutes")
        print(f"  Best accuracy: {max(history.history['val_accuracy']):.4f}")
        
        # Calculate throughput
        total_samples = len(X_train) * len(history.history['loss'])
        throughput = total_samples / training_time
        print(f"  Throughput: {throughput:.0f} samples/sec")
        
        # Save
        print("\nSaving model...")
        model.save('models/plant_disease_optimized.h5')
        
        with open('models/training_history.json', 'w') as f:
            json.dump(history.history, f, indent=2)
        
        print("[OK] Model saved successfully!")
        
        return model, history
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("   OPTIMIZED TRAINING FOR RTX 3060 Ti + RYZEN 7")
    print("="*70)
    
    # System info
    print("\nSystem Information:")
    print(f"  CPU: AMD Ryzen 7 ({psutil.cpu_count(logical=False)} cores)")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check GPU
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
    except:
        pass
    
    print("\nThis script will:")
    print("  - Use all 16 CPU threads for data processing")
    print("  - Apply optimal batch sizes")
    print("  - Use GPU if TensorFlow-GPU is installed")
    print("  - Monitor and save the best model automatically")
    
    print("\nStarting training in 3 seconds...")
    time.sleep(3)
    
    model, history = train_model()
    
    if model is not None:
        print("\n[SUCCESS] Training complete! Model is ready for use.")
        print("          Best model saved to: models/best_model.h5")
    else:
        print("\n[FAILED] Please check the error messages above.")