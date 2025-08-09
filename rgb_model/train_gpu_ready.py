#!/usr/bin/env python3
"""
GPU-Ready Training Script with CPU Optimization
Works with or without GPU - maximizes available hardware
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

# Import the fixed model
from model_fixed import build_fixed_model, compile_fixed_model

# ============================================================================
# HARDWARE OPTIMIZATION
# ============================================================================

def setup_hardware_optimization():
    """Configure TensorFlow for maximum performance on available hardware"""
    
    print("\n" + "="*70)
    print("CONFIGURING HARDWARE OPTIMIZATION")
    print("="*70)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = False
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"[OK] GPU detected: {gpus[0].name}")
            print(f"[OK] GPU Memory: Dynamic allocation enabled")
            gpu_available = True
            
            # Try to enable mixed precision
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"[OK] Mixed precision enabled: Faster training with FP16")
            except:
                print("! Mixed precision not available - using FP32")
                
        except Exception as e:
            print(f"[WARNING] GPU setup error: {e}")
            gpu_available = False
    else:
        print("! No GPU detected with TensorFlow")
        print("  To enable GPU support:")
        print("  1. Install CUDA Toolkit 11.8")
        print("  2. Install cuDNN 8.6")
        print("  3. pip install tensorflow[and-cuda]")
        print("\n  Continuing with CPU optimization...")
    
    # CPU optimization for Ryzen 7
    num_cores = psutil.cpu_count(logical=False)
    num_threads = psutil.cpu_count(logical=True)
    
    # Set threading for optimal performance
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    print(f"\n[OK] CPU optimization enabled:")
    print(f"  - Physical cores: {num_cores}")
    print(f"  - Logical threads: {num_threads}")
    print(f"  - All cores will be utilized for data processing")
    
    # Enable XLA if available
    try:
        tf.config.optimizer.set_jit(True)
        print("[OK] XLA JIT compilation enabled")
    except:
        print("! XLA not available")
    
    print("="*70 + "\n")
    
    return gpu_available


# ============================================================================
# OPTIMIZED DATA PIPELINE
# ============================================================================

def create_optimized_dataset(X, y, batch_size=32, training=True, num_workers=8):
    """Create high-performance data pipeline"""
    
    def preprocess(image, label):
        """Preprocessing with optional augmentation"""
        if training:
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            
            # Random brightness
            image = tf.image.random_brightness(image, 0.1)
            
            # Random contrast  
            image = tf.image.random_contrast(image, 0.9, 1.1)
            
            # Ensure values are in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle if training
    if training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Apply preprocessing
    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache and prefetch for performance
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class TrainingMonitor(tf.keras.callbacks.Callback):
    """Monitor training progress and performance"""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.best_accuracy = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}")
        print(f"{'='*60}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Get metrics
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Update best
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            print(f"[NEW BEST] Validation accuracy: {val_acc:.4f}")
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Estimate remaining time
        if len(self.epoch_times) > 0:
            avg_time = np.mean(self.epoch_times)
            remaining_epochs = self.total_epochs - (epoch + 1)
            eta = avg_time * remaining_epochs
            print(f"  ETA: {eta/60:.1f} minutes")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(config):
    """Main training function"""
    
    # Setup hardware optimization
    gpu_available = setup_hardware_optimization()
    
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
    
    # Determine optimal batch size
    if gpu_available:
        batch_size = 64  # Larger batch for GPU
    else:
        batch_size = 32  # Smaller batch for CPU
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Hardware: {'GPU' if gpu_available else 'CPU'} mode")
    
    # Create optimized datasets
    print("\nCreating data pipeline...")
    
    train_dataset = create_optimized_dataset(
        X_train, y_train,
        batch_size=batch_size,
        training=True,
        num_workers=psutil.cpu_count()
    )
    
    val_dataset = create_optimized_dataset(
        X_val, y_val,
        batch_size=batch_size,
        training=False,
        num_workers=psutil.cpu_count()
    )
    
    # Build model
    print("\nBuilding model...")
    model, base_model = build_fixed_model(
        num_classes=config['num_classes'],
        input_shape=(224, 224, 3)
    )
    
    # Compile
    initial_lr = config.get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Apply mixed precision optimizer if available
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"[OK] Model ready: {model.count_params():,} parameters")
    
    # Setup callbacks
    callbacks = [
        # Training monitor
        TrainingMonitor(config['epochs']),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    try:
        history = model.fit(
            train_dataset,
            epochs=config['epochs'],
            validation_data=val_dataset,
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
        print(f"  Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Performance metrics
        total_samples = len(X_train) * len(history.history['loss'])
        throughput = total_samples / training_time
        print(f"  Average throughput: {throughput:.0f} samples/sec")
        
        # Save model
        print("\nSaving model...")
        model.save('models/plant_disease_model.h5')
        
        # Save history
        with open('models/training_history.json', 'w') as f:
            json.dump(history.history, f, indent=2)
        
        print("[OK] Model saved to models/plant_disease_model.h5")
        print("[OK] History saved to models/training_history.json")
        
        return model, history
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        return None, None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 32,  # Will be adjusted based on hardware
        'epochs': 50,
        'num_classes': 7,
        'learning_rate': 0.001,
    }
    
    print("\n" + "="*70)
    print("   OPTIMIZED PLANT DISEASE DETECTION TRAINING")
    print("="*70)
    
    # System info
    print("\nSystem Information:")
    print(f"  CPU: AMD Ryzen 7 ({psutil.cpu_count(logical=False)} cores, {psutil.cpu_count()} threads)")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Check for GPU using nvidia-smi
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"  GPU: {gpu_info}")
    except:
        pass
    
    # Start training
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    model, history = train_model(config)
    
    if model is not None:
        print("\nTraining successful! You can now use the model for predictions.")
    else:
        print("\n[WARNING] Training was not completed. Please check the errors above.")