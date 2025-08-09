#!/usr/bin/env python3
"""
GPU-OPTIMIZED Training Script for RTX 3060 Ti + Ryzen 7
Maximizes hardware utilization for fastest training possible
"""

import os
import sys
import json
import time
import psutil
import GPUtil
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Import the fixed model
from model_fixed import build_fixed_model, compile_fixed_model

# ============================================================================
# GPU OPTIMIZATION CONFIGURATION
# ============================================================================

def setup_gpu_optimization():
    """Configure TensorFlow for maximum GPU performance"""
    
    print("\n" + "="*70)
    print("🚀 CONFIGURING GPU OPTIMIZATION FOR RTX 3060 Ti")
    print("="*70)
    
    # Enable GPU growth to prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✓ GPU detected: {gpus[0].name}")
            print(f"✓ GPU Memory: Dynamic allocation enabled")
            
        except RuntimeError as e:
            print(f"⚠ GPU setup error: {e}")
    else:
        print("⚠ No GPU detected - training will be slower")
    
    # Enable mixed precision for RTX 30 series (has Tensor Cores)
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision enabled: {policy.name}")
    except Exception as e:
        print(f"⚠ Mixed precision not available: {e}")
    
    # Enable XLA compilation for faster execution
    tf.config.optimizer.set_jit(True)
    print("✓ XLA JIT compilation enabled")
    
    # TensorFlow threading optimization for Ryzen 7 (8 cores, 16 threads)
    num_cores = psutil.cpu_count(logical=False)  # Physical cores
    num_threads = psutil.cpu_count(logical=True)  # Logical cores (with HT)
    
    # Optimal settings for Ryzen 7
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    print(f"✓ CPU optimization: {num_cores} cores, {num_threads} threads")
    print("="*70 + "\n")
    
    return gpus


# ============================================================================
# OPTIMIZED DATA PIPELINE
# ============================================================================

class OptimizedDataGenerator:
    """High-performance data generator with multi-threading and prefetching"""
    
    def __init__(self, X, y, batch_size=32, num_workers=4, prefetch_size=2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.indices = np.arange(len(X))
        
        # Pre-allocate buffers for zero-copy operations
        self.batch_buffer_x = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
        self.batch_buffer_y = np.zeros((batch_size, y.shape[1]), dtype=np.float32)
        
    def create_dataset(self, training=True):
        """Create optimized tf.data pipeline"""
        
        def generator():
            if training:
                np.random.shuffle(self.indices)
            
            for start_idx in range(0, len(self.X) - self.batch_size, self.batch_size):
                batch_indices = self.indices[start_idx:start_idx + self.batch_size]
                
                # Use pre-allocated buffers (faster than creating new arrays)
                np.copyto(self.batch_buffer_x, self.X[batch_indices])
                np.copyto(self.batch_buffer_y, self.y[batch_indices])
                
                # Fast augmentation using vectorized operations
                if training:
                    # Random horizontal flip (50% chance)
                    if np.random.random() < 0.5:
                        self.batch_buffer_x = self.batch_buffer_x[:, :, ::-1, :]
                    
                    # Random brightness (30% chance)
                    if np.random.random() < 0.3:
                        brightness = np.random.uniform(0.9, 1.1)
                        np.multiply(self.batch_buffer_x, brightness, out=self.batch_buffer_x)
                        np.clip(self.batch_buffer_x, 0, 1, out=self.batch_buffer_x)
                    
                    # Random contrast (30% chance)
                    if np.random.random() < 0.3:
                        contrast = np.random.uniform(0.9, 1.1)
                        mean = np.mean(self.batch_buffer_x, axis=(1, 2), keepdims=True)
                        self.batch_buffer_x = (self.batch_buffer_x - mean) * contrast + mean
                        np.clip(self.batch_buffer_x, 0, 1, out=self.batch_buffer_x)
                
                # Convert to float16 for mixed precision
                yield self.batch_buffer_x.copy(), self.batch_buffer_y.copy()
        
        # Create dataset with optimal settings
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, self.y.shape[1]), dtype=tf.float32)
            )
        )
        
        # Apply performance optimizations
        dataset = dataset.cache()  # Cache in memory
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Use parallel processing for data augmentation
        options = tf.data.Options()
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_parallelization = True
        options.threading.private_threadpool_size = self.num_workers
        dataset = dataset.with_options(options)
        
        return dataset


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor(tf.keras.callbacks.Callback):
    """Monitor GPU/CPU usage and training speed"""
    
    def __init__(self):
        super().__init__()
        self.batch_times = []
        self.epoch_start_time = None
        self.gpus = GPUtil.getGPUs()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.batch_times = []
        
        # Monitor system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        print(f"\n📊 System Status:")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   RAM Usage: {memory_percent:.1f}%")
        
        if self.gpus:
            gpu = self.gpus[0]
            gpu_util = gpu.load * 100
            gpu_memory = gpu.memoryUtil * 100
            gpu_temp = gpu.temperature
            
            print(f"   GPU Usage: {gpu_util:.1f}%")
            print(f"   GPU Memory: {gpu_memory:.1f}%")
            print(f"   GPU Temp: {gpu_temp}°C")
    
    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        # Print speed every 10 batches
        if batch % 10 == 0 and batch > 0:
            avg_batch_time = np.mean(self.batch_times[-10:])
            samples_per_sec = self.params['batch_size'] / avg_batch_time
            
            # Get current GPU utilization
            if self.gpus:
                gpu_util = self.gpus[0].load * 100
                print(f"   Batch {batch}: {samples_per_sec:.1f} samples/sec | GPU: {gpu_util:.1f}%", end='\r')
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = np.mean(self.batch_times)
        
        print(f"\n⏱ Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"   Average batch time: {avg_batch_time:.3f}s")
        print(f"   Throughput: {len(self.batch_times) * self.params['batch_size'] / epoch_time:.1f} samples/sec")


# ============================================================================
# OPTIMIZED TRAINING FUNCTION
# ============================================================================

def train_optimized(config):
    """Main training function with all optimizations"""
    
    # Setup GPU optimizations
    gpus = setup_gpu_optimization()
    
    # Load data
    print("\n📁 Loading training data...")
    data_dir = Path('./data/splits')
    
    if not data_dir.exists():
        print("⚠ Data not found. Please run setup_all_disease_datasets.py first.")
        return
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Validation samples: {len(X_val):,}")
    
    # Optimal batch size for RTX 3060 Ti (8GB VRAM)
    # Larger batch = better GPU utilization
    batch_size = config.get('batch_size', 64)  # 64 is optimal for 3060 Ti
    
    # Calculate steps
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Epochs: {config['epochs']}")
    
    # Create optimized data generators
    print("\n🔄 Creating optimized data pipeline...")
    
    # Use multiple workers for CPU preprocessing
    num_workers = min(8, psutil.cpu_count())  # Ryzen 7 has 8 cores
    
    train_gen = OptimizedDataGenerator(
        X_train, y_train, 
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_size=3
    )
    
    val_gen = OptimizedDataGenerator(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_size=2
    )
    
    train_dataset = train_gen.create_dataset(training=True)
    val_dataset = val_gen.create_dataset(training=False)
    
    # Build and compile model
    print("\n🏗️ Building optimized model...")
    
    # Build model
    model, base_model = build_fixed_model(
        num_classes=config['num_classes'],
        input_shape=(224, 224, 3)
    )
    
    # Compile with optimized settings
    initial_lr = config.get('learning_rate', 0.001)
    
    # Use optimizer with mixed precision if available
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Apply mixed precision optimizer if policy is set
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"✓ Model compiled with mixed precision")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    # Callbacks for optimization
    callbacks = [
        # Performance monitoring
        PerformanceMonitor(),
        
        # Adaptive learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
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
        
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model_gpu_optimized.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # TensorBoard for profiling (optional)
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/gpu_optimized',
            histogram_freq=0,  # Disable histograms for speed
            profile_batch='10,20',  # Profile batches 10-20
            update_freq='batch'
        )
    ]
    
    # Training with optimizations
    print("\n" + "="*70)
    print("🚀 STARTING GPU-OPTIMIZED TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config['epochs'],
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        workers=1,  # Data loading is already parallelized
        use_multiprocessing=False,  # We handle threading ourselves
        max_queue_size=20  # Larger queue for better throughput
    )
    
    total_time = time.time() - start_time
    
    # Print final statistics
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED")
    print("="*70)
    
    print(f"\n📊 Final Results:")
    print(f"   Total training time: {total_time/60:.1f} minutes")
    print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Calculate throughput
    total_samples = len(X_train) * config['epochs']
    avg_throughput = total_samples / total_time
    print(f"   Average throughput: {avg_throughput:.1f} samples/sec")
    
    # Save model
    print("\n💾 Saving optimized model...")
    model.save('models/plant_disease_gpu_optimized.h5')
    
    # Save training history
    with open('models/training_history_gpu_optimized.json', 'w') as f:
        json.dump(history.history, f, indent=2)
    
    print("✓ Model and history saved!")
    
    return model, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Training configuration optimized for RTX 3060 Ti
    config = {
        'batch_size': 64,      # Optimal for 8GB VRAM
        'epochs': 50,          # Will early stop if needed
        'num_classes': 7,
        'learning_rate': 0.001,
    }
    
    print("\n" + "="*70)
    print("   GPU-OPTIMIZED TRAINING FOR RTX 3060 Ti + RYZEN 7")
    print("="*70)
    
    # Check system
    print("\n💻 System Information:")
    print(f"   CPU: {psutil.cpu_count(logical=False)} cores, {psutil.cpu_count()} threads")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"   GPU: {gpu.name}")
        print(f"   VRAM: {gpu.memoryTotal} MB")
    
    # Start training
    model, history = train_optimized(config)