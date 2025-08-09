#!/usr/bin/env python3
"""
Benchmark script to test GPU optimization performance
Compares standard vs optimized training speed
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import psutil
import GPUtil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model_fixed import build_fixed_model, compile_fixed_model


def benchmark_standard():
    """Benchmark without optimizations"""
    print("\n" + "="*60)
    print("📊 STANDARD TRAINING (No optimizations)")
    print("="*60)
    
    # Standard setup
    tf.keras.backend.clear_session()
    
    # Create model
    model, _ = build_fixed_model(num_classes=7)
    model = compile_fixed_model(model)
    
    # Create dummy data
    batch_size = 32
    num_samples = 1000
    X = np.random.random((num_samples, 224, 224, 3)).astype(np.float32)
    y = np.eye(7)[np.random.randint(0, 7, num_samples)]
    
    # Time training
    start_time = time.time()
    
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=3,
        verbose=0
    )
    
    standard_time = time.time() - start_time
    samples_per_sec = (num_samples * 3) / standard_time
    
    print(f"⏱ Time: {standard_time:.2f}s")
    print(f"📈 Throughput: {samples_per_sec:.1f} samples/sec")
    
    return standard_time, samples_per_sec


def benchmark_optimized():
    """Benchmark with all optimizations"""
    print("\n" + "="*60)
    print("🚀 GPU-OPTIMIZED TRAINING")
    print("="*60)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Enable GPU optimizations
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Enable XLA
    tf.config.optimizer.set_jit(True)
    
    # Optimize threading
    num_cores = psutil.cpu_count(logical=False)
    num_threads = psutil.cpu_count(logical=True)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    print(f"✓ Mixed Precision: Enabled")
    print(f"✓ XLA JIT: Enabled")
    print(f"✓ CPU Threads: {num_threads}")
    
    # Create model with mixed precision
    with tf.keras.mixed_precision.LossScaleOptimizer.experimental_loss_scale():
        model, _ = build_fixed_model(num_classes=7)
    
    # Compile with mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create dummy data with larger batch size
    batch_size = 64  # Larger batch for GPU
    num_samples = 1000
    X = np.random.random((num_samples, 224, 224, 3)).astype(np.float32)
    y = np.eye(7)[np.random.randint(0, 7, num_samples)]
    
    # Create optimized dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Time training
    start_time = time.time()
    
    history = model.fit(
        dataset,
        epochs=3,
        verbose=0
    )
    
    optimized_time = time.time() - start_time
    samples_per_sec = (num_samples * 3) / optimized_time
    
    print(f"⏱ Time: {optimized_time:.2f}s")
    print(f"📈 Throughput: {samples_per_sec:.1f} samples/sec")
    
    return optimized_time, samples_per_sec


def main():
    """Run benchmarks and compare"""
    
    print("\n" + "🔥"*30)
    print("   GPU OPTIMIZATION BENCHMARK")
    print("🔥"*30)
    
    # System info
    print("\n💻 System Information:")
    print(f"   CPU: {psutil.cpu_count(logical=False)} cores, {psutil.cpu_count()} threads")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"   GPU: {gpu.name}")
        print(f"   VRAM: {gpu.memoryTotal} MB")
    else:
        print("   GPU: Not detected")
    
    # Run benchmarks
    print("\n🏁 Running benchmarks...")
    
    # Standard benchmark
    standard_time, standard_throughput = benchmark_standard()
    
    # Optimized benchmark
    optimized_time, optimized_throughput = benchmark_optimized()
    
    # Calculate improvement
    speedup = standard_time / optimized_time
    throughput_gain = optimized_throughput / standard_throughput
    
    # Results
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n🐌 Standard Training:")
    print(f"   Time: {standard_time:.2f}s")
    print(f"   Throughput: {standard_throughput:.1f} samples/sec")
    
    print(f"\n🚀 GPU-Optimized Training:")
    print(f"   Time: {optimized_time:.2f}s")
    print(f"   Throughput: {optimized_throughput:.1f} samples/sec")
    
    print(f"\n📈 PERFORMANCE GAINS:")
    print(f"   Speed-up: {speedup:.2f}x faster")
    print(f"   Throughput increase: {throughput_gain:.2f}x")
    print(f"   Time saved: {(1 - 1/speedup)*100:.1f}%")
    
    if speedup > 2:
        print("\n✅ Excellent optimization! Training is significantly faster.")
    elif speedup > 1.5:
        print("\n✅ Good optimization! Notable performance improvement.")
    elif speedup > 1.2:
        print("\n⚠️ Moderate optimization. Check GPU utilization.")
    else:
        print("\n❌ Limited improvement. Verify GPU is being used.")


if __name__ == "__main__":
    main()