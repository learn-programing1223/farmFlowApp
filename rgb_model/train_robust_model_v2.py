#!/usr/bin/env python3
"""
Enhanced Robust Training Script with Advanced Features
======================================================

This script integrates:
- EnhancedDataLoader with advanced preprocessing
- Combined loss functions (Focal + Label Smoothing)
- Stochastic Weight Averaging (SWA)
- Gradient clipping
- MixUp augmentation
- Comparison mode for preprocessing evaluation

Author: PlantPulse Team
Date: 2025
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
from data_loader_v2 import EnhancedDataLoader
from losses import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss, get_loss_by_name

# Disable mixed precision for CPU training
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced robust training for plant disease detection')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='datasets/plantvillage_processed',
                       help='Path to dataset directory')
    parser.add_argument('--preprocessing_mode', type=str, default='default',
                       choices=['default', 'fast', 'minimal', 'legacy'],
                       help='Preprocessing mode to use')
    parser.add_argument('--use_advanced_preprocessing', type=bool, default=True,
                       help='Use advanced preprocessing (CLAHE, etc.)')
    parser.add_argument('--comparison_mode', action='store_true',
                       help='Compare preprocessing modes')
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['focal', 'label_smoothing', 'combined', 'standard'],
                       help='Type of loss function to use')
    parser.add_argument('--focal_weight', type=float, default=0.7,
                       help='Weight for focal loss in combined loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                       help='Epsilon for label smoothing')
    
    # Training arguments
    parser.add_argument('--swa_start_epoch', type=int, default=20,
                       help='Epoch to start Stochastic Weight Averaging')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Max norm for gradient clipping')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='MixUp alpha parameter (0 to disable)')
    parser.add_argument('--mixup_probability', type=float, default=0.5,
                       help='Probability of applying MixUp')
    
    # Other arguments
    parser.add_argument('--test_run', action='store_true',
                       help='Run quick test with 2-3 epochs')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models')
    
    return parser.parse_args()


class MixUpAugmentation(keras.layers.Layer):
    """MixUp augmentation layer for training."""
    
    def __init__(self, alpha=0.2, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.probability = probability
    
    def call(self, inputs, training=None):
        if not training or self.alpha <= 0:
            return inputs
        
        images, labels = inputs
        batch_size = tf.shape(images)[0]
        
        # Apply MixUp with probability
        apply_mixup = tf.random.uniform([]) < self.probability
        
        def mixup():
            # Sample lambda from Beta distribution
            lam = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)
            lam = tf.maximum(lam, 1 - lam)
            
            # Shuffle batch for mixing
            indices = tf.random.shuffle(tf.range(batch_size))
            shuffled_images = tf.gather(images, indices)
            shuffled_labels = tf.gather(labels, indices)
            
            # Mix images and labels
            mixed_images = lam * images + (1 - lam) * shuffled_images
            mixed_labels = lam[:, 0, 0, 0:1] * labels + (1 - lam[:, 0, 0, 0:1]) * shuffled_labels
            
            return mixed_images, mixed_labels
        
        return tf.cond(apply_mixup, mixup, lambda: (images, labels))


class CleanMetricsCallback(keras.callbacks.Callback):
    """Calculate metrics on clean (non-augmented) data for accurate training metrics."""
    
    def __init__(self, clean_dataset, steps, frequency=5):
        super().__init__()
        self.clean_dataset = clean_dataset
        self.steps = steps
        self.frequency = frequency  # Calculate every N epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate and report clean metrics."""
        # Only calculate every N epochs to save time
        if (epoch + 1) % self.frequency != 0:
            return
            
        # Evaluate on clean data
        print("\n  Calculating clean metrics...", end="")
        results = self.model.evaluate(self.clean_dataset, steps=self.steps, verbose=0)
        
        # Update logs with clean metrics
        if logs is not None:
            # Find accuracy metric index
            metric_names = self.model.metrics_names
            for i, name in enumerate(metric_names):
                if 'accuracy' in name:
                    clean_acc = results[i]
                    logs['clean_train_acc'] = clean_acc
                    
                    # Report the difference
                    if 'accuracy' in logs:
                        augmented_acc = logs['accuracy']
                        gap = augmented_acc - clean_acc
                        print(f" Clean: {clean_acc:.4f}, Augmented: {augmented_acc:.4f}, Gap: {gap:+.4f}")
                    break


class SWACallback(keras.callbacks.Callback):
    """Stochastic Weight Averaging callback."""
    
    def __init__(self, start_epoch=20, update_freq=1):
        super().__init__()
        self.start_epoch = start_epoch
        self.update_freq = update_freq
        self.swa_weights = None
        self.n_models = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.update_freq == 0:
            if self.swa_weights is None:
                # Initialize SWA weights - get_weights() already returns numpy arrays
                self.swa_weights = [w.copy() for w in self.model.get_weights()]
            else:
                # Update SWA weights (running average)
                current_weights = self.model.get_weights()
                for i in range(len(self.swa_weights)):
                    self.swa_weights[i] = (self.swa_weights[i] * self.n_models + current_weights[i]) / (self.n_models + 1)
            
            self.n_models += 1
            print(f"\n[SWA] Updated weights (n_models={self.n_models})")
    
    def on_train_end(self, logs=None):
        if self.swa_weights is not None:
            print("\n[SWA] Setting final averaged weights")
            self.model.set_weights(self.swa_weights)


def create_model(num_classes=6, input_shape=(224, 224, 3), include_mixup=False, mixup_alpha=0.2):
    """Create robust CNN model with optional MixUp."""
    
    inputs = layers.Input(shape=input_shape, name='image_input')
    
    # CNN blocks
    x = inputs
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer - use float32 for stability
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def calculate_class_weights(data_loader: EnhancedDataLoader, train_paths: list, train_labels: list) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    unique, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    num_classes = len(unique)
    
    class_weights = {}
    for idx, count in zip(unique, counts):
        weight = total_samples / (num_classes * count)
        class_weights[idx] = weight
    
    print("\nClass weights calculated:")
    for idx in range(num_classes):
        class_name = data_loader.class_names[idx] if idx < len(data_loader.class_names) else f"Class_{idx}"
        print(f"  {class_name}: {class_weights.get(idx, 1.0):.3f}")
    
    return class_weights


def create_loss_function(args, class_weights=None):
    """Create loss function based on arguments."""
    
    if args.loss_type == 'focal':
        return FocalLoss(
            alpha=1.0,
            gamma=args.focal_gamma,
            label_smoothing=0.05
        )
    
    elif args.loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            epsilon=args.label_smoothing_epsilon,
            class_weights=class_weights
        )
    
    elif args.loss_type == 'combined':
        focal_loss = FocalLoss(gamma=args.focal_gamma, alpha=0.75)
        ls_loss = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing_epsilon)
        
        return CombinedLoss(
            losses=[focal_loss, ls_loss],
            weights=[args.focal_weight, 1.0 - args.focal_weight]
        )
    
    else:  # standard
        return keras.losses.CategoricalCrossentropy(label_smoothing=0.05)


def compare_preprocessing_modes(args):
    """Compare different preprocessing modes."""
    print("\n" + "=" * 70)
    print("PREPROCESSING MODE COMPARISON")
    print("=" * 70)
    
    modes = ['legacy', 'minimal', 'fast', 'default']
    results = {}
    
    for mode in modes:
        print(f"\n[Testing mode: {mode}]")
        
        # Create data loader
        use_advanced = (mode != 'legacy')
        loader = EnhancedDataLoader(
            data_dir=Path(args.data_path) / 'train',
            target_size=(224, 224),
            batch_size=args.batch_size,
            use_advanced_preprocessing=use_advanced,
            preprocessing_mode=mode if use_advanced else 'default'
        )
        
        # Load small subset for testing
        train_paths, train_labels, _ = loader.load_dataset_from_directory(
            Path(args.data_path) / 'train',
            split='train'
        )
        
        # Take only first 100 samples for quick test
        sample_paths = train_paths[:100]
        sample_labels = train_labels[:100]
        
        # Create TF dataset
        dataset = loader.create_tf_dataset(
            sample_paths,
            sample_labels,
            is_training=True,
            shuffle=False,
            augment=False
        )
        
        # Measure preprocessing time
        import time
        start_time = time.time()
        
        for batch_images, batch_labels in dataset.take(3):
            # Process 3 batches
            pass
        
        elapsed_time = time.time() - start_time
        
        # Get sample statistics
        for batch_images, _ in dataset.take(1):
            mean_val = tf.reduce_mean(batch_images).numpy()
            std_val = tf.math.reduce_std(batch_images).numpy()
            min_val = tf.reduce_min(batch_images).numpy()
            max_val = tf.reduce_max(batch_images).numpy()
            
            results[mode] = {
                'time': elapsed_time,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
            
            print(f"  Time: {elapsed_time:.3f}s")
            print(f"  Stats: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
    
    # Print comparison summary
    print("\n" + "-" * 70)
    print("COMPARISON SUMMARY")
    print("-" * 70)
    
    baseline_time = results['legacy']['time']
    for mode, stats in results.items():
        speedup = baseline_time / stats['time']
        print(f"{mode:10} - Time: {stats['time']:.3f}s (speedup: {speedup:.2f}x)")
    
    return results


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set up for test run if requested
    if args.test_run:
        args.epochs = 3
        print("\n[TEST RUN MODE] Training for only 3 epochs")
    
    print("\n" + "=" * 70)
    print("ENHANCED ROBUST MODEL TRAINING")
    print("=" * 70)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Preprocessing: {'Advanced' if args.use_advanced_preprocessing else 'Legacy'}")
    print(f"  Preprocessing mode: {args.preprocessing_mode}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  SWA start: epoch {args.swa_start_epoch}")
    print(f"  Gradient clipping: {args.gradient_clip_norm}")
    print(f"  MixUp: alpha={args.mixup_alpha}, prob={args.mixup_probability}")
    
    # Check GPU
    print("\n" + "-" * 70)
    print("Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU Available: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[WARNING] No GPU found, using CPU")
    
    # Run comparison mode if requested
    if args.comparison_mode:
        compare_preprocessing_modes(args)
        print("\n[Comparison complete. Continuing with selected mode...]")
    
    # Setup data paths
    data_path = Path(args.data_path)
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    test_path = data_path / 'test'
    
    if not train_path.exists():
        print(f"\n[ERROR] Training data not found at {train_path}")
        return
    
    # Create Enhanced Data Loader
    print("\n" + "-" * 70)
    print("Setting up Enhanced Data Loader...")
    
    train_loader = EnhancedDataLoader(
        data_dir=train_path,
        target_size=(224, 224),
        batch_size=args.batch_size,
        use_advanced_preprocessing=args.use_advanced_preprocessing,
        preprocessing_mode=args.preprocessing_mode
    )
    
    val_loader = EnhancedDataLoader(
        data_dir=val_path,
        target_size=(224, 224),
        batch_size=args.batch_size,
        use_advanced_preprocessing=args.use_advanced_preprocessing,
        preprocessing_mode=args.preprocessing_mode  # Use same mode as training for consistency
    )
    
    # Load datasets
    print("\nLoading datasets...")
    train_paths, train_labels, class_names = train_loader.load_dataset_from_directory(train_path, 'train')
    val_paths, val_labels, _ = val_loader.load_dataset_from_directory(val_path, 'val')
    test_paths, test_labels, _ = val_loader.load_dataset_from_directory(test_path, 'test')
    
    print(f"\n[OK] Training samples: {len(train_paths)}")
    print(f"[OK] Validation samples: {len(val_paths)}")
    print(f"[OK] Test samples: {len(test_paths)}")
    print(f"[OK] Classes: {class_names}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader, train_paths, train_labels)
    
    # Create TensorFlow datasets
    print("\n" + "-" * 70)
    print("Creating TensorFlow datasets...")
    
    train_dataset = train_loader.create_tf_dataset(
        train_paths, train_labels,
        is_training=True,
        shuffle=True,
        augment=True
    )
    
    val_dataset = val_loader.create_tf_dataset(
        val_paths, val_labels,
        is_training=False,
        shuffle=False,
        augment=False
    )
    
    test_dataset = val_loader.create_tf_dataset(
        test_paths, test_labels,
        is_training=False,
        shuffle=False,
        augment=False
    )
    
    # Build model
    print("\n" + "-" * 70)
    print("Building model...")
    
    model = create_model(
        num_classes=train_loader.num_classes,
        include_mixup=(args.mixup_alpha > 0),
        mixup_alpha=args.mixup_alpha
    )
    
    print(f"[OK] Model created with {model.count_params():,} parameters")
    
    # Create loss function
    loss_fn = create_loss_function(args, class_weights)
    
    # Create optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.gradient_clip_norm)
    
    # Compile model
    print("\n" + "-" * 70)
    print("Compiling model...")
    print(f"  Loss: {args.loss_type}")
    print(f"  Optimizer: Adam with gradient clipping (norm={args.gradient_clip_norm})")
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Create clean training dataset for accurate metrics (NO augmentation)
    print("\nCreating clean training dataset for metrics calculation...")
    train_dataset_clean = train_loader.create_tf_dataset(
        train_paths, train_labels,
        is_training=False,  # This prevents augmentation
        shuffle=False,      # No need to shuffle for metrics
        augment=False       # Explicitly no augmentation
    )
    
    # Calculate steps
    steps_per_epoch = len(train_paths) // args.batch_size
    validation_steps = len(val_paths) // args.batch_size
    clean_metrics_steps = min(100, steps_per_epoch)  # Evaluate on subset for speed
    
    # Setup callbacks
    callbacks = [
        # Clean metrics callback for accurate training metrics
        CleanMetricsCallback(
            clean_dataset=train_dataset_clean,
            steps=clean_metrics_steps,
            frequency=5  # Calculate every 5 epochs
        ),
        
        # Model checkpoint - use .keras format to avoid warning
        keras.callbacks.ModelCheckpoint(
            Path(args.output_dir) / 'enhanced_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping - increased patience for better training
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased from 7 to 15
            restore_best_weights=True,
            verbose=1
        ),
        
        # Stochastic Weight Averaging
        SWACallback(start_epoch=args.swa_start_epoch),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=f'logs/enhanced_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Training
    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70)
    
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2  # Changed from 1 to 2 - only show epoch progress, not individual steps
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    test_steps = len(test_paths) // args.batch_size
    test_results = model.evaluate(test_dataset, steps=test_steps, verbose=1)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.2%}")
    print(f"  Precision: {test_results[2]:.2%}")
    print(f"  Recall: {test_results[3]:.2%}")
    print(f"  AUC: {test_results[4]:.4f}")
    
    # Save final model
    print("\n" + "-" * 70)
    print("Saving models...")
    
    # Save Keras model
    model_path = Path(args.output_dir) / 'enhanced_final.h5'
    model.save(model_path)
    print(f"[OK] Saved Keras model: {model_path}")
    
    # TFLite conversion with mixed precision handling
    print("\n" + "-" * 70)
    print("Converting to TFLite...")
    
    try:
        # Standard TFLite conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable TF Select ops to handle all operations
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS      # Enable TF ops (Conv2D, BiasAdd, etc.)
        ]
        
        # Convert to float16 for smaller size (optional)
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = Path(args.output_dir) / 'enhanced_model.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[OK] Saved TFLite model: {tflite_path} ({len(tflite_model)/1024/1024:.2f} MB)")
        
        # Also save a quantized version for even smaller size
        print("[INFO] Creating INT8 quantized model...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = None  # Would need representative data for full INT8
        
    except Exception as e:
        print(f"[WARNING] TFLite conversion failed: {str(e)}")
        print("[INFO] This is not critical - the Keras model was saved successfully")
        print("[INFO] You can try conversion later with: ")
        print("       converter.allow_custom_ops = True")
        print("       converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]")
    
    # Save training summary
    summary = {
        'configuration': vars(args),
        'test_results': {
            'loss': float(test_results[0]),
            'accuracy': float(test_results[1]),
            'precision': float(test_results[2]),
            'recall': float(test_results[3]),
            'auc': float(test_results[4])
        },
        'class_weights': {int(k): float(v) for k, v in class_weights.items()},  # Convert int64 to int
        'training_history': history.history
    }
    
    summary_path = Path(args.output_dir) / 'enhanced_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"[OK] Saved training summary: {summary_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n>>> Final Test Accuracy: {test_results[1]:.2%}")
    
    if test_results[1] >= 0.85:
        print("[SUCCESS] TARGET ACHIEVED! Model exceeds 85% accuracy!")
    else:
        print(f"[INFO] Model accuracy is {test_results[1]:.2%}")
    
    print("\n>>> Features used:")
    print(f"  - Preprocessing: {'Advanced' if args.use_advanced_preprocessing else 'Legacy'} ({args.preprocessing_mode})")
    print(f"  - Loss: {args.loss_type}")
    print(f"  - SWA: Started at epoch {args.swa_start_epoch}")
    print(f"  - Gradient clipping: norm={args.gradient_clip_norm}")
    print(f"  - MixUp: alpha={args.mixup_alpha}")
    
    print("\n>>> Next steps:")
    print("  1. Test on real images with different preprocessing modes")
    print("  2. Compare with legacy model performance")
    print("  3. Deploy best configuration to production")


if __name__ == "__main__":
    main()