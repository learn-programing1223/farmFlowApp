#!/usr/bin/env python3
"""
Training script using the FIXED model implementation
This addresses all the issues that were causing training to fail
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Import the FIXED model implementation
from model_fixed import build_fixed_model, compile_fixed_model


def load_existing_data(data_dir='./data/splits'):
    """Load preprocessed data from previous runs"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory {data_dir} not found. "
            "Please run setup_all_disease_datasets.py first."
        )
    
    print("Loading preprocessed data...")
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    # Also load test if available
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_generator(X, y, batch_size=32, training=True):
    """Create a data generator for memory efficiency"""
    
    def generator():
        indices = np.arange(len(X))
        if training:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Simple augmentation for training
            if training:
                # Random flip
                if np.random.random() < 0.5:
                    batch_x = batch_x[:, :, ::-1, :]
                
                # Random brightness
                if np.random.random() < 0.3:
                    brightness = np.random.uniform(0.8, 1.2)
                    batch_x = np.clip(batch_x * brightness, 0, 1)
            
            yield batch_x, batch_y
    
    # Determine output signature
    output_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, y.shape[1]), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)


def train_fixed_model(args):
    """Main training function with fixed model"""
    
    print("="*60)
    print("[FIXED] TRAINING WITH FIXED MODEL IMPLEMENTATION")
    print("="*60)
    print("\nThis version fixes:")
    print("  [OK] Focal Loss calculation")
    print("  [OK] Learning rate (0.001 instead of 0.0005)")
    print("  [OK] Loss function setup (from_logits=True)")
    print("  [OK] Architecture improvements")
    print("="*60)
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("\n[WARNING] No GPU found, using CPU")
    
    # Load existing data
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_existing_data()
        
        print(f"\n[DATA] Data loaded successfully:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        if X_test is not None:
            print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Classes: {y_train.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nTo generate data, run:")
        print("  python setup_all_disease_datasets.py")
        print("  python train_disease_focused.py --samples-per-class 1000")
        return
    
    # Determine number of classes
    num_classes = y_train.shape[1]
    
    # Build the FIXED model
    print(f"\n Building fixed model for {num_classes} classes...")
    model, base_model = build_fixed_model(
        num_classes=num_classes,
        input_shape=(224, 224, 3)
    )
    
    # Compile with FIXED settings
    print("\n Compiling with fixed Focal Loss...")
    model = compile_fixed_model(
        model,
        learning_rate=args.learning_rate,
        use_focal_loss=args.use_focal_loss
    )
    
    # Print model summary
    print("\n Model Summary:")
    print(f"  Total parameters: {model.count_params():,}")
    if base_model:
        print(f"  Base model: EfficientNetB0")
        print(f"  Base trainable: {base_model.trainable}")
    
    # Setup callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model_fixed.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / 'training_log_fixed.csv')
        )
    ]
    
    # Add TensorBoard if requested
    if args.tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dir / 'tensorboard_fixed'),
                histogram_freq=1
            )
        )
    
    # Training
    print("\n[START] Starting training with FIXED model...")
    print("-"*60)
    
    # Use subset if specified
    if args.subset_size > 0:
        subset_size = min(args.subset_size, len(X_train))
        print(f"Using subset of {subset_size} samples for quick test")
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
    
    # Create data generators for memory efficiency
    if args.use_generator:
        print("Using data generators for memory efficiency")
        train_dataset = create_data_generator(
            X_train, y_train,
            batch_size=args.batch_size,
            training=True
        )
        val_dataset = create_data_generator(
            X_val, y_val,
            batch_size=args.batch_size,
            training=False
        )
        
        steps_per_epoch = len(X_train) // args.batch_size
        validation_steps = len(X_val) // args.batch_size
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Direct fit
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    print("-"*60)
    
    # Evaluate results
    print("\n[COMPLETE] Training Complete! Analyzing results...")
    
    # Get final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n[METRICS] Final Metrics:")
    print(f"  Training Accuracy: {final_train_acc:.3f} ({final_train_acc*100:.1f}%)")
    print(f"  Validation Accuracy: {final_val_acc:.3f} ({final_val_acc*100:.1f}%)")
    print(f"  Training Loss: {final_train_loss:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    
    # Check improvement
    initial_train_acc = history.history['accuracy'][0]
    initial_val_acc = history.history['val_accuracy'][0]
    
    train_improvement = final_train_acc - initial_train_acc
    val_improvement = final_val_acc - initial_val_acc
    
    print(f"\n[IMPROVEMENT] Improvement:")
    print(f"  Training: +{train_improvement:.3f} ({train_improvement*100:.1f}%)")
    print(f"  Validation: +{val_improvement:.3f} ({val_improvement*100:.1f}%)")
    
    # Test on test set if available
    if X_test is not None:
        print("\n Evaluating on test set...")
        test_results = model.evaluate(X_test, y_test, verbose=1)
        
        # Extract metrics (order may vary)
        test_loss = test_results[0]
        test_acc = test_results[1]
        
        print(f"\n[TEST] Test Results:")
        print(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
        print(f"  Test Loss: {test_loss:.4f}")
        
        # Per-class accuracy
        categories = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew',
                     'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
        
        if num_classes == len(categories):
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            print("\n[CLASSES] Per-Class Accuracy:")
            for i, cat in enumerate(categories[:num_classes]):
                mask = y_true_classes == i
                if mask.sum() > 0:
                    class_acc = (y_pred_classes[mask] == i).mean()
                    print(f"  {cat}: {class_acc:.3f} ({class_acc*100:.1f}%)")
    
    # Save final model
    final_model_path = output_dir / 'final_model_fixed.keras'
    model.save(str(final_model_path))
    print(f"\n[SAVED] Model saved to: {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'history_fixed.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"[SAVED] Training history saved to: {history_path}")
    
    # Final assessment
    print("\n" + "="*60)
    if final_val_acc > 0.3:  # Better than 30%
        print("[SUCCESS] Model is learning properly with fixes!")
        if final_val_acc > 0.7:
            print("[EXCELLENT] Excellent results! Model performing very well!")
        elif final_val_acc > 0.5:
            print(" Good results! Model is converging nicely!")
        else:
            print("[PROGRESS] Model is learning! May need more epochs or data.")
    else:
        print("[WARNING] Model accuracy still low. Suggestions:")
        print("  - Try more epochs (--epochs 20)")
        print("  - Check data quality")
        print("  - Try without focal loss (--no-focal-loss)")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with FIXED model implementation')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use focal loss (default: True)')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                       help='Use standard crossentropy instead of focal loss')
    parser.add_argument('--output-dir', type=str, default='./models/fixed',
                       help='Output directory for models')
    parser.add_argument('--subset-size', type=int, default=0,
                       help='Use subset of data for quick test (0=use all)')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--use-generator', action='store_true', default=True,
                       help='Use data generator for memory efficiency')
    
    args = parser.parse_args()
    
    print("\n[FIXED] Model Training Script")
    print(f"Settings:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Focal Loss: {args.use_focal_loss}")
    print(f"  Output Dir: {args.output_dir}")
    
    train_fixed_model(args)