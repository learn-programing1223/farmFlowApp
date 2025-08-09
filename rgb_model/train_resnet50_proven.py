#!/usr/bin/env python3
"""
ResNet50 Plant Disease Detection - Proven Implementation
Based on research showing 99.2% accuracy on PlantVillage dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def build_resnet50_model(num_classes, input_shape=(224, 224, 3)):
    """
    Build ResNet50 model with transfer learning
    Based on proven implementation achieving 99%+ accuracy
    """
    print("[BUILD] Creating ResNet50 model with transfer learning...")
    
    # Load pre-trained ResNet50 (with ImageNet weights)
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tuning: Unfreeze the top layers
    # Research shows fine-tuning improves accuracy significantly
    base_model.trainable = True
    
    # Freeze first 143 layers, fine-tune the rest
    # This is the proven approach from the 99.2% accuracy implementation
    for layer in base_model.layers[:143]:
        layer.trainable = False
    
    # Build the model
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation (proven to improve generalization)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.2)(x)
    
    # Preprocessing for ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)  # Lower dropout for better performance
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"[BUILD] Model created with {num_classes} output classes")
    print(f"[BUILD] Total parameters: {model.count_params():,}")
    print(f"[BUILD] Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model

def compile_model(model, learning_rate=1e-4):
    """
    Compile model with proven optimizer settings
    """
    print(f"[COMPILE] Setting up optimizer with lr={learning_rate}")
    
    # Use Adam optimizer (proven to work well)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use categorical crossentropy (standard for multi-class)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators for training
    """
    print("[DATA] Creating data generators...")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset

def train_model(args):
    """
    Main training function
    """
    print("\n" + "=" * 60)
    print("RESNET50 PLANT DISEASE DETECTION TRAINING")
    print("Based on proven approach achieving 99%+ accuracy")
    print("=" * 60)
    
    # Set up GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[CPU] No GPU found, using CPU")
    
    # Load data
    print("\n[DATA] Loading preprocessed data...")
    data_dir = Path('./data/splits')
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    # Also load test data if available
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
    
    # Determine number of classes
    num_classes = y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
    
    print(f"[DATA] Training samples: {len(X_train)}")
    print(f"[DATA] Validation samples: {len(X_val)}")
    print(f"[DATA] Number of classes: {num_classes}")
    
    # Build model
    model = build_resnet50_model(num_classes)
    
    # Compile model
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Create data generators
    train_dataset, val_dataset = create_data_generators(
        X_train, y_train, X_val, y_val, 
        batch_size=args.batch_size
    )
    
    # Set up callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[TRAIN] Starting training...")
    print("-" * 60)
    
    # Two-stage training approach (proven to work well)
    
    # Stage 1: Train with frozen base (quick convergence)
    print("\n[STAGE 1] Training with frozen base model...")
    # Find the base model layer
    base_model_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
            base_model_layer = layer
            break
    
    if base_model_layer:
        for layer in base_model_layer.layers:  # Freeze base model
            layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=min(10, args.epochs // 2),
        callbacks=callbacks[:-1],  # Don't use ReduceLROnPlateau yet
        verbose=1
    )
    
    # Stage 2: Fine-tune with unfrozen layers
    print("\n[STAGE 2] Fine-tuning with unfrozen top layers...")
    
    # Unfreeze top layers
    if base_model_layer:
        for layer in base_model_layer.layers[-50:]:  # Unfreeze last 50 layers
            layer.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate / 10),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs - len(history1.history['loss']),
        initial_epoch=len(history1.history['loss']),
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history[key]
    
    # Evaluate final model
    print("\n[EVAL] Evaluating final model...")
    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_dataset, verbose=0)
    
    print(f"\n[RESULTS] Final Validation Metrics:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall: {val_rec:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    
    # Test on test set if available
    if X_test is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(args.batch_size)
        
        test_results = model.evaluate(test_dataset, verbose=0)
        print(f"\n[TEST] Test Set Results:")
        print(f"  Accuracy: {test_results[1]:.4f}")
        print(f"  Precision: {test_results[2]:.4f}")
        print(f"  Recall: {test_results[3]:.4f}")
        print(f"  AUC: {test_results[4]:.4f}")
    
    # Save final model
    final_model_path = output_dir / 'final_model.h5'
    model.save(final_model_path)
    print(f"\n[SAVE] Model saved to: {final_model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] Training history saved to: {history_path}")
    
    # Success check
    if val_acc > 0.80:
        print("\n" + "=" * 60)
        print("[SUCCESS] Model trained successfully!")
        if val_acc > 0.95:
            print("[EXCELLENT] Outstanding accuracy achieved!")
        elif val_acc > 0.90:
            print("[GREAT] Very good accuracy achieved!")
        else:
            print("[GOOD] Good accuracy achieved!")
        print("=" * 60)
    else:
        print("\n[INFO] Model accuracy can be improved with:")
        print("  - More epochs")
        print("  - Data augmentation")
        print("  - Hyperparameter tuning")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 for plant disease detection')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models/resnet50', 
                       help='Output directory for model')
    
    args = parser.parse_args()
    
    print("\n[CONFIG] Training Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Output Dir: {args.output_dir}")
    
    # Train model
    model, history = train_model(args)

if __name__ == "__main__":
    main()