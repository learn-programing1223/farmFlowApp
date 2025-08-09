#!/usr/bin/env python3
"""
ULTIMATE FIX - Works around all TensorFlow/Keras issues
Uses ResNet50 which is more stable than EfficientNet
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def build_working_model(num_classes=7, input_shape=(224, 224, 3)):
    """
    Build a model that ACTUALLY WORKS
    Using ResNet50 - more stable than EfficientNet
    """
    
    # Use ResNet50 - much more stable than EfficientNet
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # CRITICAL: Unfreeze for fine-tuning (research shows this is essential)
    base_model.trainable = True
    
    # Freeze only first 50 layers initially
    for layer in base_model.layers[:50]:
        layer.trainable = False
    
    # Build the full model
    inputs = tf.keras.Input(shape=input_shape)
    
    # CRITICAL FIX: Handle [0,1] data properly
    # ResNet50 expects [-1, 1] range after preprocessing
    x = inputs * 255.0  # Scale [0,1] to [0,255]
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    # Pass through base model
    x = base_model(x)
    
    # Add custom top layers
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model


def create_augmented_data(X_train, y_train, X_val, y_val, batch_size=32):
    """Create augmented data generators"""
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Strong augmentation for training (research-recommended)
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    return train_gen, val_gen


def progressive_unfreeze(base_model, epoch):
    """Progressive unfreezing strategy from research"""
    if epoch == 3:
        print("\n[PHASE 2] Unfreezing middle layers...")
        for layer in base_model.layers[50:100]:
            layer.trainable = True
    elif epoch == 7:
        print("\n[PHASE 3] Unfreezing most layers...")
        for layer in base_model.layers[100:]:
            layer.trainable = True
    elif epoch == 10:
        print("\n[PHASE 4] Full fine-tuning...")
        base_model.trainable = True


def train_ultimate():
    """Main training function with all fixes"""
    
    print("\n" + "="*70)
    print("ULTIMATE FIX TRAINING")
    print("Using ResNet50 with proper preprocessing")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dir = Path('./data/splits')
    
    if not data_dir.exists():
        print("[ERROR] Data not found!")
        return None, None
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy').astype(np.float32)
    y_train = np.load(data_dir / 'y_train.npy').astype(np.float32)
    X_val = np.load(data_dir / 'X_val.npy').astype(np.float32)
    y_val = np.load(data_dir / 'y_val.npy').astype(np.float32)
    
    print(f"Loaded {len(X_train):,} training samples")
    print(f"Loaded {len(X_val):,} validation samples")
    print(f"Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    # Build model
    print("\nBuilding ResNet50 model...")
    model, base_model = build_working_model(num_classes=7)
    
    # Count trainable params
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Non-trainable params: {non_trainable_count:,}")
    
    # Compile with research-based settings
    initial_lr = 1e-4  # Good for transfer learning
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"Learning rate: {initial_lr}")
    print("Using label smoothing: 0.1")
    
    # Create data generators
    batch_size = 32
    train_gen, val_gen = create_augmented_data(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Calculate steps
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_resnet50_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
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
            mode='max',
            verbose=1
        ),
        
        # Progressive unfreezing
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progressive_unfreeze(base_model, epoch)
        )
    ]
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("Phase 1: Partial fine-tuning (first 50 layers frozen)")
    print("Phase 2: More unfreezing at epoch 3")
    print("Phase 3: Most unfrozen at epoch 7")
    print("Phase 4: Full fine-tuning at epoch 10")
    print("="*70 + "\n")
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/final_resnet50_model.h5')
    
    # Save history
    import json
    with open('models/training_history_resnet50.json', 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        json.dump(history_dict, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    best_acc = max(history.history['val_accuracy'])
    print(f"Best validation accuracy: {best_acc:.2%}")
    print("Model saved to: models/best_resnet50_model.h5")
    
    return model, history


if __name__ == "__main__":
    import time
    
    print("\n" + "="*70)
    print("ULTIMATE FIX FOR PLANT DISEASE DETECTION")
    print("="*70)
    
    print("\nWHY THIS WILL WORK:")
    print("1. ResNet50 is more stable than EfficientNet")
    print("2. Proper preprocessing for [0,1] data")
    print("3. Progressive unfreezing (research-proven)")
    print("4. Label smoothing for better generalization")
    print("5. Strong augmentation for plant images")
    
    print("\nEXPECTED RESULTS:")
    print("- Should see 30-40% accuracy in first few epochs")
    print("- Should reach 60-80% by epoch 10-15")
    print("- Final accuracy: 70-85%")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    try:
        model, history = train_ultimate()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()