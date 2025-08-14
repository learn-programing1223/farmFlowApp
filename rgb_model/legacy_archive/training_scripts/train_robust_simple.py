#!/usr/bin/env python3
"""
Simplified robust training with heavy augmentation
Makes model work better on real-world images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_augmentation_layer():
    """Create Keras augmentation layers for real-world robustness"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.3),
        # Simulate different photo conditions
        layers.Lambda(lambda x: tf.clip_by_value(
            x + tf.random.normal(tf.shape(x), 0, 0.05), 0, 1
        )),  # Add noise
    ])

def build_robust_cnn(input_shape=(224, 224, 3), num_classes=7):
    """Build a CNN that generalizes well"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Augmentation (only during training)
    augmented = create_augmentation_layer()(inputs, training=True)
    
    # Normalization
    x = layers.Lambda(lambda x: x * 2.0 - 1.0)(augmented)
    
    # CNN blocks with aggressive regularization
    # Block 1
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)  # Higher dropout
    
    # Block 2
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.5)(x)
    
    # Global features
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with strong regularization
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.6)(x)
    
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def mixup_data(X, y, alpha=0.2):
    """Mixup augmentation for better generalization"""
    batch_size = len(X)
    
    # Random shuffle indices
    indices = np.random.permutation(batch_size)
    
    # Mix ratio
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.expand_dims(lam, axis=[1, 2, 3])
    
    # Mix images and labels
    X_mixed = lam * X + (1 - lam) * X[indices]
    y_mixed = lam[:, :, 0, 0] * y + (1 - lam[:, :, 0, 0]) * y[indices]
    
    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)

def create_robust_generator(X, y, batch_size=32, is_training=True):
    """Generator with multiple augmentation strategies"""
    indices = np.arange(len(X))
    
    while True:
        if is_training:
            np.random.shuffle(indices)
        
        for start in range(0, len(X), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_X = X[batch_indices].copy()
            batch_y = y[batch_indices].copy()
            
            if is_training:
                # Apply mixup with probability
                if np.random.rand() > 0.5:
                    batch_X, batch_y = mixup_data(batch_X, batch_y)
                
                # Random erasing (cutout)
                for i in range(len(batch_X)):
                    if np.random.rand() > 0.5:
                        h, w = 224, 224
                        # Random rectangle
                        x1 = np.random.randint(0, w - 20)
                        y1 = np.random.randint(0, h - 20)
                        x2 = x1 + np.random.randint(10, 40)
                        y2 = y1 + np.random.randint(10, 40)
                        
                        # Fill with random color or mean
                        if np.random.rand() > 0.5:
                            batch_X[i, y1:y2, x1:x2, :] = np.random.rand()
                        else:
                            batch_X[i, y1:y2, x1:x2, :] = np.mean(batch_X[i])
            
            yield batch_X, batch_y

def train_robust_model():
    """Train model with focus on real-world performance"""
    
    print("\n" + "="*70)
    print("ROBUST TRAINING FOR REAL-WORLD IMAGES")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_dir = Path('./data/splits')
    
    X_train = np.load(data_dir / 'X_train.npy').astype(np.float32)
    y_train = np.load(data_dir / 'y_train.npy').astype(np.float32)
    X_val = np.load(data_dir / 'X_val.npy').astype(np.float32)
    y_val = np.load(data_dir / 'y_val.npy').astype(np.float32)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # Build model
    print("\nBuilding robust model...")
    model = build_robust_cnn()
    
    # Use different optimizers for better generalization
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"Total parameters: {model.count_params():,}")
    
    # Create generators
    batch_size = 32
    train_gen = create_robust_generator(X_train, y_train, batch_size, is_training=True)
    val_gen = create_robust_generator(X_val, y_val, batch_size, is_training=False)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/robust_simple_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
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
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Add cosine annealing for better convergence
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.5 * (1 + np.cos(np.pi * epoch / 30)))
        )
    ]
    
    print("\n" + "="*50)
    print("TRAINING WITH HEAVY AUGMENTATION")
    print("="*50)
    print("Augmentation strategies:")
    print("- Random flips, rotations, zoom")
    print("- Brightness and contrast changes")
    print("- Gaussian noise injection")
    print("- Mixup augmentation")
    print("- Random erasing (cutout)")
    print("- Strong regularization (dropout, L2)")
    
    # Train
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=30,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save('models/robust_simple_final.h5')
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/plant_disease_robust.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.1f} MB")
    
    # Save history
    with open('models/robust_training_history.json', 'w') as f:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Model saved to: models/robust_simple_final.h5")
    print("\nThis model uses:")
    print("- Heavy data augmentation")
    print("- Mixup and cutout")
    print("- Strong regularization")
    print("- Should work better on real photos!")
    
    return model

if __name__ == "__main__":
    model = train_robust_model()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("1. Test with real images")
    print("2. Use test-time augmentation for predictions")
    print("3. Consider ensemble with original model")
    print("4. Collect more diverse training data if needed")