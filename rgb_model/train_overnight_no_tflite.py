#!/usr/bin/env python3
"""
OVERNIGHT TRAINING SCRIPT - NO TFLITE CONVERSION
This script trains the model but SKIPS TFLite conversion to avoid errors
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from datetime import datetime
import argparse

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU configured: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

print("=" * 70)
print("OVERNIGHT TRAINING - NO TFLITE ISSUES")
print(f"Start time: {datetime.now()}")
print("=" * 70)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
args = parser.parse_args()

# Configuration
CONFIG = {
    'data_dir': 'datasets/plantvillage_processed',
    'input_shape': (224, 224, 3),
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'initial_lr': args.learning_rate,
    'model_name': f'overnight_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
}

print(f"\nConfiguration:")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch Size: {CONFIG['batch_size']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")

# 1. Load data
print("\n1. Loading data...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(CONFIG['data_dir'], 'train'),
    target_size=CONFIG['input_shape'][:2],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(CONFIG['data_dir'], 'val'),
    target_size=CONFIG['input_shape'][:2],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False
)

test_generator = val_datagen.flow_from_directory(
    os.path.join(CONFIG['data_dir'], 'test'),
    target_size=CONFIG['input_shape'][:2],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
print(f"   Classes: {list(train_generator.class_indices.keys())}")
print(f"   Train samples: {train_generator.samples}")
print(f"   Val samples: {val_generator.samples}")
print(f"   Test samples: {test_generator.samples}")

# 2. Calculate class weights
print("\n2. Calculating class weights...")
class_counts = {}
for class_name in train_generator.class_indices.keys():
    class_dir = os.path.join(CONFIG['data_dir'], 'train', class_name)
    if os.path.exists(class_dir):
        count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = count

total_samples = sum(class_counts.values())
n_classes = len([c for c in class_counts.values() if c > 0])

class_weights = {}
for idx, (class_name, count) in enumerate(class_counts.items()):
    if count > 0:
        weight = total_samples / (n_classes * count)
        class_weights[idx] = weight
        print(f"   {class_name}: {count} samples, weight={weight:.3f}")
    else:
        class_weights[idx] = 0.0
        print(f"   {class_name}: 0 samples (EMPTY)")

# 3. Build model
print("\n3. Building model...")
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, 3, padding='same', input_shape=input_shape),
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
        layers.Dropout(0.4),
        
        # Block 4
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = create_model(CONFIG['input_shape'], num_classes)
print(f"   Total parameters: {model.count_params():,}")

# 4. Compile model
print("\n4. Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# 5. Setup callbacks
print("\n5. Setting up callbacks...")
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

callbacks_list = [
    callbacks.ModelCheckpoint(
        f'models/{CONFIG["model_name"]}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.TensorBoard(
        log_dir=f'logs/{CONFIG["model_name"]}',
        histogram_freq=1
    ),
    # Add CSV logger for tracking
    callbacks.CSVLogger(
        f'models/{CONFIG["model_name"]}_training.csv'
    )
]

# 6. Train model
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

try:
    history = model.fit(
        train_generator,
        epochs=CONFIG['epochs'],
        validation_data=val_generator,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # 7. Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    test_results = model.evaluate(test_generator, verbose=1)
    test_loss, test_acc, test_prec, test_rec, test_auc = test_results
    
    print(f"\nTest Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall: {test_rec:.4f}")
    print(f"   AUC: {test_auc:.4f}")
    
    # 8. Check predictions distribution
    print("\n8. Prediction Analysis:")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    
    unique, counts = np.unique(pred_classes, return_counts=True)
    print("   Predicted class distribution:")
    for class_idx, count in zip(unique, counts):
        if class_idx < len(train_generator.class_indices):
            class_name = list(train_generator.class_indices.keys())[class_idx]
            print(f"      {class_name}: {count} predictions")
    
    # 9. Save model and results (NO TFLITE!)
    print("\n9. Saving results...")
    
    # Save final model
    final_model_path = f'models/{CONFIG["model_name"]}_final.h5'
    model.save(final_model_path)
    print(f"   ✓ Model saved: {final_model_path}")
    
    # Save training history
    history_path = f'models/{CONFIG["model_name"]}_history.json'
    with open(history_path, 'w') as f:
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
        json.dump(history_dict, f, indent=2)
    print(f"   ✓ History saved: {history_path}")
    
    # Save training summary
    summary_path = f'models/{CONFIG["model_name"]}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Epochs: {CONFIG['epochs']}\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n")
        f.write(f"Learning Rate: {CONFIG['initial_lr']}\n")
        f.write(f"Train Samples: {train_generator.samples}\n")
        f.write(f"Val Samples: {val_generator.samples}\n")
        f.write(f"Test Samples: {test_generator.samples}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall: {test_rec:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"\nBest Val Accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Best Val Loss: {min(history.history['val_loss']):.4f}\n")
    print(f"   ✓ Summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - NO TFLITE ERRORS!")
    print(f"End time: {datetime.now()}")
    print("=" * 70)
    print("\n✓ Model saved successfully without TFLite conversion")
    print("✓ You can convert to TFLite later if needed")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("Saving current model state...")
    model.save(f'models/{CONFIG["model_name"]}_interrupted.h5')
    print("Model saved as interrupted state")
    
except Exception as e:
    print(f"\n\n❌ Error during training: {e}")
    print("Attempting to save current model state...")
    try:
        model.save(f'models/{CONFIG["model_name"]}_error.h5')
        print("Model saved despite error")
    except:
        print("Could not save model")