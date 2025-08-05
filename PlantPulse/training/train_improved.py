"""
Improved training script with regularization to prevent overfitting
Addresses the issues found in the previous training runs
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import cv2
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased for better gradient estimates
EPOCHS = 50
LEARNING_RATE = 0.0001  # Reduced learning rate
L2_WEIGHT = 0.01  # L2 regularization strength
DROPOUT_RATE = 0.6  # Increased dropout

print(f"TensorFlow version: {tf.__version__}")
print("Improvements over previous version:")
print("- Stronger regularization (L2 + higher dropout)")
print("- Early stopping with patience=5")
print("- Reduced model complexity")
print("- Better data augmentation")
print("- Learning rate scheduling")

def create_regularized_model() -> keras.Model:
    """Create a simpler CNN with strong regularization"""
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='thermal_input')
    
    # Data augmentation layers (only applied during training)
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    
    # Simpler CNN backbone with L2 regularization
    x = layers.Conv2D(16, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.5)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Smaller dense layer with heavy dropout
    features = layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    features = layers.Dropout(DROPOUT_RATE)(features)
    features = layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    features = layers.Dropout(DROPOUT_RATE)(features)
    
    # Multi-task outputs with regularization
    water_stress_output = layers.Dense(1, activation='sigmoid', 
                                     name='water_stress',
                                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    disease_output = layers.Dense(4, activation='softmax', 
                                 name='disease',
                                 kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    nutrients_output = layers.Dense(3, activation='sigmoid', 
                                   name='nutrients',
                                   kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    # Simplified segmentation (optional, can be removed to reduce complexity)
    seg = layers.Dense(14 * 14)(features)
    seg = layers.Reshape((14, 14, 1))(seg)
    seg = layers.UpSampling2D(16)(seg)  # 224x224
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', 
                                       name='segmentation',
                                       kernel_regularizer=regularizers.l2(L2_WEIGHT))(seg)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[water_stress_output, disease_output, nutrients_output, segmentation_output]
    )
    
    return model

def generate_better_synthetic_data(num_samples: int, augment: bool = True) -> Tuple[np.ndarray, Dict]:
    """Generate synthetic data with more variety and realistic patterns"""
    print(f"Generating {num_samples} synthetic samples with better diversity...")
    
    images = []
    labels = {
        'water_stress': [],
        'disease': [],
        'nutrients': [],
        'segmentation': []
    }
    
    for i in range(num_samples):
        # Base temperature with more variation
        ambient_temp = np.random.uniform(20, 35)
        
        # Create more realistic thermal patterns
        img = np.ones((IMG_SIZE, IMG_SIZE)) * ambient_temp
        
        # Add gaussian noise for realism
        noise = np.random.normal(0, 0.5, (IMG_SIZE, IMG_SIZE))
        img += noise
        
        # Random plant characteristics
        plant_type = np.random.choice(['healthy', 'water_stressed', 'diseased', 'nutrient_deficient'])
        
        # Create plant regions with more realistic shapes
        num_leaves = np.random.randint(3, 8)
        for _ in range(num_leaves):
            # Random elliptical leaf shapes
            center_x = np.random.randint(40, IMG_SIZE - 40)
            center_y = np.random.randint(40, IMG_SIZE - 40)
            angle = np.random.uniform(0, 360)
            axes = (np.random.randint(20, 50), np.random.randint(10, 30))
            
            # Create mask for leaf
            leaf_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cv2.ellipse(leaf_mask, (center_x, center_y), axes, angle, 0, 360, 1, -1)
            
            # Apply temperature based on condition
            if plant_type == 'healthy':
                leaf_temp = ambient_temp - np.random.uniform(4, 8)
                labels['water_stress'].append(np.random.uniform(0, 0.2))
                labels['disease'].append([1.0, 0.0, 0.0, 0.0])
                labels['nutrients'].append([0.7, 0.7, 0.7])
            
            elif plant_type == 'water_stressed':
                stress_level = np.random.uniform(0.3, 0.9)
                leaf_temp = ambient_temp + (stress_level * 6)
                labels['water_stress'].append(stress_level)
                labels['disease'].append([1.0, 0.0, 0.0, 0.0])
                labels['nutrients'].append([0.5, 0.5, 0.5])
            
            elif plant_type == 'diseased':
                disease_class = np.random.randint(1, 4)
                leaf_temp = ambient_temp + np.random.uniform(-2, 4)
                labels['water_stress'].append(np.random.uniform(0, 0.3))
                disease_vec = [0.0, 0.0, 0.0, 0.0]
                disease_vec[disease_class] = 1.0
                labels['disease'].append(disease_vec)
                labels['nutrients'].append([0.5, 0.5, 0.5])
            
            else:  # nutrient_deficient
                leaf_temp = ambient_temp - np.random.uniform(1, 3)
                labels['water_stress'].append(np.random.uniform(0, 0.2))
                labels['disease'].append([1.0, 0.0, 0.0, 0.0])
                nutrient_vec = [0.7, 0.7, 0.7]
                nutrient_vec[np.random.randint(3)] = 0.2
                labels['nutrients'].append(nutrient_vec)
            
            # Apply leaf temperature
            img[leaf_mask == 1] = leaf_temp + np.random.normal(0, 0.3, np.sum(leaf_mask))
        
        # Create segmentation mask
        seg_mask = (img < ambient_temp - 2).astype(np.float32)
        labels['segmentation'].append(seg_mask)
        
        # Add more realistic variations
        if augment and np.random.random() > 0.5:
            # Add shadows or lighting variations
            gradient = np.linspace(0, 1, IMG_SIZE).reshape(1, -1)
            if np.random.random() > 0.5:
                gradient = gradient.T
            img += gradient * np.random.uniform(-2, 2)
        
        images.append(img)
    
    # Convert to arrays and normalize
    images = np.array(images)
    images = (images - 15) / 25  # Normalize to ~[0, 1]
    images = np.expand_dims(images, -1)
    
    # Fix label arrays to match number of samples
    final_labels = {
        'water_stress': [],
        'disease': [],
        'nutrients': [],
        'segmentation': []
    }
    
    for i in range(num_samples):
        idx = i % len(labels['water_stress'])
        final_labels['water_stress'].append(labels['water_stress'][idx])
        final_labels['disease'].append(labels['disease'][idx])
        final_labels['nutrients'].append(labels['nutrients'][idx])
        final_labels['segmentation'].append(labels['segmentation'][idx])
    
    for key in final_labels:
        final_labels[key] = np.array(final_labels[key])
    
    return images, final_labels

def plot_training_history(history, save_path='training_history_improved.png'):
    """Plot training history with focus on overfitting detection"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Disease classification accuracy
    ax = axes[0, 0]
    ax.plot(history['disease_accuracy'], label='Train', linewidth=2)
    ax.plot(history['val_disease_accuracy'], label='Validation', linewidth=2)
    ax.axhline(y=0.25, color='gray', linestyle='--', label='Random (25%)')
    ax.set_title('Disease Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Disease loss
    ax = axes[0, 1]
    ax.plot(history['disease_loss'], label='Train', linewidth=2)
    ax.plot(history['val_disease_loss'], label='Validation', linewidth=2)
    ax.set_title('Disease Classification Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total loss
    ax = axes[1, 0]
    ax.plot(history['loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overfitting metric
    ax = axes[1, 1]
    train_acc = history['disease_accuracy']
    val_acc = history['val_disease_accuracy']
    overfitting_gap = [t - v for t, v in zip(train_acc, val_acc)]
    
    colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' for gap in overfitting_gap]
    ax.bar(range(len(overfitting_gap)), overfitting_gap, color=colors)
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Moderate overfitting')
    ax.axhline(y=0.2, color='red', linestyle='--', label='Severe overfitting')
    ax.set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Improved Training with Regularization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training pipeline with improvements"""
    print("\nPlantPulse Improved Training Pipeline")
    print("=" * 40)
    
    # Generate data with better diversity
    print("\n1. Generating training data...")
    train_images, train_labels = generate_better_synthetic_data(2000, augment=True)
    val_images, val_labels = generate_better_synthetic_data(500, augment=False)
    
    print(f"   Training samples: {len(train_images)}")
    print(f"   Validation samples: {len(val_images)}")
    print(f"   Data shape: {train_images.shape}")
    print(f"   Data range: [{train_images.min():.2f}, {train_images.max():.2f}]")
    
    # Build model
    print("\n2. Building regularized model...")
    model = create_regularized_model()
    
    # Count parameters
    total_params = model.count_params()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size estimate: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Compile with balanced loss weights
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            'water_stress': 'mse',
            'disease': 'categorical_crossentropy',
            'nutrients': 'mse',
            'segmentation': 'binary_crossentropy'
        },
        loss_weights={
            'water_stress': 1.0,
            'disease': 2.0,  # Increased weight for main task
            'nutrients': 0.5,
            'segmentation': 0.2  # Reduced to prevent dominating
        },
        metrics={
            'water_stress': ['mae'],
            'disease': ['accuracy'],
            'nutrients': ['mae'],
            'segmentation': ['accuracy']
        }
    )
    
    # Callbacks with early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_disease_loss',
            patience=5,
            mode='min',  # Added mode for loss minimization
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_disease_loss',
            factor=0.5,
            patience=3,
            mode='min',  # Added mode for loss minimization
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'plant_health_improved_best.h5',
            monitor='val_disease_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train
    print(f"\n3. Training for up to {EPOCHS} epochs with early stopping...")
    history = model.fit(
        train_images,
        {
            'water_stress': train_labels['water_stress'],
            'disease': train_labels['disease'],
            'nutrients': train_labels['nutrients'],
            'segmentation': train_labels['segmentation']
        },
        validation_data=(
            val_images,
            {
                'water_stress': val_labels['water_stress'],
                'disease': val_labels['disease'],
                'nutrients': val_labels['nutrients'],
                'segmentation': val_labels['segmentation']
            }
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n4. Saving model...")
    model.save('plant_health_improved.h5')
    
    # Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history.history)
    
    # Save history
    with open('training_history_improved.json', 'w') as f:
        json.dump(history.history, f)
    
    # Evaluate final performance
    print("\n6. Final Evaluation:")
    final_epoch = len(history.history['loss'])
    final_train_acc = history.history['disease_accuracy'][-1]
    final_val_acc = history.history['val_disease_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"   Stopped at epoch: {final_epoch}")
    print(f"   Final training accuracy: {final_train_acc:.1%}")
    print(f"   Final validation accuracy: {final_val_acc:.1%}")
    print(f"   Overfitting gap: {overfitting_gap:.1%}")
    
    if final_val_acc > 0.4:
        print("   ✅ Model shows reasonable generalization!")
    elif final_val_acc > 0.3:
        print("   ⚠️  Model has moderate generalization")
    else:
        print("   ❌ Model still showing poor generalization")
    
    # Convert to TFLite
    print("\n7. Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('plant_health_improved.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Training complete!")
    print(f"   - Keras model: plant_health_improved.h5")
    print(f"   - Best checkpoint: plant_health_improved_best.h5")
    print(f"   - TFLite model: plant_health_improved.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    print(f"   - Training history: training_history_improved.json")
    print(f"   - History plot: training_history_improved.png")

if __name__ == "__main__":
    main()