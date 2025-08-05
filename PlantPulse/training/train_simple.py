"""
Simplified training script for PlantPulse model
Works with TensorFlow 2.20+
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from typing import Tuple, Dict

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10  # Reduced for testing
LEARNING_RATE = 0.001

print(f"TensorFlow version: {tf.__version__}")

def create_simple_model() -> keras.Model:
    """Create a simplified CNN model for plant health analysis"""
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='thermal_input')
    
    # Simple CNN backbone
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    features = layers.Dense(256, activation='relu')(x)
    features = layers.Dropout(0.5)(features)
    
    # Multi-task outputs
    water_stress_output = layers.Dense(1, activation='sigmoid', name='water_stress')(features)
    disease_output = layers.Dense(4, activation='softmax', name='disease')(features)
    nutrients_output = layers.Dense(3, activation='sigmoid', name='nutrients')(features)
    
    # Simple segmentation output
    seg = layers.Dense(56 * 56)(features)
    seg = layers.Reshape((56, 56, 1))(seg)
    seg = layers.UpSampling2D(4)(seg)  # 224x224
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(seg)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[water_stress_output, disease_output, nutrients_output, segmentation_output]
    )
    
    return model

def generate_synthetic_data(num_samples: int) -> Tuple[np.ndarray, Dict]:
    """Generate simple synthetic thermal data"""
    print(f"Generating {num_samples} synthetic samples...")
    
    images = []
    labels = {
        'water_stress': [],
        'disease': [],
        'nutrients': [],
        'segmentation': []
    }
    
    for i in range(num_samples):
        # Create random thermal image
        img = np.random.normal(25, 5, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        
        # Add some structure
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < IMG_SIZE // 3:
                    img[y, x] -= np.random.uniform(3, 6)  # Plants are cooler
        
        images.append(img)
        
        # Random labels
        labels['water_stress'].append(np.random.random())
        
        disease_vec = [0, 0, 0, 0]
        disease_vec[np.random.randint(4)] = 1
        labels['disease'].append(disease_vec)
        
        labels['nutrients'].append(np.random.random(3))
        
        # Simple segmentation mask
        seg_mask = np.zeros((IMG_SIZE, IMG_SIZE))
        cv2.circle(seg_mask, (center_x, center_y), IMG_SIZE // 3, 1, -1)
        labels['segmentation'].append(seg_mask)
    
    # Convert to arrays
    images = np.array(images)
    for key in labels:
        labels[key] = np.array(labels[key])
    
    # Normalize images
    images = (images - 15) / 25  # Normalize to ~[0, 1]
    images = np.expand_dims(images, -1)  # Add channel dimension
    
    return images, labels

def main():
    print("PlantPulse Simple Training")
    print("=" * 40)
    
    # Generate data
    train_images, train_labels = generate_synthetic_data(1000)
    val_images, val_labels = generate_synthetic_data(200)
    
    print(f"Data shape: {train_images.shape}")
    print(f"Data range: [{train_images.min():.2f}, {train_images.max():.2f}]")
    
    # Build model
    print("\nBuilding model...")
    model = create_simple_model()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            'water_stress': 'mse',
            'disease': 'categorical_crossentropy',
            'nutrients': 'mse',
            'segmentation': 'binary_crossentropy'
        },
        metrics={
            'water_stress': ['mae'],
            'disease': ['accuracy'],
            'nutrients': ['mae'],
            'segmentation': ['accuracy']
        }
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
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
        verbose=1
    )
    
    # Save model
    print("\nSaving model...")
    model.save('plant_health_simple.h5')
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('plant_health_v1.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Training complete!")
    print(f"   - Keras model: plant_health_simple.h5")
    print(f"   - TFLite model: plant_health_v1.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()