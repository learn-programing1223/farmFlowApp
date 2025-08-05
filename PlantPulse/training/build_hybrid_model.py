#!/usr/bin/env python3
"""
Build Hybrid RGB + Thermal Model for PlantPulse
Supports both RGB-only and RGB+Thermal inputs
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Model configuration
IMG_SIZE = 224
RGB_CHANNELS = 3
THERMAL_CHANNELS = 1

def build_rgb_branch(input_shape=(IMG_SIZE, IMG_SIZE, RGB_CHANNELS)):
    """Build RGB processing branch using MobileNetV3"""
    
    # Load pre-trained MobileNetV3 for RGB
    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Fine-tune last layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    rgb_input = keras.Input(shape=input_shape, name='rgb_input')
    x = base_model(rgb_input, training=True)
    
    # Additional processing
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    rgb_features = layers.Dense(128, activation='relu', name='rgb_features')(x)
    
    return rgb_input, rgb_features

def build_thermal_branch(input_shape=(IMG_SIZE, IMG_SIZE, THERMAL_CHANNELS)):
    """Build thermal processing branch"""
    
    thermal_input = keras.Input(shape=input_shape, name='thermal_input')
    
    # Custom layers for thermal processing
    x = layers.Conv2D(32, 3, strides=2, padding='same')(thermal_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    thermal_features = layers.Dense(128, activation='relu', name='thermal_features')(x)
    
    return thermal_input, thermal_features

def build_hybrid_model():
    """Build multi-modal model that can work with RGB-only or RGB+Thermal"""
    
    # Build both branches
    rgb_input, rgb_features = build_rgb_branch()
    thermal_input, thermal_features = build_thermal_branch()
    
    # Create a thermal availability flag
    thermal_available = keras.Input(shape=(1,), name='thermal_available')
    
    # Fusion layer - combines features when both modalities available
    def fusion_layer(inputs):
        rgb_feat, thermal_feat, thermal_flag = inputs
        
        # If thermal available, combine features
        # If not, use only RGB features
        combined = tf.where(
            thermal_flag > 0.5,
            tf.concat([rgb_feat, thermal_feat], axis=-1),  # Both features
            tf.concat([rgb_feat, tf.zeros_like(thermal_feat)], axis=-1)  # RGB only
        )
        return combined
    
    fused_features = layers.Lambda(
        fusion_layer, 
        name='fusion'
    )([rgb_features, thermal_features, thermal_available])
    
    # Shared processing layers
    x = layers.Dense(256, activation='relu')(fused_features)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Multi-task outputs
    outputs = {}
    
    # 1. Water stress (regression) - more important with thermal
    water_branch = layers.Dense(64, activation='relu')(x)
    outputs['water_stress'] = layers.Dense(1, activation='sigmoid', name='water_stress')(water_branch)
    
    # 2. Disease classification (visible in RGB)
    disease_branch = layers.Dense(64, activation='relu')(x)
    outputs['disease'] = layers.Dense(4, activation='softmax', name='disease')(disease_branch)
    
    # 3. Nutrient status (multi-label)
    nutrient_branch = layers.Dense(64, activation='relu')(x)
    outputs['nutrients'] = layers.Dense(3, activation='sigmoid', name='nutrients')(nutrient_branch)
    
    # 4. General health score (works with both)
    health_branch = layers.Dense(32, activation='relu')(x)
    outputs['health_score'] = layers.Dense(1, activation='sigmoid', name='health_score')(health_branch)
    
    # Create model
    model = keras.Model(
        inputs={
            'rgb_input': rgb_input,
            'thermal_input': thermal_input,
            'thermal_available': thermal_available
        },
        outputs=outputs
    )
    
    return model

def prepare_data_for_hybrid(rgb_images, thermal_images=None):
    """Prepare data for hybrid model input"""
    
    batch_size = len(rgb_images)
    
    if thermal_images is not None:
        # Both RGB and thermal available
        thermal_available = np.ones((batch_size, 1), dtype=np.float32)
    else:
        # Only RGB available
        thermal_images = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        thermal_available = np.zeros((batch_size, 1), dtype=np.float32)
    
    return {
        'rgb_input': rgb_images,
        'thermal_input': thermal_images,
        'thermal_available': thermal_available
    }

def test_hybrid_model():
    """Test the hybrid model with different input scenarios"""
    
    print("Building Hybrid RGB+Thermal Model...")
    model = build_hybrid_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'water_stress': 'mse',
            'disease': 'categorical_crossentropy',
            'nutrients': 'binary_crossentropy',
            'health_score': 'binary_crossentropy'
        },
        loss_weights={
            'water_stress': 2.0,  # More important with thermal
            'disease': 1.0,
            'nutrients': 1.0,
            'health_score': 1.0
        },
        metrics={
            'water_stress': ['mae'],
            'disease': ['accuracy'],
            'nutrients': ['accuracy'],
            'health_score': ['mae']
        }
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Test with dummy data
    print("\n\nTesting with RGB-only input:")
    rgb_only = prepare_data_for_hybrid(
        rgb_images=np.random.rand(4, IMG_SIZE, IMG_SIZE, 3)
    )
    predictions = model.predict(rgb_only, verbose=0)
    print(f"Water stress: {predictions['water_stress'][0][0]:.3f}")
    print(f"Disease: {predictions['disease'][0].argmax()}")
    print(f"Health score: {predictions['health_score'][0][0]:.3f}")
    
    print("\n\nTesting with RGB+Thermal input:")
    rgb_thermal = prepare_data_for_hybrid(
        rgb_images=np.random.rand(4, IMG_SIZE, IMG_SIZE, 3),
        thermal_images=np.random.rand(4, IMG_SIZE, IMG_SIZE, 1)
    )
    predictions = model.predict(rgb_thermal, verbose=0)
    print(f"Water stress: {predictions['water_stress'][0][0]:.3f}")
    print(f"Disease: {predictions['disease'][0].argmax()}")
    print(f"Health score: {predictions['health_score'][0][0]:.3f}")
    
    return model

if __name__ == "__main__":
    model = test_hybrid_model()
    
    print("\n\n✅ Hybrid model created successfully!")
    print("\nKey features:")
    print("- Works with RGB-only input (free tier)")
    print("- Enhanced accuracy with thermal camera (premium)")
    print("- Shared feature learning between modalities")
    print("- Automatic adaptation based on available sensors")