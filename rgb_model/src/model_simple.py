"""
Simplified model without EfficientNet to avoid compatibility issues
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class SimplePlantDiseaseModel:
    """
    Simple CNN model for plant disease detection
    Works with any TensorFlow version
    """
    
    def __init__(self, num_classes=7, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a simple but effective CNN"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Conv Block 1
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 2
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 3
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 4
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='simple_plant_disease_model')
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model(self):
        return self.model


def test_simple_model():
    """Test the simple model"""
    print("Testing Simple Plant Disease Model...")
    
    # Create model
    model = SimplePlantDiseaseModel(num_classes=7)
    model.compile_model()
    
    # Test forward pass
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = model.get_model()(dummy_input)
    
    print(f"✓ Model created successfully")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Total parameters: {model.get_model().count_params():,}")
    
    return model


if __name__ == "__main__":
    test_simple_model()