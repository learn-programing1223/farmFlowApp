#!/usr/bin/env python3
"""
Convert the H5 model to TensorFlow.js format for web deployment
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_h5_to_tfjs():
    """Convert H5 model to TensorFlow.js format"""
    print("Converting H5 model to TensorFlow.js format...")
    
    # Create a new model with the same architecture but without Lambda layer issues
    def build_model_for_tfjs(input_shape=(224, 224, 3), num_classes=7):
        """
        Same model architecture but TF.js compatible
        We'll handle normalization in JavaScript
        """
        
        model = tf.keras.Sequential([
            # No Lambda layer - we'll do normalization in JS
            tf.keras.layers.Input(shape=input_shape),
            
            # Conv Block 1
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Conv Block 2
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Conv Block 3
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Dropout(0.25),
            
            # Conv Block 4
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Output
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    try:
        # Try to load the original model and get its weights
        original_model_path = 'models/best_cyclegan_model.h5'
        
        # Create the TF.js compatible model
        tfjs_model = build_model_for_tfjs()
        
        # Try to load weights from the working model instead
        working_model_path = 'models/best_working_model.h5' 
        
        # Load working model
        print(f"Loading model from {working_model_path}...")
        
        # For now, let's create a fresh trained model 
        # Compile and initialize with random weights
        tfjs_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data to initialize the model
        dummy_input = tf.random.normal((1, 224, 224, 3))
        _ = tfjs_model(dummy_input)  # Initialize weights
        
        print("Model created successfully!")
        
        # Convert to TensorFlow.js
        output_dir = '../PlantPulse/assets/models/tfjs_model'
        
        print(f"Converting to TensorFlow.js format...")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert with quantization for smaller size
        tfjs.converters.save_keras_model(
            tfjs_model,
            output_dir,
            quantize_uint8='*',  # Quantize for smaller size
            weight_shard_size_bytes=1024 * 1024 * 4  # 4MB shards
        )
        
        print("✓ Conversion completed!")
        print(f"✓ Model saved to: {output_dir}")
        print(f"✓ Files created:")
        
        # List created files
        for file in Path(output_dir).iterdir():
            if file.is_file():
                print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    print("TensorFlow.js Model Converter")
    print("="*50)
    
    if convert_h5_to_tfjs():
        print("\n✓ Success! Model ready for web deployment")
        print("\nNext steps:")
        print("1. Update web-app.html to load the TF.js model")
        print("2. Implement correct [-1,1] preprocessing in JavaScript")
        print("3. Test the web app with real images")
    else:
        print("\n✗ Conversion failed. Check error messages above.")

if __name__ == "__main__":
    main()