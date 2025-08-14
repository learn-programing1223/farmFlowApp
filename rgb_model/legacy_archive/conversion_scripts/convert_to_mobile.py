#!/usr/bin/env python3
"""
Convert the trained model to mobile-friendly formats:
1. TensorFlow Lite for React Native
2. TensorFlow.js for web deployment
"""

import tensorflow as tf
import tensorflowjs as tfjs
from pathlib import Path
import json
import numpy as np

def convert_to_tflite():
    """
    Convert Keras model to TensorFlow Lite with quantization
    """
    print("\n" + "="*70)
    print("CONVERTING MODEL TO TENSORFLOW LITE")
    print("="*70)
    
    # Load the best model
    model_path = 'models/best_working_model.h5'
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Get original model size
    import os
    original_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Original model size: {original_size_mb:.2f} MB")
    
    # Convert to TFLite with quantization
    print("\nConverting to TensorFlow Lite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for full integer quantization
    def representative_dataset():
        # Load a subset of training data for calibration
        data_path = Path('data/splits/X_train.npy')
        if data_path.exists():
            data = np.load(data_path).astype(np.float32)
            # Use first 100 samples for calibration
            for i in range(min(100, len(data))):
                # Add batch dimension and yield
                yield [data[i:i+1]]
        else:
            # Fallback: generate random data
            for _ in range(100):
                yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Optional: Force full integer quantization (smaller but may lose accuracy)
    # Commented out to maintain accuracy
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = 'models/plant_disease_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nTFLite model saved to: {tflite_path}")
    print(f"TFLite model size: {tflite_size_mb:.2f} MB")
    print(f"Size reduction: {(1 - tflite_size_mb/original_size_mb)*100:.1f}%")
    
    return tflite_path

def convert_to_tfjs():
    """
    Convert Keras model to TensorFlow.js format
    """
    print("\n" + "="*70)
    print("CONVERTING MODEL TO TENSORFLOW.JS")
    print("="*70)
    
    # Load the model
    model_path = 'models/best_working_model.h5'
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    tfjs_dir = Path('models/tfjs_model')
    tfjs_dir.mkdir(exist_ok=True)
    
    # Convert to TensorFlow.js
    print(f"Converting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, str(tfjs_dir))
    
    print(f"TensorFlow.js model saved to: {tfjs_dir}")
    
    # Check the generated files
    model_json = tfjs_dir / 'model.json'
    if model_json.exists():
        with open(model_json, 'r') as f:
            model_config = json.load(f)
        
        # Count weight files
        weight_files = list(tfjs_dir.glob('*.bin'))
        total_size_mb = sum(f.stat().st_size for f in weight_files) / (1024 * 1024)
        
        print(f"Generated files:")
        print(f"  - model.json (architecture)")
        print(f"  - {len(weight_files)} weight shards")
        print(f"  - Total size: {total_size_mb:.2f} MB")
    
    return str(tfjs_dir)

def test_tflite_model():
    """
    Test the converted TFLite model to ensure it works
    """
    print("\n" + "="*70)
    print("TESTING TFLITE MODEL")
    print("="*70)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='models/plant_disease_model.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Input Shape: {input_details[0]['shape']}")
    print(f"Model Input Type: {input_details[0]['dtype']}")
    print(f"Model Output Shape: {output_details[0]['shape']}")
    
    # Test with random input
    test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTest inference successful!")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {np.argmax(output[0])}")
    print(f"Confidence: {np.max(output[0]):.2%}")
    
    # Test speed
    import time
    print("\nBenchmarking inference speed...")
    times = []
    for _ in range(10):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS capability: {1000/avg_time:.1f}")

def create_model_metadata():
    """
    Create metadata file for the model
    """
    metadata = {
        "model_name": "PlantPulse Disease Detector",
        "version": "1.0.0",
        "accuracy": 0.9510,
        "classes": [
            "Blight",
            "Healthy",
            "Leaf_Spot",
            "Mosaic_Virus",
            "Nutrient_Deficiency",
            "Powdery_Mildew",
            "Rust"
        ],
        "input_shape": [224, 224, 3],
        "preprocessing": {
            "normalization": "scale_to_[-1,1]",
            "description": "Multiply pixel values by 2.0 and subtract 1.0"
        },
        "training_info": {
            "dataset": "PlantVillage",
            "samples": 14000,
            "test_accuracy": 0.9510,
            "validation_accuracy": 0.9552
        }
    }
    
    # Save metadata
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nModel metadata saved to: models/model_metadata.json")
    return metadata

def main():
    """
    Main conversion pipeline
    """
    print("\n" + "="*70)
    print("MOBILE MODEL CONVERSION PIPELINE")
    print("="*70)
    print("\nThis will prepare your model for deployment in:")
    print("1. React Native apps (iOS/Android)")
    print("2. Web applications")
    print("3. Progressive Web Apps (PWAs)")
    
    # Check if tensorflowjs is installed
    try:
        import tensorflowjs
    except ImportError:
        print("\nInstalling tensorflowjs...")
        import subprocess
        subprocess.run(["pip", "install", "tensorflowjs"], check=True)
        import tensorflowjs
    
    # Convert to TFLite
    tflite_path = convert_to_tflite()
    
    # Convert to TensorFlow.js
    tfjs_path = convert_to_tfjs()
    
    # Test the TFLite model
    test_tflite_model()
    
    # Create metadata
    metadata = create_model_metadata()
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"1. TFLite model: models/plant_disease_model.tflite")
    print(f"2. TensorFlow.js model: models/tfjs_model/")
    print(f"3. Model metadata: models/model_metadata.json")
    
    print("\n✅ Your model is ready for mobile deployment!")
    print("\nNext steps:")
    print("1. Copy plant_disease_model.tflite to your React Native app")
    print("2. Use @tensorflow/tfjs-react-native to load the model")
    print("3. Build your camera interface")
    print("4. Ship to App Store! 🚀")

if __name__ == "__main__":
    main()