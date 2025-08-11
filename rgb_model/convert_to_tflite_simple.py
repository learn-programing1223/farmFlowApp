#!/usr/bin/env python3
"""
Simple TensorFlow Lite converter for mobile deployment
Converts the model to TFLite format with optimization
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import os

def convert_to_tflite():
    """Convert the trained model to TensorFlow Lite format"""
    print("\n" + "="*70)
    print("CONVERTING TO TENSORFLOW LITE")
    print("="*70)
    
    # Load the model
    model_path = 'models/best_working_model.h5'
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Get model size
    original_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Original H5 model size: {original_size_mb:.2f} MB")
    
    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options
    print("Applying optimizations...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Representative dataset for better quantization
    def representative_dataset():
        """Generate representative data for quantization calibration"""
        for _ in range(100):
            # Generate random data in the same range as training data
            data = np.random.random((1, 224, 224, 3)).astype(np.float32)
            yield [data]
    
    # Enable quantization for smaller model
    converter.representative_dataset = representative_dataset
    
    # Convert
    print("Converting model...")
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

def test_tflite_model(tflite_path):
    """Test the converted TFLite model"""
    print("\n" + "="*70)
    print("TESTING TFLITE MODEL")
    print("="*70)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel details:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    # Test with random input
    print("\nTesting inference...")
    test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Inference successful!")
    print(f"Output shape: {output.shape}")
    print(f"Sum of probabilities: {np.sum(output[0]):.4f}")
    
    # Benchmark speed
    import time
    print("\nBenchmarking inference speed...")
    times = []
    
    for _ in range(10):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    avg_time_ms = np.mean(times) * 1000
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"FPS capability: {1000/avg_time_ms:.1f} fps")
    
    return True

def create_metadata():
    """Create metadata for the mobile app"""
    metadata = {
        "model_name": "PlantPulse Disease Detector",
        "version": "1.0.0",
        "format": "tflite",
        "accuracy": {
            "test": 0.9510,
            "validation": 0.9552
        },
        "classes": [
            "Blight",
            "Healthy",
            "Leaf_Spot",
            "Mosaic_Virus",
            "Nutrient_Deficiency",
            "Powdery_Mildew",
            "Rust"
        ],
        "input": {
            "shape": [1, 224, 224, 3],
            "normalization": "0_to_1",
            "preprocessing": "Model includes [-1,1] normalization layer"
        },
        "performance": {
            "parameters": 1440807,
            "size_mb": None  # Will be updated
        }
    }
    
    # Update size if model exists
    tflite_path = Path('models/plant_disease_model.tflite')
    if tflite_path.exists():
        metadata['performance']['size_mb'] = tflite_path.stat().st_size / (1024 * 1024)
    
    # Save metadata
    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    return metadata

def main():
    """Main conversion pipeline"""
    print("\n" + "="*70)
    print("TFLITE CONVERSION FOR MOBILE DEPLOYMENT")
    print("="*70)
    print("\nThis will convert your 95% accuracy model for:")
    print("- React Native apps")
    print("- Flutter apps")
    print("- Native iOS/Android apps")
    print("- Edge devices")
    
    try:
        # Convert to TFLite
        tflite_path = convert_to_tflite()
        
        # Test the converted model
        success = test_tflite_model(tflite_path)
        
        if success:
            # Create metadata
            metadata = create_metadata()
            
            print("\n" + "="*70)
            print("CONVERSION SUCCESSFUL!")
            print("="*70)
            print("\nYour model is ready for mobile deployment!")
            print(f"\nTFLite model: models/plant_disease_model.tflite")
            print(f"Metadata: models/model_metadata.json")
            print(f"\nModel size: {metadata['performance']['size_mb']:.2f} MB")
            print("\nNext steps:")
            print("1. Copy plant_disease_model.tflite to your mobile app")
            print("2. Use TensorFlow Lite SDK in your app")
            print("3. Deploy on-device for offline plant disease detection!")
            
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure TensorFlow is up to date")
        print("2. Check that the model file exists")
        print("3. Try running with Python 3.8-3.10")

if __name__ == "__main__":
    main()