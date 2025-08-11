#!/usr/bin/env python3
"""
Test the CycleGAN model with CORRECT [-1,1] normalization
This script will test both H5 and TFLite versions
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import json

def preprocess_image_correct(image_path, target_size=(224, 224)):
    """
    Preprocess image with CORRECT [-1,1] normalization 
    to match training preprocessing
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize to [0, 1] first
    img = img.astype(np.float32) / 255.0
    
    # CRITICAL: Convert to [-1, 1] as done in training
    img = img * 2.0 - 1.0
    
    return img

def preprocess_image_wrong(image_path, target_size=(224, 224)):
    """
    Old preprocessing with [0,1] normalization (WRONG for CycleGAN model)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize to [0, 1] ONLY (wrong!)
    img = img.astype(np.float32) / 255.0
    
    return img

def test_h5_model():
    """Test the H5 model with correct preprocessing"""
    print("\n" + "="*70)
    print("TESTING H5 MODEL WITH CORRECT PREPROCESSING")
    print("="*70)
    
    # Load the CycleGAN model
    print("\nLoading CycleGAN H5 model...")
    model_path = 'models/best_cyclegan_model.h5'
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Class names
    class_names = [
        'Blight',
        'Healthy', 
        'Leaf_Spot',
        'Mosaic_Virus',
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    # Create test_images directory if it doesn't exist
    test_dir = Path('test_images')
    test_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_dir.glob(ext))
    
    if not image_files:
        print(f"\nNo images found in {test_dir}/")
        print("\nTo test on real images:")
        print("1. Download plant disease images from Google Images")
        print("2. Save them in 'rgb_model/test_images/' folder")
        print("3. Run this script again")
        return
    
    print(f"\nFound {len(image_files)} test images")
    print("-" * 70)
    
    results = []
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Test with CORRECT preprocessing ([-1,1])
        img_correct = preprocess_image_correct(img_path)
        if img_correct is None:
            continue
        
        # Test with WRONG preprocessing ([0,1])
        img_wrong = preprocess_image_wrong(img_path)
        if img_wrong is None:
            continue
        
        # Add batch dimensions
        img_correct_batch = np.expand_dims(img_correct, axis=0)
        img_wrong_batch = np.expand_dims(img_wrong, axis=0)
        
        # Predict with correct preprocessing
        predictions_correct = model.predict(img_correct_batch, verbose=0)
        top_pred_correct = np.argmax(predictions_correct[0])
        confidence_correct = predictions_correct[0][top_pred_correct]
        
        # Predict with wrong preprocessing 
        predictions_wrong = model.predict(img_wrong_batch, verbose=0)
        top_pred_wrong = np.argmax(predictions_wrong[0])
        confidence_wrong = predictions_wrong[0][top_pred_wrong]
        
        print(f"  CORRECT [-1,1] preprocessing:")
        print(f"    Prediction: {class_names[top_pred_correct]}")
        print(f"    Confidence: {confidence_correct:.1%}")
        
        print(f"  WRONG [0,1] preprocessing:")
        print(f"    Prediction: {class_names[top_pred_wrong]}")  
        print(f"    Confidence: {confidence_wrong:.1%}")
        
        # Show top 3 for correct preprocessing
        top3_indices = np.argsort(predictions_correct[0])[-3:][::-1]
        print(f"  Top 3 predictions (correct preprocessing):")
        for i, idx in enumerate(top3_indices, 1):
            print(f"    {i}. {class_names[idx]}: {predictions_correct[0][idx]:.1%}")
        
        results.append({
            'image': img_path.name,
            'correct_pred': class_names[top_pred_correct],
            'correct_conf': float(confidence_correct),
            'wrong_pred': class_names[top_pred_wrong], 
            'wrong_conf': float(confidence_wrong)
        })
        
        print("-" * 50)
    
    # Save results
    results_file = 'preprocessing_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results

def test_tflite_model():
    """Test the TFLite model"""
    print("\n" + "="*70)
    print("TESTING TFLITE MODEL")
    print("="*70)
    
    # Load TFLite model
    tflite_path = 'models/plant_disease_cyclegan_robust.tflite'
    
    if not Path(tflite_path).exists():
        print(f"TFLite model not found: {tflite_path}")
        return
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    class_names = [
        'Blight',
        'Healthy', 
        'Leaf_Spot',
        'Mosaic_Virus',
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    test_dir = Path('test_images')
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_dir.glob(ext))
    
    if not image_files:
        print("No test images found")
        return
    
    print(f"\nFound {len(image_files)} test images")
    print("-" * 50)
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Preprocess with correct [-1,1] normalization
        img = preprocess_image_correct(img_path)
        if img is None:
            continue
        
        # Add batch dimension and convert to float32
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Get top prediction
        top_pred = np.argmax(predictions[0])
        confidence = predictions[0][top_pred]
        
        print(f"  TFLite Prediction: {class_names[top_pred]}")
        print(f"  TFLite Confidence: {confidence:.1%}")
        
        # Show top 3
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        print(f"  Top 3 predictions:")
        for i, idx in enumerate(top3_indices, 1):
            print(f"    {i}. {class_names[idx]}: {predictions[0][idx]:.1%}")
        
        print("-" * 30)

def main():
    """Main function"""
    print("Plant Disease Model Debugging Tool")
    print("Testing preprocessing normalization issues")
    
    # Test H5 model with different preprocessing
    h5_results = test_h5_model()
    
    # Test TFLite model
    test_tflite_model()
    
    if h5_results:
        print("\n" + "="*70)
        print("SUMMARY OF PREPROCESSING COMPARISON")
        print("="*70)
        
        for result in h5_results:
            print(f"\nImage: {result['image']}")
            print(f"  Correct [-1,1]: {result['correct_pred']} ({result['correct_conf']:.1%})")
            print(f"  Wrong [0,1]:   {result['wrong_pred']} ({result['wrong_conf']:.1%})")
            
            if result['correct_pred'] != result['wrong_pred']:
                print("  *** DIFFERENT PREDICTIONS! ***")

if __name__ == "__main__":
    main()