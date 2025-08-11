#!/usr/bin/env python3
"""
Test only the TFLite model with different normalization approaches
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import json

def create_test_data():
    """Create some test data since no real images are available"""
    print("Creating synthetic test data...")
    
    # Create realistic-looking plant leaf images
    test_images = []
    
    # Generate 5 synthetic images
    for i in range(5):
        # Start with random noise
        img = np.random.random((224, 224, 3))
        
        # Add some structure to make it leaf-like
        center_x, center_y = 112, 112
        for y in range(224):
            for x in range(224):
                # Distance from center
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Create leaf-like shape
                if dist < 80:
                    # Green leafy area
                    img[y, x, 0] = 0.2 + 0.3 * np.random.random()  # Low red
                    img[y, x, 1] = 0.5 + 0.4 * np.random.random()  # Medium-high green
                    img[y, x, 2] = 0.1 + 0.3 * np.random.random()  # Low blue
                    
                    # Add some disease-like spots for variety
                    if i < 2 and np.random.random() < 0.1:  # First 2 images have "blight"
                        img[y, x, :] = [0.4, 0.3, 0.2]  # Brownish spots
                    elif i == 2 and np.random.random() < 0.05:  # Third image has "powdery mildew"  
                        img[y, x, :] = [0.9, 0.9, 0.9]  # White spots
        
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        test_images.append(img_uint8)
        
        # Save for visualization
        cv2.imwrite(f'synthetic_test_{i}.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    print(f"Created {len(test_images)} synthetic test images")
    return test_images

def preprocess_image_correct(img_array, target_size=(224, 224)):
    """Correct preprocessing with [-1,1] normalization"""
    # Ensure it's RGB and right size
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)
    
    # Convert to float32 and normalize to [0, 1] first
    img = img_array.astype(np.float32) / 255.0
    
    # CRITICAL: Convert to [-1, 1] as done in training
    img = img * 2.0 - 1.0
    
    return img

def preprocess_image_wrong(img_array, target_size=(224, 224)):
    """Wrong preprocessing with [0,1] normalization only"""
    # Ensure it's RGB and right size
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size)
    
    # Convert to float32 and normalize to [0, 1] ONLY (wrong!)
    img = img_array.astype(np.float32) / 255.0
    
    return img

def test_tflite_preprocessing():
    """Test TFLite model with different preprocessing"""
    print("\n" + "="*70)
    print("TESTING TFLITE MODEL WITH DIFFERENT PREPROCESSING")
    print("="*70)
    
    # Load TFLite model
    tflite_path = 'models/plant_disease_cyclegan_robust.tflite'
    
    if not Path(tflite_path).exists():
        print(f"TFLite model not found: {tflite_path}")
        return
    
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    
    class_names = [
        'Blight',
        'Healthy', 
        'Leaf_Spot',
        'Mosaic_Virus',
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    # Create test data
    test_images = create_test_data()
    
    print("\n" + "-"*70)
    print("PREPROCESSING COMPARISON")
    print("-"*70)
    
    results = []
    
    for i, img_array in enumerate(test_images):
        print(f"\nTesting synthetic image {i} (Expected: {'Blight' if i < 2 else 'Powdery Mildew' if i == 2 else 'Healthy'})")
        
        # Test with CORRECT preprocessing ([-1,1])
        img_correct = preprocess_image_correct(img_array)
        img_correct_batch = np.expand_dims(img_correct, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img_correct_batch)
        interpreter.invoke()
        predictions_correct = interpreter.get_tensor(output_details[0]['index'])
        
        top_pred_correct = np.argmax(predictions_correct[0])
        confidence_correct = predictions_correct[0][top_pred_correct]
        
        # Test with WRONG preprocessing ([0,1])
        img_wrong = preprocess_image_wrong(img_array)
        img_wrong_batch = np.expand_dims(img_wrong, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], img_wrong_batch)
        interpreter.invoke()
        predictions_wrong = interpreter.get_tensor(output_details[0]['index'])
        
        top_pred_wrong = np.argmax(predictions_wrong[0])
        confidence_wrong = predictions_wrong[0][top_pred_wrong]
        
        print(f"  CORRECT [-1,1] preprocessing:")
        print(f"    Prediction: {class_names[top_pred_correct]}")
        print(f"    Confidence: {confidence_correct:.1%}")
        
        print(f"  WRONG [0,1] preprocessing:")
        print(f"    Prediction: {class_names[top_pred_wrong]}")
        print(f"    Confidence: {confidence_wrong:.1%}")
        
        # Show full distribution for correct preprocessing
        print(f"  Full distribution (correct preprocessing):")
        for j, class_name in enumerate(class_names):
            print(f"    {class_name}: {predictions_correct[0][j]:.1%}")
        
        # Check if predictions differ
        if top_pred_correct != top_pred_wrong:
            print("  *** DIFFERENT PREDICTIONS! ***")
        
        results.append({
            'image': f'synthetic_{i}',
            'correct_pred': class_names[top_pred_correct],
            'correct_conf': float(confidence_correct),
            'wrong_pred': class_names[top_pred_wrong],
            'wrong_conf': float(confidence_wrong),
            'different': top_pred_correct != top_pred_wrong
        })
        
        print("-" * 50)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    different_count = sum(1 for r in results if r['different'])
    print(f"Images with different predictions: {different_count}/{len(results)}")
    
    if different_count > 0:
        print("\n*** PREPROCESSING MISMATCH CONFIRMED ***")
        print("The model gives different results with [-1,1] vs [0,1] normalization")
        print("This proves the web app needs to use [-1,1] normalization!")
    else:
        print("\nNo preprocessing differences found on synthetic data")
        print("May need to test with more diverse/realistic images")
    
    # Save results
    with open('preprocessing_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    print("TFLite Model Preprocessing Test")
    print("Testing if [-1,1] vs [0,1] normalization makes a difference")
    
    results = test_tflite_preprocessing()
    
    print(f"\nResults saved to preprocessing_test_results.json")
    print("Synthetic test images saved as synthetic_test_*.jpg")

if __name__ == "__main__":
    main()