#!/usr/bin/env python3
"""
Test model with real internet images - simplified version
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import urllib.request
from PIL import Image
import io

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import locale
    if locale.getpreferredencoding().upper() != 'UTF-8':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# Working image URLs (tested and accessible)
TEST_IMAGES = {
    'Blight': [
        'https://cdn.pixabay.com/photo/2021/07/19/10/42/tomato-6477833_640.jpg',  # tomato blight
        'https://cdn.pixabay.com/photo/2020/06/26/15/14/tomato-5342991_640.jpg',
    ],
    'Powdery_Mildew': [
        'https://cdn.pixabay.com/photo/2019/08/13/19/48/powdery-mildew-4404131_640.jpg',
        'https://cdn.pixabay.com/photo/2020/05/25/17/03/leaf-5219563_640.jpg',
    ],
    'Healthy': [
        'https://cdn.pixabay.com/photo/2016/08/30/13/13/green-1630852_640.jpg',
        'https://cdn.pixabay.com/photo/2018/06/04/23/42/green-3454414_640.jpg',
        'https://cdn.pixabay.com/photo/2016/07/24/20/48/tulip-1539279_640.jpg'
    ]
}

def download_image(url):
    """Download image from URL"""
    try:
        # Create request with headers
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Download
        with urllib.request.urlopen(req, timeout=10) as response:
            img_data = response.read()
        
        # Open and convert
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(img) / 255.0
        return img_array, True
        
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")
        return None, False


def test_model_performance():
    """Test the model with real images"""
    print("\n" + "="*70)
    print("TESTING MODEL WITH REAL-WORLD IMAGES")
    print("="*70)
    
    # Load model
    model_path = 'models/best_working_model.h5'
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading model...")
    model = tf.keras.models.load_model(model_path)
    
    # Class names
    class_names = [
        'Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus',
        'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust'
    ]
    
    # Test statistics
    total_tested = 0
    correct = 0
    class_results = {}
    
    # Test each category
    for true_class, urls in TEST_IMAGES.items():
        if true_class not in class_names:
            continue
            
        true_idx = class_names.index(true_class)
        class_correct = 0
        class_total = 0
        
        print(f"\nTesting {true_class}:")
        print("-" * 40)
        
        for i, url in enumerate(urls, 1):
            print(f"  Image {i}: ", end="")
            
            # Download
            img_array, success = download_image(url)
            if not success:
                continue
            
            # Predict
            pred = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
            pred_idx = np.argmax(pred)
            pred_class = class_names[pred_idx]
            confidence = pred[pred_idx]
            
            # Check result
            is_correct = (pred_idx == true_idx)
            total_tested += 1
            class_total += 1
            
            if is_correct:
                correct += 1
                class_correct += 1
                print(f"CORRECT - {pred_class} ({confidence:.1%})")
            else:
                print(f"WRONG - Predicted {pred_class} ({confidence:.1%})")
        
        # Store class accuracy
        if class_total > 0:
            class_accuracy = class_correct / class_total * 100
            class_results[true_class] = {
                'accuracy': class_accuracy,
                'correct': class_correct,
                'total': class_total
            }
            print(f"  Class Accuracy: {class_accuracy:.1f}% ({class_correct}/{class_total})")
    
    # Overall results
    overall_accuracy = (correct / total_tested * 100) if total_tested > 0 else 0
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Total Images Tested: {total_tested}")
    print(f"Correct Predictions: {correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    
    print("\nPer-Class Performance:")
    for class_name, results in class_results.items():
        print(f"  {class_name:20s}: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if overall_accuracy >= 85:
        print("SUCCESS: Model performs well on real-world images!")
    elif overall_accuracy >= 70:
        print("MODERATE: Model shows decent performance but needs improvement")
    else:
        print("POOR: Model struggles with real-world images")
        print("\nThe issue is likely due to:")
        print("1. Training only on PlantVillage dataset (controlled environment)")
        print("2. Lack of diverse backgrounds and lighting conditions")
        print("3. No style transfer or domain adaptation")
        
    print(f"\nGap between claimed (95%) and actual ({overall_accuracy:.1f}%): {95 - overall_accuracy:.1f}%")
    
    return overall_accuracy


if __name__ == "__main__":
    accuracy = test_model_performance()
    
    if accuracy < 85:
        print("\n" + "="*70)
        print("SOLUTION: IMPLEMENTING ROBUST TRAINING")
        print("="*70)
        print("The model needs to be retrained with:")
        print("1. CycleGAN-style augmentation to simulate real photos")
        print("2. Multiple datasets (PlantDoc, PlantNet)")
        print("3. Advanced augmentation pipeline")
        print("4. Test-time augmentation")
        print("\nRun: python train_robust_model.py")