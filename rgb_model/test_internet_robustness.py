#!/usr/bin/env python3
"""
Test model robustness with internet images
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Disease classes
CLASSES = ['Blight', 'Healthy', 'Leaf_Spot', 
           'Mosaic_Virus', 'Nutrient_Deficiency', 'Powdery_Mildew']

def download_and_preprocess(url):
    """Download and preprocess image from URL"""
    try:
        # Download
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def test_model_robustness():
    """Test model with various internet images"""
    
    print("=" * 70)
    print("ROBUSTNESS TEST WITH INTERNET IMAGES")
    print("=" * 70)
    
    # Check if model exists
    model_path = 'models/plantvillage_robust_best.h5'
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at {model_path}")
        print("Training must complete first!")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("[OK] Model loaded successfully")
    
    # Test images (replace with actual URLs)
    test_images = [
        {
            'url': 'https://plantix.net/en/library/assets/custom/crop-images/tomato.jpeg',
            'expected': 'Healthy',
            'description': 'Healthy tomato leaf'
        },
        # Add more test URLs here
    ]
    
    print("\n" + "-" * 70)
    print("Testing with Internet Images:")
    print("-" * 70)
    
    results = []
    for test in test_images:
        print(f"\nTesting: {test['description']}")
        print(f"URL: {test['url'][:50]}...")
        
        # Download and preprocess
        img_array = download_and_preprocess(test['url'])
        if img_array is None:
            continue
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        
        # Get results
        top_idx = np.argmax(predictions[0])
        top_class = CLASSES[top_idx]
        top_prob = predictions[0][top_idx]
        
        # Display results
        print(f"Predicted: {top_class} ({top_prob*100:.2f}%)")
        print(f"Expected: {test['expected']}")
        
        # Show all probabilities
        print("\nAll probabilities:")
        for i, prob in enumerate(predictions[0]):
            print(f"  {CLASSES[i]:20s}: {prob*100:6.2f}%")
        
        # Assess confidence
        if top_prob > 0.8:
            confidence = "HIGH"
        elif top_prob > 0.6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        print(f"\nConfidence Level: {confidence}")
        
        # Check if correct
        correct = top_class == test['expected']
        print(f"Result: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
        
        results.append({
            'description': test['description'],
            'predicted': top_class,
            'expected': test['expected'],
            'confidence': top_prob,
            'correct': correct
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct_count / total * 100
        
        print(f"\nAccuracy: {correct_count}/{total} ({accuracy:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Average Confidence: {avg_confidence*100:.1f}%")
        
        # Confidence distribution
        high_conf = sum(1 for r in results if r['confidence'] > 0.8)
        med_conf = sum(1 for r in results if 0.6 < r['confidence'] <= 0.8)
        low_conf = sum(1 for r in results if r['confidence'] <= 0.6)
        
        print(f"\nConfidence Distribution:")
        print(f"  High (>80%): {high_conf}")
        print(f"  Medium (60-80%): {med_conf}")
        print(f"  Low (<60%): {low_conf}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS FOR IMPROVEMENT:")
    print("-" * 70)
    print("1. Add more diverse training data (field images)")
    print("2. Implement test-time augmentation")
    print("3. Use ensemble of multiple models")
    print("4. Fine-tune on your specific use case")
    print("5. Collect and retrain on failure cases")

if __name__ == "__main__":
    test_model_robustness()