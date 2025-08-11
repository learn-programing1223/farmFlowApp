#!/usr/bin/env python3
"""
Test the model on real-world images from the internet
Place test images in rgb_model/test_images/ folder
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import json

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for the model"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def test_on_real_images():
    """Test the model on real-world images"""
    print("\n" + "="*70)
    print("TESTING ON REAL-WORLD IMAGES")
    print("="*70)
    
    # Load the model
    print("\nLoading model...")
    model_path = 'models/best_working_model.h5'
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
        
        # Create sample test with random data
        print("\n" + "-"*50)
        print("DEMO: Testing with random image to verify model works...")
        random_img = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(random_img, verbose=0)
        print(f"Model output shape: {prediction.shape}")
        print("Model is working correctly!")
        return
    
    print(f"\nFound {len(image_files)} test images")
    print("-" * 50)
    
    results = []
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Preprocess image
        img = preprocess_image(img_path)
        if img is None:
            continue
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        print(f"\nPredictions for {img_path.name}:")
        print("-" * 30)
        
        result = {
            'image': img_path.name,
            'predictions': []
        }
        
        for i, idx in enumerate(top_3_indices):
            confidence = predictions[0][idx]
            disease = class_names[idx]
            
            if i == 0:
                print(f"  PRIMARY: {disease:20s} - {confidence:.1%} confidence")
                result['primary_prediction'] = disease
                result['primary_confidence'] = float(confidence)
            else:
                print(f"           {disease:20s} - {confidence:.1%}")
            
            result['predictions'].append({
                'disease': disease,
                'confidence': float(confidence)
            })
        
        # Analysis
        primary_confidence = predictions[0][top_3_indices[0]]
        if primary_confidence > 0.90:
            print(f"\n  >> HIGH CONFIDENCE: Likely {class_names[top_3_indices[0]]}")
        elif primary_confidence > 0.70:
            print(f"\n  >> MODERATE CONFIDENCE: Probably {class_names[top_3_indices[0]]}")
        else:
            print(f"\n  >> LOW CONFIDENCE: Uncertain, possibly {class_names[top_3_indices[0]]}")
        
        results.append(result)
    
    # Save results
    with open('test_images/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tested {len(results)} images")
    print(f"Results saved to: test_images/results.json")
    
    # Show summary statistics
    if results:
        avg_confidence = np.mean([r['primary_confidence'] for r in results])
        high_conf = sum(1 for r in results if r['primary_confidence'] > 0.9)
        print(f"\nAverage confidence: {avg_confidence:.1%}")
        print(f"High confidence predictions (>90%): {high_conf}/{len(results)}")

def download_sample_diseased_images():
    """Provide URLs for sample diseased plant images to test"""
    print("\n" + "="*70)
    print("SAMPLE IMAGES TO TEST")
    print("="*70)
    print("\nYou can download these types of images from Google:")
    print("\n1. Search for: 'tomato early blight disease'")
    print("2. Search for: 'potato late blight leaves'")
    print("3. Search for: 'apple rust disease leaves'")
    print("4. Search for: 'powdery mildew cucumber'")
    print("5. Search for: 'mosaic virus tobacco leaves'")
    print("6. Search for: 'nitrogen deficiency corn leaves'")
    print("7. Search for: 'healthy tomato plant leaves'")
    print("\nSave images to: rgb_model/test_images/")
    print("Then run this script again!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REAL-WORLD IMAGE TESTING")
    print("="*70)
    
    # Test on real images
    test_on_real_images()
    
    # If no images found, show how to get them
    test_dir = Path('test_images')
    if not list(test_dir.glob('*.jpg')) and not list(test_dir.glob('*.png')):
        download_sample_diseased_images()