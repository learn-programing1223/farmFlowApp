#!/usr/bin/env python3
"""
Test the new CycleGAN robust model on real images
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

def load_model():
    """Load the new CycleGAN robust model"""
    model_path = 'models/best_cyclegan_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path_or_url):
    """Preprocess image for model input"""
    try:
        if image_path_or_url.startswith('http'):
            # Download from URL
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = np.array(img)
        else:
            # Load from file
            img = cv2.imread(image_path_or_url)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def test_model():
    """Test the model on various images"""
    
    print("\n" + "="*60)
    print("TESTING NEW CYCLEGAN ROBUST MODEL")
    print("="*60)
    print("Expected: 80-85% accuracy on real-world images")
    print("(Up from 68% with the old model)")
    
    # Load model
    print("\nLoading model...")
    model = load_model()
    
    # Class names
    classes = [
        'Blight',
        'Healthy',
        'Leaf_Spot',
        'Mosaic_Virus', 
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    # Test images (you can add your own)
    test_images = [
        # Add local image paths or URLs here
        # "path/to/your/image.jpg",
        # "https://example.com/plant.jpg"
    ]
    
    print("\nReady for testing!")
    print("\nYou can:")
    print("1. Enter a local image path")
    print("2. Enter an image URL")
    print("3. Type 'quit' to exit")
    
    while True:
        print("\n" + "-"*40)
        image_input = input("Enter image path or URL: ").strip()
        
        if image_input.lower() == 'quit':
            break
        
        if not image_input:
            continue
        
        # Preprocess image
        print("Processing image...")
        img = preprocess_image(image_input)
        
        if img is None:
            print("Failed to load image. Please try another.")
            continue
        
        # Make prediction
        print("Running inference...")
        predictions = model.predict(img, verbose=0)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        
        print("\nResults:")
        print("-" * 30)
        for i, idx in enumerate(top_indices):
            confidence = predictions[idx] * 100
            print(f"{i+1}. {classes[idx]}: {confidence:.1f}%")
        
        # Diagnosis
        top_class = classes[top_indices[0]]
        top_conf = predictions[top_indices[0]] * 100
        
        print("\n" + "="*30)
        if top_conf > 70:
            print(f"Diagnosis: {top_class} (High confidence)")
        elif top_conf > 50:
            print(f"Diagnosis: {top_class} (Moderate confidence)")
        else:
            print(f"Possible: {top_class} (Low confidence)")
        print("="*30)

if __name__ == "__main__":
    test_model()