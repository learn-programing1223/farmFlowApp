#!/usr/bin/env python3
"""
Test the trained model on real-world plant disease images
Downloads test images from the internet and evaluates model performance
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
from pathlib import Path

def download_test_image(url, name):
    """Download a test image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except Exception as e:
        print(f"Error downloading {name}: {str(e)[:50]}")
    return None

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def test_model():
    """Test model on real-world images"""
    
    print("=" * 60)
    print("REAL-WORLD MODEL TESTING")
    print("=" * 60)
    
    # Check for model file
    model_paths = [
        'models/robust_final_best.h5',
        'models/robust_final_model.h5',
        'models/best_cyclegan_model.h5'
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            print(f"\nLoading model: {path}")
            model = tf.keras.models.load_model(path, compile=False)
            break
    
    if model is None:
        print("No model found! Please train first.")
        return
    
    # Class names
    class_names = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus',
                   'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']
    
    # Test images with known conditions
    test_images = {
        'Blight (Tomato)': 'https://extension.umd.edu/sites/default/files/styles/optimized/public/2021-04/HGIC_veg_early%20blight%20plant_2019_IMG_0675.jpg',
        'Blight (Potato)': 'https://cdn.britannica.com/89/126689-004-D622CD2F/Potato-leaf-blight.jpg',
        'Healthy (Tomato)': 'https://www.almanac.com/sites/default/files/styles/or/public/image_nodes/tomatoes-growing.jpg',
        'Powdery Mildew': 'https://www.planetnatural.com/wp-content/uploads/2023/10/powdery-mildew-leaves.jpg',
        'Leaf Spot': 'https://www.missouribotanicalgarden.org/Portals/0/Gardening/Gardening%20Help/images/Pests/Bacterial_Leaf_Spot1885.jpg',
        'Rust': 'https://www.gardeningknowhow.com/wp-content/uploads/2020/11/orange-rust.jpg',
        'Mosaic Virus': 'https://www.missouribotanicalgarden.org/Portals/0/Gardening/Gardening%20Help/images/Pests/Tobacco_Mosaic2213.jpg',
    }
    
    print("\nTesting on real-world images:")
    print("-" * 40)
    
    correct = 0
    total = 0
    
    for name, url in test_images.items():
        print(f"\nTesting: {name}")
        expected = name.split(' (')[0]
        
        # Download image
        img = download_test_image(url, name)
        if img is None:
            continue
        
        # Preprocess
        img_array = preprocess_image(img)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = class_names[predicted_idx]
        
        # Check if correct
        is_correct = predicted_class == expected
        if is_correct:
            correct += 1
        total += 1
        
        # Print results
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted_class} ({confidence:.2%})")
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
        # Show top 3 predictions
        top_3 = np.argsort(predictions[0])[-3:][::-1]
        print("  Top 3 predictions:")
        for idx in top_3:
            print(f"    - {class_names[idx]}: {predictions[0][idx]:.2%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy < 50:
        print("\n⚠ Model needs more training!")
        print("The synthetic data may not be sufficient.")
        print("Consider:")
        print("1. Training for more epochs")
        print("2. Downloading real PlantVillage data")
        print("3. Using transfer learning from ImageNet")
    elif accuracy < 80:
        print("\n⚠ Model performance is moderate")
        print("Some improvement needed for production use")
    else:
        print("\n✓ Model is performing well!")
        print("Ready for deployment")

if __name__ == "__main__":
    test_model()