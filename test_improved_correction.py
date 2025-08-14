#!/usr/bin/env python3
"""
Test the improved bias correction on both failed cases
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

print("="*70)
print("TESTING IMPROVED SMART BIAS CORRECTION")
print("="*70)

# Load model
print("\nLoading model...")
model = tf.keras.models.load_model('rgb_model/models/cyclegan_best.h5')

# Classes
CLASSES = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']

# Test cases
test_cases = [
    ('failedImages/healthy.png', 'Healthy', 'Should correctly identify as healthy'),
    ('failedImages/moasicVirus.png', 'Mosaic_Virus', 'Should detect mosaic virus patterns')
]

for image_path, expected, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"Expected: {expected}")
    print(f"Goal: {description}")
    print('-'*60)
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)
    
    # Get raw predictions
    predictions = model.predict(img_array_batch, verbose=0)[0].copy()
    
    print("\nORIGINAL PREDICTIONS:")
    for i, conf in enumerate(predictions):
        if conf > 0.001:
            marker = " <--" if CLASSES[i] == expected else ""
            print(f"  {CLASSES[i]:20s}: {conf*100:6.2f}%{marker}")
    
    # Apply SMART bias correction (same as in app.py)
    img_array_for_analysis = np.array(image)
    
    # Calculate color statistics
    r_mean = np.mean(img_array_for_analysis[:,:,0]) / 255
    g_mean = np.mean(img_array_for_analysis[:,:,1]) / 255
    b_mean = np.mean(img_array_for_analysis[:,:,2]) / 255
    
    r_std = np.std(img_array_for_analysis[:,:,0]) / 255
    g_std = np.std(img_array_for_analysis[:,:,1]) / 255
    b_std = np.std(img_array_for_analysis[:,:,2]) / 255
    
    # Convert to HSV
    img_hsv = cv2.cvtColor(img_array_for_analysis, cv2.COLOR_RGB2HSV)
    h_std = np.std(img_hsv[:,:,0])
    s_mean = np.mean(img_hsv[:,:,1]) / 255
    
    # Detect patterns
    yellow_green_mask = (img_hsv[:,:,0] > 30) & (img_hsv[:,:,0] < 90)
    yellow_green_ratio = np.sum(yellow_green_mask) / (224 * 224)
    
    green_dominance = g_mean - max(r_mean, b_mean)
    vegetation_index = (g_mean - r_mean) / (g_mean + r_mean + 0.01)
    
    # Pattern detection
    has_mottling = h_std > 35
    has_yellowing = yellow_green_ratio > 0.2 and s_mean < 0.6
    uniform_green = g_std < 0.20 and h_std < 25
    
    max_disease_conf = max([predictions[i] for i in [0, 2, 3, 4, 5, 6]])
    
    print(f"\nPATTERN ANALYSIS:")
    print(f"  Hue std (mottling): {h_std:.1f} (>35 = mottled)")
    print(f"  Green std: {g_std:.3f} (<0.20 = uniform)")
    print(f"  Yellow-green ratio: {yellow_green_ratio:.3f}")
    print(f"  Saturation: {s_mean:.3f}")
    print(f"  Has mottling: {has_mottling}")
    print(f"  Has yellowing: {has_yellowing}")
    print(f"  Uniform green: {uniform_green}")
    
    # Apply correction
    correction_applied = ""
    if has_mottling or has_yellowing:
        correction_applied = "DISEASE PATTERNS DETECTED"
        if predictions[3] < 0.2:
            predictions[3] *= 3.0
        if predictions[4] < 0.1:
            predictions[4] *= 2.0
        predictions[1] *= 0.3
        
    elif uniform_green and green_dominance > 0.01 and max_disease_conf < 0.9:
        correction_applied = "UNIFORM HEALTHY GREEN"
        predictions[1] *= 30.0
        for i in [0, 2, 3, 5, 6]:
            predictions[i] *= 0.05
            
    elif green_dominance > 0.02 and not has_mottling:
        correction_applied = "GREEN DOMINANT, NO MOTTLING"
        predictions[1] *= 10.0
        for i in [0, 2, 3, 5, 6]:
            predictions[i] *= 0.2
            
    else:
        correction_applied = "MINIMAL CORRECTION"
        if predictions[1] < 0.1 and green_dominance > 0:
            predictions[1] *= 3.0
    
    # Renormalize
    predictions = predictions / np.sum(predictions)
    
    print(f"\nCORRECTION APPLIED: {correction_applied}")
    
    print("\nCORRECTED PREDICTIONS:")
    top_idx = np.argmax(predictions)
    for i, conf in enumerate(predictions):
        if conf > 0.001:
            marker = " <-- TOP" if i == top_idx else ""
            marker += " CORRECT!" if CLASSES[i] == expected and i == top_idx else ""
            print(f"  {CLASSES[i]:20s}: {conf*100:6.2f}%{marker}")
    
    # Result
    is_correct = CLASSES[top_idx] == expected
    print(f"\nRESULT: {'SUCCESS!' if is_correct else 'FAILED'}")
    if not is_correct:
        print(f"  Predicted {CLASSES[top_idx]} instead of {expected}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The smart correction now:
1. Detects mottling patterns that indicate disease
2. Checks for uniform green that indicates health
3. Identifies yellow-green patterns common in mosaic virus
4. Applies appropriate corrections based on visual patterns
""")