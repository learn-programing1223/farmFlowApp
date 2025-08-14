#!/usr/bin/env python3
"""
Test both failed cases to understand the bias correction issue
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

print("="*70)
print("TESTING FAILED CASES - HEALTHY vs MOSAIC VIRUS")
print("="*70)

# Load model
print("\nLoading 96% accuracy model...")
model = tf.keras.models.load_model('rgb_model/models/cyclegan_best.h5')

# Classes
CLASSES = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']

# Test both images
test_cases = [
    ('failedImages/healthy.png', 'Actually Healthy'),
    ('failedImages/moasicVirus.png', 'Actually Mosaic Virus')
]

for image_path, actual_condition in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing: {image_path}")
    print(f"Actual Condition: {actual_condition}")
    print('-'*50)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)
    
    # Get raw predictions
    predictions_raw = model.predict(img_array_batch, verbose=0)[0]
    
    print("\nRAW MODEL PREDICTIONS:")
    for i, conf in enumerate(predictions_raw):
        if conf > 0.001:  # Only show significant predictions
            print(f"  {CLASSES[i]:20s}: {conf*100:6.2f}%")
    
    # Analyze image characteristics
    img_np = np.array(image_resized)
    
    # Color statistics
    r_mean = np.mean(img_np[:,:,0]) / 255
    g_mean = np.mean(img_np[:,:,1]) / 255
    b_mean = np.mean(img_np[:,:,2]) / 255
    
    # Color variance (important for detecting mottled patterns)
    r_std = np.std(img_np[:,:,0]) / 255
    g_std = np.std(img_np[:,:,1]) / 255
    b_std = np.std(img_np[:,:,2]) / 255
    
    # Convert to HSV for better pattern detection
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h_mean = np.mean(img_hsv[:,:,0]) / 180  # Hue
    s_mean = np.mean(img_hsv[:,:,1]) / 255  # Saturation
    v_mean = np.mean(img_hsv[:,:,2]) / 255  # Value
    
    # Detect yellow-green patterns (common in mosaic virus)
    yellow_green_mask = (img_hsv[:,:,0] > 30) & (img_hsv[:,:,0] < 90)  # Yellow-green hue range
    yellow_green_ratio = np.sum(yellow_green_mask) / (224 * 224)
    
    # Detect color variation (mottling)
    color_variance = np.std(img_hsv[:,:,0])  # Variance in hue
    
    print(f"\nIMAGE CHARACTERISTICS:")
    print(f"  RGB means: R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f}")
    print(f"  RGB std devs: R={r_std:.3f}, G={g_std:.3f}, B={b_std:.3f}")
    print(f"  HSV means: H={h_mean:.3f}, S={s_mean:.3f}, V={v_mean:.3f}")
    print(f"  Yellow-green ratio: {yellow_green_ratio:.3f}")
    print(f"  Color variance (mottling): {color_variance:.1f}")
    
    # Current bias correction logic
    green_dominance = g_mean - max(r_mean, b_mean)
    vegetation_index = (g_mean - r_mean) / (g_mean + r_mean + 0.01)
    is_very_green = green_dominance > 0.02
    has_vegetation = vegetation_index > 0.03
    no_brown = (r_mean - b_mean) < 0.05
    good_green = g_mean > 0.6
    
    print(f"\nCURRENT BIAS DETECTION:")
    print(f"  Green dominance: {green_dominance:.3f} (threshold > 0.02)")
    print(f"  Vegetation index: {vegetation_index:.3f} (threshold > 0.03)")
    print(f"  Good green: {good_green} (G > 0.6)")
    print(f"  No brown: {no_brown}")
    print(f"  -> Would apply boost: {(is_very_green or good_green) and (has_vegetation or no_brown)}")
    
    # Apply current correction
    predictions_corrected = predictions_raw.copy()
    if (is_very_green or good_green) and (has_vegetation or no_brown):
        predictions_corrected[1] *= 50.0  # Healthy boost
        for i in [0, 2, 3, 5, 6]:  # Disease indices
            predictions_corrected[i] *= 0.02
    predictions_corrected = predictions_corrected / np.sum(predictions_corrected)
    
    print(f"\nWITH CURRENT CORRECTION:")
    top_idx = np.argmax(predictions_corrected)
    print(f"  Predicted: {CLASSES[top_idx]} ({predictions_corrected[top_idx]*100:.1f}%)")
    
    # Suggested improved detection
    has_mottling = color_variance > 20  # High hue variance indicates mottling
    has_yellowing = yellow_green_ratio > 0.3 and s_mean < 0.7  # Yellow-green with lower saturation
    uniform_green = g_std < 0.15 and color_variance < 15  # Uniform green color
    
    print(f"\nIMPROVED DETECTION FEATURES:")
    print(f"  Has mottling pattern: {has_mottling} (variance={color_variance:.1f})")
    print(f"  Has yellowing: {has_yellowing}")
    print(f"  Uniform green: {uniform_green}")
    
    # Suggested correction
    predictions_improved = predictions_raw.copy()
    
    if has_mottling or has_yellowing:
        # Don't boost healthy for mottled/yellowed plants
        print(f"  -> Detected disease patterns, NOT boosting healthy")
        # Maybe even boost disease predictions slightly
        predictions_improved[3] *= 2.0  # Boost Mosaic_Virus
        predictions_improved[4] *= 1.5  # Boost Nutrient_Deficiency
    elif uniform_green and good_green:
        # Only boost healthy for uniformly green plants
        print(f"  -> Uniform healthy green, boosting healthy")
        predictions_improved[1] *= 50.0
        for i in [0, 2, 3, 5, 6]:
            predictions_improved[i] *= 0.02
    else:
        print(f"  -> Ambiguous, using moderate correction")
        predictions_improved[1] *= 5.0  # Moderate boost
    
    predictions_improved = predictions_improved / np.sum(predictions_improved)
    
    print(f"\nWITH IMPROVED CORRECTION:")
    top_idx_improved = np.argmax(predictions_improved)
    print(f"  Predicted: {CLASSES[top_idx_improved]} ({predictions_improved[top_idx_improved]*100:.1f}%)")
    
    # Check if improved
    is_correct = (actual_condition == "Actually Healthy" and CLASSES[top_idx_improved] == "Healthy") or \
                 (actual_condition == "Actually Mosaic Virus" and CLASSES[top_idx_improved] == "Mosaic_Virus")
    
    if is_correct:
        print(f"  CORRECT!")
    else:
        print(f"  Still incorrect")

print("\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)
print("""
1. Add mottling detection using color variance
2. Check for yellow-green patterns specific to mosaic virus  
3. Only boost healthy for uniformly green plants
4. Consider HSV color space for better disease pattern detection
5. Use texture analysis to detect veiny/mottled patterns
""")