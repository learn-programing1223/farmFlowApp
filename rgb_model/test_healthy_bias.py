#!/usr/bin/env python3
"""
Test why the model struggles with healthy plants
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from collections import Counter

print("="*70)
print("TESTING HEALTHY PLANT DETECTION ISSUE")
print("="*70)

# Load model
print("\nLoading your 96% model...")
model = tf.keras.models.load_model('models/cyclegan_best.h5')

# Class names
classes = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']

# Test on healthy images specifically
test_path = Path('datasets/ultimate_cyclegan/test/Healthy')

if not test_path.exists():
    print("ERROR: Test dataset not found")
    exit(1)

print(f"\nTesting on {len(list(test_path.glob('*.jpg')))} healthy test images...")

# Analyze predictions
predictions_summary = Counter()
confidence_when_correct = []
confidence_when_wrong = []
misclassified_as = Counter()

# Test each healthy image
healthy_images = list(test_path.glob('*.jpg'))[:50]  # Test first 50

print("\nAnalyzing healthy plant predictions...\n")

for img_path in healthy_images:
    # Load and preprocess
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img, verbose=0)[0]
    pred_class = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    predictions_summary[pred_class] += 1
    
    if pred_class == 'Healthy':
        confidence_when_correct.append(confidence)
    else:
        confidence_when_wrong.append(confidence)
        misclassified_as[pred_class] += 1
        
        # Show specific misclassifications
        if len(misclassified_as) <= 5:  # Show first 5 errors
            print(f"ERROR: {img_path.name}: Predicted as {pred_class} ({confidence:.1f}%)")

# Calculate statistics
total = len(healthy_images)
correct = predictions_summary['Healthy']
accuracy = (correct / total) * 100

print("\n" + "="*70)
print("HEALTHY PLANT DETECTION RESULTS")
print("="*70)

print(f"\nCORRECT: Correctly identified as Healthy: {correct}/{total} ({accuracy:.1f}%)")

if correct > 0:
    avg_confidence_correct = np.mean(confidence_when_correct)
    print(f"   Average confidence when correct: {avg_confidence_correct:.1f}%")

if len(confidence_when_wrong) > 0:
    avg_confidence_wrong = np.mean(confidence_when_wrong)
    print(f"\nERROR: Misclassified as diseased: {total-correct}/{total}")
    print(f"   Average confidence when wrong: {avg_confidence_wrong:.1f}%")
    
    print("\nMost common misclassifications:")
    for disease, count in misclassified_as.most_common(3):
        percentage = (count / (total-correct)) * 100
        print(f"   - {disease}: {count} times ({percentage:.0f}% of errors)")

# Analyze prediction distribution
print("\nFull prediction breakdown:")
for cls, count in predictions_summary.most_common():
    percentage = (count / total) * 100
    print(f"   {cls:20s}: {count:3d} ({percentage:5.1f}%)")

# Check if model has systematic bias
print("\nDiagnosis:")
if accuracy < 50:
    print("   WARNING - SEVERE ISSUE: Model is biased against healthy plants!")
    print("   - The model tends to find disease even when there is none")
    
    if 'Nutrient_Deficiency' in misclassified_as:
        print("   - Often confuses healthy green with nutrient deficiency")
    if 'Leaf_Spot' in misclassified_as:
        print("   - May interpret shadows/artifacts as leaf spots")
    if 'Powdery_Mildew' in misclassified_as:
        print("   - Might see reflections/shine as powdery mildew")
        
elif accuracy < 80:
    print("   WARNING - MODERATE ISSUE: Healthy detection needs improvement")
    print("   - Model has difficulty with clean, disease-free plants")
else:
    print("   CORRECT: Healthy detection is actually working well!")
    print("   - Issue might be with specific image types")

print("\nRecommendations:")
if accuracy < 70:
    print("   1. The CycleGAN augmentation may be too aggressive on healthy images")
    print("   2. Consider retraining with more emphasis on healthy plants")
    print("   3. Adjust prediction threshold - require higher confidence for disease")
    print("   4. The model may have learned that 'outdoor/field = diseased'")

print("\n" + "="*70)