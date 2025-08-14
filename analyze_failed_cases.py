#!/usr/bin/env python3
"""
Analyze failed test cases to identify patterns and improve the model
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

print("="*70)
print("FAILED CASES ANALYSIS")
print("="*70)

# Paths
failed_dir = Path('failedImages')
feedback_log = Path('feedback_logs/feedback_log.jsonl')

# Check if we have failed cases
if not failed_dir.exists():
    print("\nNo failedImages directory found. Creating it...")
    failed_dir.mkdir(exist_ok=True)
    print("Directory created. Start collecting failed cases using app_with_feedback.py")
    exit()

# Count failed images
failed_images = list(failed_dir.glob('*.png')) + list(failed_dir.glob('*.jpg'))
print(f"\nFound {len(failed_images)} failed test cases")

if len(failed_images) == 0:
    print("\nNo failed images yet. Use the feedback app to collect misclassified cases.")
    exit()

# Analyze patterns
misclassification_matrix = defaultdict(lambda: defaultdict(int))
failure_reasons = []

# Parse filenames to understand misclassifications
for img_path in failed_images:
    filename = img_path.name
    
    # Expected format: "Predicted_predicted_but_Actual_actual_timestamp.png"
    if '_predicted_but_' in filename:
        parts = filename.split('_predicted_but_')
        predicted = parts[0]
        actual_parts = parts[1].split('_')
        actual = actual_parts[0] if actual_parts else 'Unknown'
        
        misclassification_matrix[actual][predicted] += 1
        failure_reasons.append({
            'file': filename,
            'predicted': predicted,
            'actual': actual
        })

# Load feedback log if it exists
if feedback_log.exists():
    print(f"\nLoading feedback log from {feedback_log}")
    with open(feedback_log, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if not entry.get('correct', True):
                    predicted = entry.get('predicted', 'Unknown')
                    actual = entry.get('actual', 'Unknown')
                    misclassification_matrix[actual][predicted] += 1
            except:
                continue

# Display analysis
print("\n" + "-"*50)
print("MISCLASSIFICATION PATTERNS")
print("-"*50)

# Most common misclassifications
all_misclassifications = []
for actual, predictions in misclassification_matrix.items():
    for predicted, count in predictions.items():
        if actual != predicted:
            all_misclassifications.append((actual, predicted, count))

all_misclassifications.sort(key=lambda x: x[2], reverse=True)

print("\nTop Misclassification Patterns:")
for i, (actual, predicted, count) in enumerate(all_misclassifications[:10], 1):
    print(f"{i}. {actual} -> {predicted}: {count} cases")

# Confusion matrix for failed cases
print("\n" + "-"*50)
print("CONFUSION MATRIX (Failed Cases Only)")
print("-"*50)

classes = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust', 'Unknown']

print(f"\n{'Actual/Predicted':20s}", end='')
for cls in classes:
    print(f"{cls[:8]:>10s}", end='')
print()

for actual_cls in classes:
    if actual_cls in misclassification_matrix:
        print(f"{actual_cls:20s}", end='')
        for pred_cls in classes:
            count = misclassification_matrix[actual_cls].get(pred_cls, 0)
            if count > 0:
                print(f"{count:>10d}", end='')
            else:
                print(f"{'':>10s}", end='')
        print()

# Identify problem areas
print("\n" + "-"*50)
print("KEY INSIGHTS")
print("-"*50)

# Classes most often misclassified
actual_error_counts = Counter()
for actual, predictions in misclassification_matrix.items():
    actual_error_counts[actual] = sum(predictions.values())

print("\nClasses Most Often Misclassified:")
for cls, count in actual_error_counts.most_common(5):
    print(f"  - {cls}: {count} errors")

# Classes the model confuses things for
predicted_error_counts = Counter()
for actual, predictions in misclassification_matrix.items():
    for predicted, count in predictions.items():
        if actual != predicted:
            predicted_error_counts[predicted] += count

print("\nClasses Model Incorrectly Predicts Most:")
for cls, count in predicted_error_counts.most_common(5):
    print(f"  - {cls}: {count} false positives")

# Specific problem patterns
print("\n" + "-"*50)
print("RECOMMENDED IMPROVEMENTS")
print("-"*50)

# Check for healthy plant issues
healthy_misclassified = actual_error_counts.get('Healthy', 0)
healthy_false_positives = predicted_error_counts.get('Healthy', 0)

if healthy_misclassified > len(failed_images) * 0.2:
    print("\n1. HEALTHY PLANT DETECTION ISSUE:")
    print("   - The model struggles to identify healthy plants")
    print("   - Consider increasing bias correction strength")
    print("   - May need more healthy plant training data")

if 'Blight' in predicted_error_counts and predicted_error_counts['Blight'] > len(failed_images) * 0.3:
    print("\n2. BLIGHT OVER-PREDICTION:")
    print("   - Model tends to predict Blight too often")
    print("   - Consider reducing Blight sensitivity")
    print("   - Check if training data is biased toward Blight")

# Check for confusion between similar diseases
similar_pairs = [
    ('Blight', 'Leaf_Spot'),
    ('Nutrient_Deficiency', 'Mosaic_Virus'),
    ('Powdery_Mildew', 'Rust')
]

for cls1, cls2 in similar_pairs:
    confusion_count = (misclassification_matrix[cls1].get(cls2, 0) + 
                      misclassification_matrix[cls2].get(cls1, 0))
    if confusion_count > 3:
        print(f"\n3. CONFUSION BETWEEN {cls1} and {cls2}:")
        print(f"   - {confusion_count} cases of confusion")
        print("   - These diseases may look similar")
        print("   - Consider additional training on distinguishing features")

# Image analysis recommendations
print("\n" + "-"*50)
print("NEXT STEPS")
print("-"*50)

print("\n1. Review failed images manually:")
print("   - Check if images are actually mislabeled")
print("   - Look for common visual patterns in failures")
print("   - Identify if certain image types cause problems")

print("\n2. Retrain with failed cases:")
print("   - Add correctly labeled failed images to training set")
print("   - Use data augmentation on problem classes")
print("   - Consider class weighting adjustments")

print("\n3. Adjust bias correction:")
if healthy_misclassified > 0:
    print("   - Increase healthy plant boost factor")
if 'Blight' in predicted_error_counts:
    print("   - Reduce Blight sensitivity")

print("\n4. Collect more feedback:")
print("   - Continue using app_with_feedback.py")
print("   - Focus on edge cases and difficult images")
print("   - Build a larger test set of real-world images")

# Generate report file
report_path = 'failure_analysis_report.txt'
with open(report_path, 'w') as f:
    f.write("FAILED CASES ANALYSIS REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total Failed Cases: {len(failed_images)}\n\n")
    
    f.write("Top Misclassification Patterns:\n")
    for i, (actual, predicted, count) in enumerate(all_misclassifications[:10], 1):
        f.write(f"{i}. {actual} -> {predicted}: {count} cases\n")
    
    f.write("\n\nRecommendations:\n")
    f.write("- Review images in failedImages/ folder\n")
    f.write("- Use insights to adjust model training\n")
    f.write("- Continue collecting feedback\n")

print(f"\n📊 Report saved to: {report_path}")
print("="*70)