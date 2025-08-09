#!/usr/bin/env python3
"""
Final Model Evaluation on Test Set
This script evaluates the trained model on the held-out test set
that was never seen during training or validation.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model_on_test_set():
    """
    Evaluates the final model on the test set for unbiased performance metrics.
    """
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load the test set (never seen during training!)
    print("\nLoading held-out test set...")
    data_dir = Path('./data/splits')
    
    X_test = np.load(data_dir / 'X_test.npy').astype(np.float32)
    y_test = np.load(data_dir / 'y_test.npy').astype(np.float32)
    
    print(f"Test set size: {len(X_test):,} samples")
    print(f"Number of classes: {y_test.shape[1]}")
    print("This data was NEVER seen during training or validation!")
    
    # Load the best model
    print("\nLoading best model...")
    model_path = 'models/best_working_model.h5'
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET...")
    print("="*50)
    
    # Get predictions
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=1
    )
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Precision: {test_precision:.2%}")
    print(f"Test Recall: {test_recall:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate F1 Score
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"Test F1 Score: {f1_score:.2%}")
    
    # Get predictions for detailed analysis
    print("\nGenerating detailed predictions...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Load class names
    class_names = [
        'Blight',
        'Healthy', 
        'Leaf_Spot',
        'Mosaic_Virus',
        'Nutrient_Deficiency',
        'Powdery_Mildew',
        'Rust'
    ]
    
    # Generate classification report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=3)
    print(report)
    
    # Calculate confidence statistics
    print("\n" + "="*50)
    print("CONFIDENCE ANALYSIS")
    print("="*50)
    
    # Get max confidence for each prediction
    max_confidences = np.max(y_pred_probs, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (y_pred == y_true)
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    print(f"Average confidence (correct predictions): {correct_confidences.mean():.2%}")
    print(f"Average confidence (incorrect predictions): {incorrect_confidences.mean():.2%}")
    print(f"Predictions with >90% confidence: {(max_confidences > 0.9).sum()} ({(max_confidences > 0.9).mean():.1%})")
    print(f"Predictions with >95% confidence: {(max_confidences > 0.95).sum()} ({(max_confidences > 0.95).mean():.1%})")
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix without seaborn
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_test.png', dpi=150)
    print("Confusion matrix saved to: models/confusion_matrix_test.png")
    
    # Per-class accuracy
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20s}: {per_class_accuracy[i]:.2%}")
    
    # Find hardest examples (lowest confidence correct predictions)
    print("\n" + "="*50)
    print("ANALYSIS INSIGHTS")
    print("="*50)
    
    if len(incorrect_confidences) > 0:
        print(f"Total misclassifications: {len(incorrect_confidences)} out of {len(X_test)}")
        
        # Find which classes are most confused
        misclassified_true = y_true[~correct_mask]
        misclassified_pred = y_pred[~correct_mask]
        
        print("\nMost common confusions:")
        confusions = {}
        for true_idx, pred_idx in zip(misclassified_true, misclassified_pred):
            key = f"{class_names[true_idx]} -> {class_names[pred_idx]}"
            confusions[key] = confusions.get(key, 0) + 1
        
        # Sort by frequency
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        for confusion, count in sorted_confusions[:5]:
            print(f"  {confusion}: {count} times")
    else:
        print("Perfect classification! No errors found.")
    
    # Save evaluation results
    results = {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_loss': float(test_loss),
        'test_f1_score': float(f1_score),
        'total_test_samples': len(X_test),
        'average_confidence_correct': float(correct_confidences.mean()),
        'average_confidence_incorrect': float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else None,
        'high_confidence_90_percent': float((max_confidences > 0.9).mean()),
        'high_confidence_95_percent': float((max_confidences > 0.95).mean())
    }
    
    with open('models/test_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: models/test_evaluation_results.json")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if test_accuracy >= 0.90:
        print("✅ EXCELLENT: Model achieves >90% accuracy on unseen test data!")
        print("   This model is production-ready.")
    elif test_accuracy >= 0.85:
        print("✅ GOOD: Model achieves target 85% accuracy on test data.")
        print("   Ready for deployment with confidence thresholding.")
    elif test_accuracy >= 0.80:
        print("⚠️ ACCEPTABLE: Model achieves 80%+ accuracy.")
        print("   Consider additional training or ensemble methods.")
    else:
        print("❌ NEEDS IMPROVEMENT: Model below 80% on test data.")
        print("   Review training approach and data quality.")
    
    # Compare to validation accuracy
    print("\n" + "="*50)
    print("GENERALIZATION CHECK")
    print("="*50)
    print(f"Validation accuracy (during training): 95.52%")
    print(f"Test accuracy (never seen before): {test_accuracy:.2%}")
    gap = abs(0.9552 - test_accuracy)
    print(f"Generalization gap: {gap:.2%}")
    
    if gap < 0.05:
        print("✅ Excellent generalization! Model performs consistently.")
    elif gap < 0.10:
        print("✅ Good generalization. Minor difference is normal.")
    else:
        print("⚠️ Larger gap detected. Model may need regularization.")
    
    return results


def test_on_random_internet_images():
    """
    Optional: Test on completely new images from the internet
    """
    print("\n" + "="*70)
    print("TESTING ON NEW INTERNET IMAGES (Optional)")
    print("="*70)
    
    print("\nTo test on random internet images:")
    print("1. Download plant disease images from Google")
    print("2. Save them in 'rgb_model/test_images/' folder")
    print("3. Run this script again")
    
    test_images_dir = Path('./test_images')
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob('*.jpg')) + \
                     list(test_images_dir.glob('*.png'))
        
        if image_files:
            print(f"\nFound {len(image_files)} test images")
            # Add prediction code here if needed
        else:
            print("No images found in test_images folder")
    else:
        print(f"Creating folder: {test_images_dir}")
        test_images_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    # Main evaluation on test set
    results = evaluate_model_on_test_set()
    
    # Optional: test on internet images
    print("\n" + "-"*70)
    test_on_random_internet_images()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)