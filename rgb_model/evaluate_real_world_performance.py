#!/usr/bin/env python3
"""
Comprehensive evaluation on real-world images from the internet
This is the ultimate test - if it works here, it works in production
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import json
from datetime import datetime

class RealWorldEvaluator:
    """
    Test model on actual images from the internet
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
        self.class_names = [
            'Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus',
            'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust'
        ]
        
        # Real-world test images from internet
        self.test_urls = {
            'Blight': [
                'https://extension.umd.edu/sites/default/files/styles/optimized/public/2021-04/HGIC_veg_early%20blight%20plant_2019_IMG_0675.jpg',
                'https://www.gardeningknowhow.com/wp-content/uploads/2021/07/Corn-leaf-blight.jpg',
                'https://cdn.britannica.com/89/126689-004-D622CD2F/Potato-leaf-blight.jpg',
            ],
            'Healthy': [
                'https://gardenerspath.com/wp-content/uploads/2022/06/Tomato-Plant-with-Healthy-Green-Leaves.jpg',
                'https://www.thespruce.com/thmb/5_2oXyg5_RZQQs_hG0kMilnJTfQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/grow-healthy-tomato-plants-1402587-hero-9281f1b7e9c04ea1997dc0087a3b86e4.jpg',
            ],
            'Powdery_Mildew': [
                'https://www.planetnatural.com/wp-content/uploads/2023/10/powdery-mildew-leaves.jpg',
                'https://hort.extension.wisc.edu/files/2015/07/Powdery-mildew-on-pumpkin.jpg',
            ],
            'Leaf_Spot': [
                'https://www.missouribotanicalgarden.org/Portals/0/Gardening/Gardening%20Help/images/Pests/Bacterial_Leaf_Spot1885.jpg',
                'https://extension.umn.edu/sites/default/files/septoria-leaf-spot-tomato-Michelle-Grabowski.jpg',
            ],
            'Rust': [
                'https://www.gardeningknowhow.com/wp-content/uploads/2020/11/orange-rust.jpg',
                'https://extension.umn.edu/sites/default/files/bean-rust-Howard-Schwartz-CSU-Bugwood.jpg',
            ],
            'Mosaic_Virus': [
                'https://extension.umn.edu/sites/default/files/tomato-mosaic-virus.jpg',
                'https://www.missouribotanicalgarden.org/Portals/0/Gardening/Gardening%20Help/images/Pests/Tobacco_Mosaic2213.jpg',
            ],
            'Nutrient_Deficiency': [
                'https://www.haifa-group.com/sites/default/files/article/K_deficiency_leaf_2.jpg',
                'https://www.smart-fertilizer.com/wp-content/uploads/2016/05/Nitrogen-deficiency-in-corn.jpg',
            ]
        }
    
    def load_model(self):
        """Load the model for evaluation"""
        
        if self.model_path.endswith('.tflite'):
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_type = 'tflite'
        else:
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_type = 'keras'
    
    def download_image(self, url):
        """Download image from URL"""
        
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            return np.array(img)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        
        if image is None:
            return None
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize (adjust based on model training)
        image = image.astype(np.float32) / 255.0
        
        # For models trained with [-1,1] normalization
        # image = image * 2.0 - 1.0
        
        return image
    
    def predict(self, image):
        """Get model predictions"""
        
        if image is None:
            return None
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        if self.model_type == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], image_batch)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            predictions = self.model.predict(image_batch, verbose=0)[0]
        
        return predictions
    
    def evaluate_all(self):
        """Evaluate on all test images"""
        
        print("="*60)
        print("REAL-WORLD EVALUATION")
        print("="*60)
        
        results = {
            'total': 0,
            'correct': 0,
            'per_class_accuracy': {},
            'confusion_matrix': np.zeros((len(self.class_names), len(self.class_names))),
            'failed_downloads': 0,
            'predictions': []
        }
        
        for true_class, urls in self.test_urls.items():
            true_idx = self.class_names.index(true_class)
            class_correct = 0
            class_total = 0
            
            print(f"\nTesting {true_class}:")
            print("-" * 40)
            
            for url in urls:
                # Download image
                print(f"Downloading: {url[:50]}...")
                image = self.download_image(url)
                
                if image is None:
                    results['failed_downloads'] += 1
                    continue
                
                # Preprocess
                processed = self.preprocess_image(image)
                
                # Predict
                predictions = self.predict(processed)
                
                if predictions is None:
                    continue
                
                # Get top prediction
                pred_idx = np.argmax(predictions)
                pred_class = self.class_names[pred_idx]
                confidence = predictions[pred_idx]
                
                # Check if correct
                is_correct = pred_idx == true_idx
                
                # Update results
                results['total'] += 1
                class_total += 1
                
                if is_correct:
                    results['correct'] += 1
                    class_correct += 1
                
                results['confusion_matrix'][true_idx, pred_idx] += 1
                
                # Store prediction details
                results['predictions'].append({
                    'url': url,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'confidence': float(confidence),
                    'correct': is_correct
                })
                
                # Print result
                status = "✓" if is_correct else "✗"
                print(f"  {status} Predicted: {pred_class} ({confidence:.1%})")
                
                # Show top 3 predictions if wrong
                if not is_correct:
                    top_3 = np.argsort(predictions)[-3:][::-1]
                    print("    Top 3 predictions:")
                    for idx in top_3:
                        print(f"      - {self.class_names[idx]}: {predictions[idx]:.1%}")
            
            # Per-class accuracy
            if class_total > 0:
                class_accuracy = class_correct / class_total
                results['per_class_accuracy'][true_class] = class_accuracy
                print(f"  Class Accuracy: {class_accuracy:.1%} ({class_correct}/{class_total})")
        
        # Overall results
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)
        
        overall_accuracy = results['correct'] / max(results['total'], 1)
        print(f"\nOverall Accuracy: {overall_accuracy:.1%} ({results['correct']}/{results['total']})")
        
        print("\nPer-Class Accuracy:")
        for class_name, acc in results['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.1%}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print("True \\ Predicted ->", end="")
        for name in self.class_names:
            print(f"\t{name[:6]}", end="")
        print()
        
        for i, true_name in enumerate(self.class_names):
            print(f"{true_name[:10]:10}", end="")
            for j in range(len(self.class_names)):
                count = int(results['confusion_matrix'][i, j])
                print(f"\t{count}", end="")
            print()
        
        # Problem areas
        print("\nProblem Areas:")
        for pred in results['predictions']:
            if not pred['correct'] and pred['confidence'] > 0.5:
                print(f"  High confidence error: {pred['true_class']} -> {pred['predicted_class']} ({pred['confidence']:.1%})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results['confusion_matrix'] = results['confusion_matrix'].tolist()
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Performance assessment
        print("\n" + "="*60)
        print("PERFORMANCE ASSESSMENT")
        print("="*60)
        
        if overall_accuracy >= 0.85:
            print("✓ EXCELLENT: Model is production-ready!")
            print("  The model performs well on real-world images")
        elif overall_accuracy >= 0.75:
            print("✓ GOOD: Model is nearly ready")
            print("  Consider more training on failed cases")
        elif overall_accuracy >= 0.65:
            print("⚠ MODERATE: Model needs improvement")
            print("  Significant domain gap still exists")
        else:
            print("✗ POOR: Model not ready for production")
            print("  Major improvements needed")
        
        print("\nRecommendations:")
        if overall_accuracy < 0.85:
            print("1. Collect more real-world training data")
            print("2. Increase augmentation intensity")
            print("3. Use ensemble of models")
            print("4. Fine-tune on failed cases")
            print("5. Consider different architecture")
        
        return results


def test_on_custom_images():
    """Test on user-provided images"""
    
    print("\n" + "="*60)
    print("CUSTOM IMAGE TESTING")
    print("="*60)
    
    custom_dir = Path("custom_test_images")
    if not custom_dir.exists():
        print(f"\nCreate folder '{custom_dir}' and add test images")
        return
    
    evaluator = RealWorldEvaluator('models/best_cyclegan_model.h5')
    
    for img_path in custom_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"\nTesting: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            processed = evaluator.preprocess_image(image)
            
            # Predict
            predictions = evaluator.predict(processed)
            
            # Show results
            top_3 = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top_3):
                print(f"  {i+1}. {evaluator.class_names[idx]}: {predictions[idx]:.1%}")


def main():
    print("Real-World Model Evaluation")
    print("="*60)
    
    # Check for model
    model_path = Path('models/best_cyclegan_model.h5')
    
    if not model_path.exists():
        print("Model not found!")
        print("Train a model first using train_ultimate_model.py")
        return
    
    # Run evaluation
    evaluator = RealWorldEvaluator(model_path)
    results = evaluator.evaluate_all()
    
    # Test custom images if available
    test_on_custom_images()


if __name__ == "__main__":
    main()