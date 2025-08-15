#!/usr/bin/env python3
"""
Real-World Inference Script with Test-Time Augmentation
========================================================

Production-ready inference for plant disease detection on internet-sourced images.
Uses TTA from augmentation_pipeline for robust predictions.

Author: PlantPulse Team
Date: 2025
"""

import os
import sys
import argparse
import json
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
from data_loader_v2 import EnhancedDataLoader
from augmentation_pipeline import AugmentationPipeline
from losses import get_loss_by_name

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


class RealWorldInference:
    """
    Robust inference engine for real-world plant disease detection.
    Supports Test-Time Augmentation for improved accuracy.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessing_mode: str = 'default',
        use_tta: bool = True,
        tta_count: int = 5,
        confidence_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model (.h5 file)
            preprocessing_mode: Preprocessing mode ('default', 'fast', 'minimal', 'legacy')
            use_tta: Whether to use Test-Time Augmentation
            tta_count: Number of augmentations for TTA
            confidence_threshold: Minimum confidence for predictions
            verbose: Print detailed information
        """
        self.model_path = Path(model_path)
        self.preprocessing_mode = preprocessing_mode
        self.use_tta = use_tta
        self.tta_count = tta_count
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Class names (must match training)
        self.class_names = [
            'Blight', 'Healthy', 'Leaf_Spot', 
            'Mosaic_Virus', 'Nutrient_Deficiency', 'Powdery_Mildew'
        ]
        
        # Initialize components
        self._initialize_model()
        self._initialize_preprocessor()
        self._initialize_tta()
        
    def _initialize_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.verbose:
            print(f"Loading model from: {self.model_path}")
        
        # Load model with custom objects if needed
        custom_objects = {}
        try:
            # Try loading with potential custom losses
            from losses import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
            custom_objects = {
                'FocalLoss': FocalLoss,
                'LabelSmoothingCrossEntropy': LabelSmoothingCrossEntropy,
                'CombinedLoss': CombinedLoss
            }
        except:
            pass
        
        self.model = keras.models.load_model(
            self.model_path,
            custom_objects=custom_objects,
            compile=False  # We don't need optimizer for inference
        )
        
        # Compile with dummy optimizer for metrics
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if self.verbose:
            print(f"Model loaded successfully: {self.model.count_params():,} parameters")
    
    def _initialize_preprocessor(self):
        """Initialize the enhanced data loader for preprocessing."""
        self.data_loader = EnhancedDataLoader(
            data_dir=Path("."),  # Dummy path, we'll load individual images
            target_size=(224, 224),
            batch_size=1,
            use_advanced_preprocessing=True,
            preprocessing_mode=self.preprocessing_mode
        )
        
        if self.verbose:
            print(f"Preprocessor initialized with mode: {self.preprocessing_mode}")
    
    def _initialize_tta(self):
        """Initialize Test-Time Augmentation pipeline."""
        if self.use_tta:
            # Use the TTA pipelines from augmentation_pipeline
            # This returns a list of augmentation pipelines
            self.tta_pipelines = AugmentationPipeline.create_test_time_augmentation_pipeline()
            if self.verbose:
                print(f"TTA initialized with {len(self.tta_pipelines)} augmentation variations")
        else:
            self.tta_pipelines = None
            if self.verbose:
                print("TTA disabled")
    
    def predict_single_image(
        self,
        image_path: Union[str, Path],
        return_all_scores: bool = False
    ) -> Dict:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to the image
            return_all_scores: Return scores for all classes
            
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        # Start timing
        start_time = time.time()
        
        # Load and preprocess image
        preprocessed_image = self.data_loader.preprocess_image(
            str(image_path),
            apply_augmentation=False,
            is_training=False
        )
        
        preprocess_time = time.time() - start_time
        
        # Apply TTA if enabled
        if self.use_tta and self.tta_pipelines:
            predictions = self._predict_with_tta(preprocessed_image)
        else:
            predictions = self._predict_single(preprocessed_image)
        
        inference_time = time.time() - start_time - preprocess_time
        total_time = time.time() - start_time
        
        # Get top prediction
        top_class_idx = np.argmax(predictions)
        top_confidence = float(predictions[top_class_idx])
        top_class = self.class_names[top_class_idx]
        
        # Calculate uncertainty (entropy)
        entropy = -np.sum(predictions * np.log(predictions + 1e-10))
        uncertainty = entropy / np.log(len(self.class_names))  # Normalized
        
        result = {
            'image_path': str(image_path),
            'prediction': top_class,
            'confidence': top_confidence,
            'uncertainty': float(uncertainty),
            'meets_threshold': top_confidence >= self.confidence_threshold,
            'timing': {
                'preprocessing_ms': preprocess_time * 1000,
                'inference_ms': inference_time * 1000,
                'total_ms': total_time * 1000
            },
            'tta_used': self.use_tta,
            'preprocessing_mode': self.preprocessing_mode
        }
        
        if return_all_scores:
            result['all_scores'] = {
                class_name: float(score)
                for class_name, score in zip(self.class_names, predictions)
            }
        
        return result
    
    def _predict_single(self, image: np.ndarray) -> np.ndarray:
        """Single prediction without TTA."""
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image_batch, verbose=0)
        
        return predictions[0]
    
    def _predict_with_tta(self, image: np.ndarray) -> np.ndarray:
        """Prediction with Test-Time Augmentation."""
        predictions_list = []
        
        # Original prediction
        predictions_list.append(self._predict_single(image))
        
        # Apply TTA augmentations
        # Use min of tta_count and available pipelines
        num_augs = min(self.tta_count - 1, len(self.tta_pipelines))
        
        for i in range(num_augs):
            # Convert to uint8 for albumentations
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Apply augmentation from the pipeline list
            pipeline = self.tta_pipelines[i % len(self.tta_pipelines)]
            augmented = pipeline(image=image_uint8)
            augmented_image = augmented['image'].astype(np.float32) / 255.0
            
            # Predict on augmented image
            pred = self._predict_single(augmented_image)
            predictions_list.append(pred)
        
        # Average predictions
        averaged_predictions = np.mean(predictions_list, axis=0)
        
        # Calculate variance for uncertainty estimation
        if self.verbose:
            variance = np.var(predictions_list, axis=0)
            max_variance = np.max(variance)
            print(f"TTA variance: {max_variance:.4f}")
        
        return averaged_predictions
    
    def predict_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
        output_format: str = 'json'
    ) -> List[Dict]:
        """
        Predict diseases for all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            recursive: Search subdirectories
            output_format: Output format ('json' or 'csv')
            
        Returns:
            List of prediction results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all images
        if recursive:
            image_files = []
            for fmt in SUPPORTED_FORMATS:
                image_files.extend(directory_path.rglob(f'*{fmt}'))
        else:
            image_files = []
            for fmt in SUPPORTED_FORMATS:
                image_files.extend(directory_path.glob(f'*{fmt}'))
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        for idx, image_path in enumerate(image_files, 1):
            try:
                if self.verbose:
                    print(f"\nProcessing [{idx}/{len(image_files)}]: {image_path.name}")
                
                result = self.predict_single_image(image_path, return_all_scores=True)
                results.append(result)
                
                if self.verbose:
                    print(f"  Prediction: {result['prediction']} "
                          f"(confidence: {result['confidence']:.2%})")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'json':
            output_file = f'inference_results_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:  # csv
            output_file = f'inference_results_{timestamp}.csv'
            if results and 'error' not in results[0]:
                fieldnames = list(results[0].keys())
                # Flatten nested dictionaries
                for result in results:
                    if 'timing' in result:
                        for key, value in result['timing'].items():
                            result[f'timing_{key}'] = value
                        del result['timing']
                    if 'all_scores' in result:
                        for key, value in result['all_scores'].items():
                            result[f'score_{key}'] = value
                        del result['all_scores']
                
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary statistics
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return
        
        print("\n" + "=" * 70)
        print("INFERENCE SUMMARY")
        print("=" * 70)
        
        # Class distribution
        class_counts = {}
        for result in valid_results:
            pred = result['prediction']
            class_counts[pred] = class_counts.get(pred, 0) + 1
        
        print("\nClass Distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(valid_results)) * 100
            print(f"  {class_name:20} {count:3d} ({percentage:5.1f}%)")
        
        # Confidence statistics
        confidences = [r['confidence'] for r in valid_results]
        print(f"\nConfidence Statistics:")
        print(f"  Mean:   {np.mean(confidences):.2%}")
        print(f"  Median: {np.median(confidences):.2%}")
        print(f"  Min:    {np.min(confidences):.2%}")
        print(f"  Max:    {np.max(confidences):.2%}")
        
        # Timing statistics
        if 'timing' in valid_results[0]:
            total_times = [r['timing']['total_ms'] for r in valid_results]
            print(f"\nTiming Statistics (ms):")
            print(f"  Mean:   {np.mean(total_times):.2f}")
            print(f"  Median: {np.median(total_times):.2f}")
            print(f"  Min:    {np.min(total_times):.2f}")
            print(f"  Max:    {np.max(total_times):.2f}")
        
        # Threshold statistics
        above_threshold = sum(1 for r in valid_results if r['meets_threshold'])
        print(f"\nThreshold Statistics (>= {self.confidence_threshold:.0%}):")
        print(f"  Above threshold: {above_threshold}/{len(valid_results)} "
              f"({(above_threshold/len(valid_results))*100:.1f}%)")
    
    def benchmark_tta(self, image_path: Union[str, Path], iterations: int = 10):
        """
        Benchmark TTA vs non-TTA performance.
        
        Args:
            image_path: Path to test image
            iterations: Number of iterations for benchmarking
        """
        image_path = Path(image_path)
        
        print("\n" + "=" * 70)
        print("TTA BENCHMARK")
        print("=" * 70)
        print(f"Image: {image_path.name}")
        print(f"Iterations: {iterations}")
        
        # Benchmark without TTA
        self.use_tta = False
        times_no_tta = []
        
        print("\nWithout TTA:")
        for i in range(iterations):
            start = time.time()
            result_no_tta = self.predict_single_image(image_path)
            times_no_tta.append((time.time() - start) * 1000)
            
            if i == 0:
                print(f"  Prediction: {result_no_tta['prediction']}")
                print(f"  Confidence: {result_no_tta['confidence']:.2%}")
        
        print(f"  Mean time: {np.mean(times_no_tta):.2f} ms")
        print(f"  Std dev:   {np.std(times_no_tta):.2f} ms")
        
        # Benchmark with TTA
        self.use_tta = True
        times_with_tta = []
        
        print("\nWith TTA:")
        for i in range(iterations):
            start = time.time()
            result_tta = self.predict_single_image(image_path)
            times_with_tta.append((time.time() - start) * 1000)
            
            if i == 0:
                print(f"  Prediction: {result_tta['prediction']}")
                print(f"  Confidence: {result_tta['confidence']:.2%}")
                print(f"  Uncertainty: {result_tta['uncertainty']:.4f}")
        
        print(f"  Mean time: {np.mean(times_with_tta):.2f} ms")
        print(f"  Std dev:   {np.std(times_with_tta):.2f} ms")
        
        # Comparison
        speedup = np.mean(times_with_tta) / np.mean(times_no_tta)
        print(f"\nTTA Overhead: {speedup:.2f}x slower")
        
        # Confidence comparison
        if result_no_tta['prediction'] == result_tta['prediction']:
            print(f"Predictions match: {result_tta['prediction']}")
            conf_diff = result_tta['confidence'] - result_no_tta['confidence']
            print(f"Confidence difference: {conf_diff:+.2%}")
        else:
            print(f"Predictions differ: {result_no_tta['prediction']} vs {result_tta['prediction']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-world inference for plant disease detection'
    )
    
    # Input arguments
    parser.add_argument('input_path', type=str,
                       help='Path to image file or directory')
    parser.add_argument('--model_path', type=str,
                       default='models/enhanced_best.h5',
                       help='Path to trained model')
    
    # Preprocessing arguments
    parser.add_argument('--preprocessing_mode', type=str,
                       default='default',
                       choices=['default', 'fast', 'minimal', 'legacy'],
                       help='Preprocessing mode')
    
    # TTA arguments
    parser.add_argument('--use_tta', action='store_true', default=True,
                       help='Use Test-Time Augmentation')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false',
                       help='Disable Test-Time Augmentation')
    parser.add_argument('--tta_count', type=int, default=5,
                       help='Number of augmentations for TTA')
    
    # Output arguments
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold')
    parser.add_argument('--output_format', type=str, default='json',
                       choices=['json', 'csv'],
                       help='Output format for batch predictions')
    parser.add_argument('--recursive', action='store_true',
                       help='Search subdirectories for images')
    
    # Other arguments
    parser.add_argument('--benchmark', action='store_true',
                       help='Run TTA benchmark on single image')
    parser.add_argument('--benchmark_iterations', type=int, default=10,
                       help='Number of iterations for benchmarking')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed information')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                       help='Minimal output')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("REAL-WORLD PLANT DISEASE INFERENCE")
    print("=" * 70)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        # Try alternative paths
        alt_paths = [
            Path('models/plantvillage_robust_best.h5'),
            Path('models/plantvillage_robust_final.h5'),
            Path('models/robust_plantvillage_best.h5')
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"Using alternative model: {alt_path}")
                model_path = alt_path
                break
        else:
            print(f"ERROR: No model found. Tried:")
            print(f"  - {args.model_path}")
            for alt_path in alt_paths:
                print(f"  - {alt_path}")
            return
    
    # Initialize inference engine
    print(f"\nInitializing inference engine...")
    print(f"  Model: {model_path}")
    print(f"  Preprocessing: {args.preprocessing_mode}")
    print(f"  TTA: {'Enabled' if args.use_tta else 'Disabled'}")
    
    inference = RealWorldInference(
        model_path=str(model_path),
        preprocessing_mode=args.preprocessing_mode,
        use_tta=args.use_tta,
        tta_count=args.tta_count,
        confidence_threshold=args.confidence_threshold,
        verbose=args.verbose
    )
    
    input_path = Path(args.input_path)
    
    # Run benchmark if requested
    if args.benchmark:
        if input_path.is_file():
            inference.benchmark_tta(input_path, args.benchmark_iterations)
        else:
            print("Benchmark requires a single image file")
        return
    
    # Process input
    if input_path.is_file():
        # Single image prediction
        print(f"\nProcessing single image: {input_path.name}")
        result = inference.predict_single_image(input_path, return_all_scores=True)
        
        # Print results
        print("\n" + "-" * 70)
        print("PREDICTION RESULTS")
        print("-" * 70)
        print(f"Image: {result['image_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Uncertainty: {result['uncertainty']:.4f}")
        print(f"Meets threshold: {'Yes' if result['meets_threshold'] else 'No'}")
        
        if 'all_scores' in result:
            print("\nAll class scores:")
            for class_name, score in sorted(result['all_scores'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"  {class_name:20} {score:.2%}")
        
        print(f"\nTiming:")
        print(f"  Preprocessing: {result['timing']['preprocessing_ms']:.2f} ms")
        print(f"  Inference:     {result['timing']['inference_ms']:.2f} ms")
        print(f"  Total:         {result['timing']['total_ms']:.2f} ms")
        
        # Save single result
        output_file = f"inference_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")
        
    elif input_path.is_dir():
        # Directory batch prediction
        print(f"\nProcessing directory: {input_path}")
        results = inference.predict_directory(
            input_path,
            recursive=args.recursive,
            output_format=args.output_format
        )
        
    else:
        print(f"ERROR: Path not found: {input_path}")


if __name__ == "__main__":
    main()