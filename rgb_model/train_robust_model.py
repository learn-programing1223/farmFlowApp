#!/usr/bin/env python3
"""
Train Robust RGB Universal Plant Disease Detection Model
Target: 80%+ validation accuracy with PlantVillage + PlantDoc datasets
"""

# Suppress protobuf warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

import argparse
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import MultiDatasetLoader
from training_robust import RobustProgressiveTrainer
from tflite_converter import TFLiteConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Robust RGB Plant Disease Detection Model'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Base directory containing datasets')
    parser.add_argument('--output-dir', type=str, default='./models/rgb_robust',
                       help='Directory to save models and logs')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--samples-per-class', type=int, default=1000,
                       help='Samples per class for balanced dataset (default: 1000)')
    parser.add_argument('--plantvillage-subset', type=float, default=1.0,
                       help='Fraction of PlantVillage to use (default: 1.0)')
    
    # Model arguments
    parser.add_argument('--dropout-rate', type=float, default=0.4,
                       help='Dropout rate (default: 0.4)')
    parser.add_argument('--l2-regularization', type=float, default=0.0005,
                       help='L2 regularization (default: 0.0005)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--stage1-epochs', type=int, default=15,
                       help='Epochs for stage 1 (default: 15)')
    parser.add_argument('--stage2-epochs', type=int, default=20,
                       help='Epochs for stage 2 (default: 20)')
    parser.add_argument('--stage3-epochs', type=int, default=10,
                       help='Epochs for stage 3 (default: 10)')
    
    # Learning rates
    parser.add_argument('--stage1-lr', type=float, default=0.001,
                       help='Learning rate for stage 1 (default: 0.001)')
    parser.add_argument('--stage2-lr', type=float, default=0.0001,
                       help='Learning rate for stage 2 (default: 0.0001)')
    parser.add_argument('--stage3-lr', type=float, default=0.00001,
                       help='Learning rate for stage 3 (default: 0.00001)')
    
    # Advanced training options
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='Number of warmup epochs (default: 3)')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use focal loss for class imbalance')
    parser.add_argument('--focal-alpha', type=float, default=0.75,
                       help='Focal loss alpha (default: 0.75)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--mixup-alpha', type=float, default=0.3,
                       help='MixUp alpha (default: 0.3)')
    parser.add_argument('--cutmix-alpha', type=float, default=1.5,
                       help='CutMix alpha (default: 1.5)')
    
    # Other options
    parser.add_argument('--use-cache', action='store_true', default=False,
                       help='Use cached dataset if available')
    parser.add_argument('--use-augmented', action='store_true', default=False,
                       help='Include synthetic augmented images')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--convert-tflite', action='store_true', default=True,
                       help='Convert to TFLite after training')
    
    return parser.parse_args()


def check_gpu_availability():
    """Check and configure GPU if available."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPUs found. Training will be slower on CPU.")
        # CPU optimizations
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)


def load_and_prepare_data(args):
    """Load datasets and prepare for training."""
    logger.info("Loading datasets...")
    
    # Initialize data loader
    loader = MultiDatasetLoader(
        base_data_dir=args.data_dir,
        target_size=(args.image_size, args.image_size)
    )
    
    # Load all available datasets
    all_datasets = loader.load_all_datasets(
        use_cache=args.use_cache,
        plantvillage_subset=args.plantvillage_subset,
        include_augmented=args.use_augmented
    )
    
    if not all_datasets:
        logger.error("No datasets found. Please ensure PlantVillage and/or PlantDoc are in the data directory.")
        return None, None
    
    # Create balanced dataset
    logger.info(f"Creating balanced dataset with {args.samples_per_class} samples per class...")
    X, y = loader.create_balanced_dataset(
        all_datasets, 
        samples_per_class=args.samples_per_class
    )
    
    # Split data
    logger.info("Splitting data into train/val/test sets...")
    splits = loader.prepare_train_val_test_split(X, y)
    
    return splits, loader


def evaluate_model(model, test_data, batch_size=32) -> Dict:
    """Evaluate model on test data."""
    x_test, y_test = test_data
    
    logger.info("\nEvaluating model on test set...")
    
    # Create test generator
    test_gen = tf.keras.utils.Sequence
    
    # Evaluate
    results = model.evaluate(
        x_test, y_test,
        batch_size=batch_size,
        verbose=1
    )
    
    # Create results dictionary
    metric_names = model.metrics_names
    evaluation_results = dict(zip(metric_names, results))
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    for metric, value in evaluation_results.items():
        if 'accuracy' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    return evaluation_results


def main(args):
    """Main training function."""
    logger.info("Starting Robust RGB Plant Disease Detection Model Training")
    logger.info(f"Target: 80%+ validation accuracy")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check GPU
    check_gpu_availability()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Log to file
    file_handler = logging.FileHandler(
        output_path / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load and prepare data
    splits, loader = load_and_prepare_data(args)
    if splits is None:
        return
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Save configuration
    config = {
        'data_config': {
            'samples_per_class': args.samples_per_class,
            'image_size': args.image_size,
            'num_train': len(X_train),
            'num_val': len(X_val),
            'num_test': len(X_test)
        },
        'model_config': {
            'num_classes': y_train.shape[1],
            'input_shape': (args.image_size, args.image_size, 3),
            'dropout_rate': args.dropout_rate,
            'l2_regularization': args.l2_regularization
        },
        'training_config': {
            'batch_size': args.batch_size,
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs': args.stage2_epochs,
            'stage3_epochs': args.stage3_epochs,
            'stage1_lr': args.stage1_lr,
            'stage2_lr': args.stage2_lr,
            'stage3_lr': args.stage3_lr,
            'warmup_epochs': args.warmup_epochs,
            'use_focal_loss': args.use_focal_loss,
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
            'mixup_alpha': args.mixup_alpha,
            'cutmix_alpha': args.cutmix_alpha
        }
    }
    
    with open(output_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {output_path / 'training_config.json'}")
    
    if args.evaluate_only:
        # Load and evaluate existing model
        logger.info("Evaluation mode - loading existing model...")
        from model_robust import RobustDiseaseDetector
        
        model_path = output_path / 'final' / 'model.keras'
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return
            
        model = tf.keras.models.load_model(model_path)
        evaluate_model(model, (X_test, y_test), args.batch_size)
        return
    
    # Initialize trainer
    logger.info("Initializing robust progressive trainer...")
    trainer = RobustProgressiveTrainer(
        config['model_config'],
        config['training_config'],
        output_dir=args.output_dir
    )
    
    # Run progressive training
    logger.info("Starting progressive training...")
    history = trainer.train_progressive(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*50)
    
    test_results = evaluate_model(
        trainer.detector.model,
        (X_test, y_test),
        args.batch_size
    )
    
    # Save test results
    with open(output_path / 'test_results.json', 'w') as f:
        json.dump({
            'test_accuracy': float(test_results.get('accuracy', 0)),
            'test_loss': float(test_results.get('loss', 0)),
            'all_metrics': {k: float(v) for k, v in test_results.items()}
        }, f, indent=2)
    
    # Convert to TFLite if requested
    if args.convert_tflite and trainer.best_val_accuracy >= 0.75:
        logger.info("\nConverting to TensorFlow Lite...")
        converter = TFLiteConverter(
            model_path=str(output_path / 'final' / 'model.keras'),
            output_dir=str(output_path / 'tflite')
        )
        
        # Use a sample from validation data
        representative_data = X_val[:100]
        
        tflite_path = converter.convert_to_tflite(
            quantization_type='int8',
            representative_data=representative_data
        )
        
        if tflite_path:
            converter.evaluate_tflite_model(tflite_path, X_test[:100], y_test[:100])
    
    logger.info("\nTraining complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {trainer.best_val_accuracy:.2%}")
    print(f"Test Accuracy: {test_results.get('accuracy', 0):.2%}")
    print(f"Model saved to: {output_path}")
    
    if trainer.best_val_accuracy >= 0.80:
        print("\n✅ SUCCESS: Achieved 80%+ validation accuracy!")
    else:
        print(f"\n⚠️  Target not reached. Consider:")
        print("   1. Training for more epochs")
        print("   2. Using a GPU for faster training")
        print("   3. Downloading PlantNet dataset for more diversity")
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)