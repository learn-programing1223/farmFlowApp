#!/usr/bin/env python3
"""
Main training script for RGB Universal Plant Disease Detection Model
Achieves 80%+ validation accuracy using progressive training and focal loss
"""

import os
import argparse
import json
import numpy as np
import warnings
# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Protobuf gencode.*')

import tensorflow as tf
from pathlib import Path
from datetime import datetime
import logging
import sys

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import MultiDatasetLoader
from training import ProgressiveTrainer
from model import UniversalDiseaseDetector


def setup_logging(output_dir: str):
    """Sets up logging configuration."""
    log_file = Path(output_dir) / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def check_gpu():
    """Checks GPU availability and configuration."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logging.info(f"Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                logging.info(f"  GPU {i}: {gpu}")
        except RuntimeError as e:
            logging.error(f"GPU configuration error: {e}")
    else:
        logging.warning("No GPUs found. Training will be slower on CPU.")
    
    return len(gpus) > 0


def main(args):
    """Main training pipeline."""
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("Starting RGB Universal Plant Disease Detection Model Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Set memory limit for stability
    if not has_gpu and args.memory_limit:
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('CPU')[0], 
            True
        )
    
    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = MultiDatasetLoader(
        base_data_dir=args.data_dir,
        target_size=(args.image_size, args.image_size)
    )
    
    # Load or prepare data
    if args.use_cached_splits and (Path(args.data_dir) / 'splits').exists():
        logger.info("Loading cached data splits...")
        splits = data_loader.load_splits()
    else:
        # Load all datasets
        logger.info("Loading datasets...")
        all_datasets = data_loader.load_all_datasets(
            use_cache=args.use_dataset_cache,
            plantvillage_subset=args.plantvillage_subset
        )
        
        if not all_datasets:
            logger.error("No datasets found. Please download at least one dataset.")
            return
        
        # Create balanced dataset
        logger.info(f"Creating balanced dataset with {args.samples_per_class} samples per class...")
        X, y = data_loader.create_balanced_dataset(
            all_datasets, 
            samples_per_class=args.samples_per_class
        )
        
        # Split data
        logger.info("Splitting data into train/val/test sets...")
        splits = data_loader.prepare_train_val_test_split(
            X, y,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    # Get data splits
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Model configuration
    model_config = {
        'num_classes': y_train.shape[1],
        'input_shape': (args.image_size, args.image_size, 3),
        'dropout_rate': args.dropout_rate,
        'l2_regularization': args.l2_regularization,
        'class_names': ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew',
                       'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    }
    
    # Training configuration
    training_config = {
        'use_focal_loss': args.use_focal_loss,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'stage1_lr': args.stage1_lr,
        'stage2_lr': args.stage2_lr,
        'stage3_lr': args.stage3_lr,
        'use_mixup': args.use_mixup,
        'mixup_alpha': args.mixup_alpha,
        'cutmix_alpha': args.cutmix_alpha,
        'early_stopping_patience': args.early_stopping_patience,
        'reduce_lr_patience': args.reduce_lr_patience
    }
    
    # Save configurations
    config_path = output_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': model_config,
            'training_config': training_config,
            'data_stats': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'num_classes': model_config['num_classes']
            },
            'args': vars(args)
        }, f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    # Initialize trainer
    logger.info("Initializing progressive trainer...")
    trainer = ProgressiveTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir
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
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate_model(
        test_data=(X_test, y_test),
        batch_size=args.batch_size
    )
    
    logger.info(f"Test Results:")
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Check if we achieved target accuracy
    test_accuracy = test_results.get('accuracy', 0)
    if test_accuracy >= 0.80:
        logger.info(f"✓ Target accuracy achieved! Test accuracy: {test_accuracy:.2%}")
    else:
        logger.warning(f"✗ Target accuracy not achieved. Test accuracy: {test_accuracy:.2%}")
        logger.info("Consider: increasing training data, adjusting hyperparameters, or training longer")
    
    logger.info("Training complete!")
    logger.info(f"Models saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RGB Universal Plant Disease Detection Model"
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Base directory for datasets')
    parser.add_argument('--output-dir', type=str, default='./models/rgb_model',
                       help='Output directory for models and logs')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--samples-per-class', type=int, default=500,
                       help='Samples per class for balanced dataset (default: 500)')
    parser.add_argument('--plantvillage-subset', type=float, default=1.0,
                       help='Fraction of PlantVillage dataset to use (default: 1.0)')
    
    # Split arguments
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio (default: 0.15)')
    
    # Model arguments
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--l2-regularization', type=float, default=0.001,
                       help='L2 regularization factor (default: 0.001)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--stage1-epochs', type=int, default=15,
                       help='Stage 1 epochs (default: 15)')
    parser.add_argument('--stage2-epochs', type=int, default=10,
                       help='Stage 2 epochs (default: 10)')
    parser.add_argument('--stage3-epochs', type=int, default=5,
                       help='Stage 3 epochs (default: 5)')
    
    # Learning rates
    parser.add_argument('--stage1-lr', type=float, default=0.001,
                       help='Stage 1 learning rate (default: 0.001)')
    parser.add_argument('--stage2-lr', type=float, default=0.0001,
                       help='Stage 2 learning rate (default: 0.0001)')
    parser.add_argument('--stage3-lr', type=float, default=0.00001,
                       help='Stage 3 learning rate (default: 0.00001)')
    
    # Loss and augmentation
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use focal loss (default: True)')
    parser.add_argument('--focal-alpha', type=float, default=0.75,
                       help='Focal loss alpha (default: 0.75)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--use-mixup', action='store_true', default=True,
                       help='Use MixUp augmentation (default: True)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='MixUp alpha (default: 0.2)')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                       help='CutMix alpha (default: 1.0)')
    
    # Callbacks
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--reduce-lr-patience', type=int, default=5,
                       help='Reduce LR patience (default: 5)')
    
    # Caching
    parser.add_argument('--use-dataset-cache', action='store_true', default=True,
                       help='Use cached dataset loading (default: True)')
    parser.add_argument('--use-cached-splits', action='store_true', default=False,
                       help='Use previously saved data splits (default: False)')
    
    # System
    parser.add_argument('--memory-limit', type=int, default=None,
                       help='Memory limit in MB (default: None)')
    
    args = parser.parse_args()
    
    # Run training
    main(args)