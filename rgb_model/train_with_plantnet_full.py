#!/usr/bin/env python3
"""
Training script optimized for using FULL PlantNet-300K dataset
Uses streaming to handle 31.7GB dataset without memory overflow
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
from pathlib import Path
import numpy as np
from datetime import datetime

from data_loader_streaming import CombinedStreamingLoader
from model import UniversalDiseaseDetector
from training import ProgressiveTrainer


def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"Found {len(gpus)} GPU(s)")
            
            # Set mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled for faster training")
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found - training will be slower")
    
    return len(gpus) > 0


def create_callbacks(output_dir, patience=10):
    """Create training callbacks with early stopping"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Model checkpoint - save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch'
        ),
        
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            str(output_dir / 'training_history.csv')
        )
    ]
    
    return callbacks


def train_with_plantnet_full(args):
    """Main training function using full PlantNet dataset"""
    
    print("=" * 60)
    print("RGB Universal Disease Detector Training")
    print("With FULL PlantNet-300K Dataset")
    print("=" * 60)
    
    # Setup
    has_gpu = setup_gpu()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        # Default config
        config = {
            'model_config': {
                'num_classes': 7,
                'input_shape': [224, 224, 3],
                'dropout_rate': 0.4,
                'l2_regularization': 0.0005
            },
            'training_config': {
                'use_focal_loss': True,
                'focal_alpha': 0.75,
                'focal_gamma': 2.0,
                'use_mixup': True,
                'mixup_alpha': 0.2,
                'use_cutmix': True,
                'cutmix_alpha': 1.0
            }
        }
    
    # Create data loader with streaming
    print("\nInitializing data loaders...")
    loader = CombinedStreamingLoader(
        data_dir=args.data_dir,
        use_plantnet_full=True,
        plantnet_samples=args.plantnet_samples  # None for all ~300K images
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_data, test_data = loader.prepare_mixed_training_data()
    
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    print(f"\nDataset sizes:")
    print(f"  Training: Streaming (Regular + {args.plantnet_samples or 'ALL'} PlantNet images)")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Build model
    print("\nBuilding model...")
    detector = UniversalDiseaseDetector(
        num_classes=config['model_config']['num_classes'],
        input_shape=tuple(config['model_config']['input_shape']),
        dropout_rate=config['model_config']['dropout_rate'],
        l2_regularization=config['model_config']['l2_regularization']
    )
    
    # Progressive training approach
    print("\nStarting progressive training...")
    print("Early stopping will prevent overfitting automatically")
    
    # Stage 1: Feature extraction (frozen backbone)
    print("\n" + "=" * 50)
    print("STAGE 1: Feature Extraction")
    print("=" * 50)
    
    detector.compile_model(
        learning_rate=args.stage1_lr,
        use_focal_loss=config['training_config']['use_focal_loss'],
        focal_alpha=config['training_config']['focal_alpha'],
        focal_gamma=config['training_config']['focal_gamma']
    )
    
    # Estimate steps
    steps_per_epoch = loader.estimate_steps_per_epoch()
    val_steps = len(X_val) // args.batch_size
    
    print(f"Steps per epoch: ~{steps_per_epoch}")
    
    # Train Stage 1
    history_stage1 = detector.model.fit(
        train_dataset,
        epochs=args.stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        callbacks=create_callbacks(output_dir / 'stage1', args.patience),
        verbose=1
    )
    
    # Stage 2: Partial fine-tuning
    if not args.stage1_only:
        print("\n" + "=" * 50)
        print("STAGE 2: Partial Fine-tuning")
        print("=" * 50)
        
        # Unfreeze top layers
        detector.unfreeze_layers(20)
        
        # Recompile with lower learning rate
        detector.compile_model(
            learning_rate=args.stage2_lr,
            use_focal_loss=config['training_config']['use_focal_loss'],
            focal_alpha=config['training_config']['focal_alpha'],
            focal_gamma=config['training_config']['focal_gamma']
        )
        
        # Train Stage 2
        history_stage2 = detector.model.fit(
            train_dataset,
            epochs=args.stage2_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            callbacks=create_callbacks(output_dir / 'stage2', args.patience),
            verbose=1
        )
        
        # Stage 3: Full fine-tuning
        if args.full_training:
            print("\n" + "=" * 50)
            print("STAGE 3: Full Fine-tuning")
            print("=" * 50)
            
            # Unfreeze all layers
            detector.unfreeze_layers(-1)
            
            # Recompile with very low learning rate
            detector.compile_model(
                learning_rate=args.stage3_lr,
                use_focal_loss=config['training_config']['use_focal_loss'],
                focal_alpha=config['training_config']['focal_alpha'],
                focal_gamma=config['training_config']['focal_gamma']
            )
            
            # Train Stage 3
            history_stage3 = detector.model.fit(
                train_dataset,
                epochs=args.stage3_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=(X_val, y_val),
                callbacks=create_callbacks(output_dir / 'stage3', args.patience // 2),
                verbose=1
            )
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    # Test set evaluation
    test_loss, test_acc, test_precision, test_recall, test_auc = detector.model.evaluate(
        X_test, y_test, verbose=1
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Save final model
    final_path = output_dir / 'final_model.keras'
    detector.save_model(str(final_path))
    print(f"\nFinal model saved to {final_path}")
    
    # Save test results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'plantnet_samples': args.plantnet_samples or 'all',
        'training_completed': datetime.now().isoformat()
    }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    if test_acc >= 0.80:
        print("🎉 Target accuracy of 80% achieved!")
    else:
        print(f"Current accuracy: {test_acc:.1%}")
        print("Consider:")
        print("- Using more PlantNet samples")
        print("- Adjusting augmentation parameters")
        print("- Increasing training epochs")


def main():
    parser = argparse.ArgumentParser(
        description='Train RGB model with full PlantNet dataset'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Base data directory')
    parser.add_argument('--plantnet-samples', type=int, default=None,
                       help='Number of PlantNet samples (None for all ~300K)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--stage1-epochs', type=int, default=10,
                       help='Stage 1 epochs')
    parser.add_argument('--stage2-epochs', type=int, default=10,
                       help='Stage 2 epochs')
    parser.add_argument('--stage3-epochs', type=int, default=5,
                       help='Stage 3 epochs')
    
    # Learning rates
    parser.add_argument('--stage1-lr', type=float, default=0.001,
                       help='Stage 1 learning rate')
    parser.add_argument('--stage2-lr', type=float, default=0.0001,
                       help='Stage 2 learning rate')
    parser.add_argument('--stage3-lr', type=float, default=0.00001,
                       help='Stage 3 learning rate')
    
    # Training control
    parser.add_argument('--stage1-only', action='store_true',
                       help='Only train stage 1')
    parser.add_argument('--full-training', action='store_true',
                       help='Include stage 3 full fine-tuning')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                       default='./models/plantnet_full',
                       help='Output directory')
    parser.add_argument('--config', type=str,
                       default='./improved_config.json',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Run training
    train_with_plantnet_full(args)


if __name__ == "__main__":
    main()