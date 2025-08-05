#!/usr/bin/env python3
"""
Alternative training script that doesn't save large arrays to disk
Use this if you have disk space issues
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training_robust import RobustProgressiveTrainer

# Import the main training script functions
from train_robust_model import setup_args, check_gpu, load_and_prepare_data

def main(args):
    """Main training function without disk caching"""
    
    # Check GPU
    check_gpu()
    
    print("\n" + "="*50)
    print("TRAINING WITHOUT DISK CACHE")
    print("="*50)
    
    # Temporarily disable cache saving
    original_save_splits = None
    try:
        from data_loader import MultiDatasetLoader
        original_save_splits = MultiDatasetLoader._save_splits
        # Replace with no-op function
        MultiDatasetLoader._save_splits = lambda self, splits: print("\nSkipping disk save due to space constraints...")
    except:
        pass
    
    # Load and prepare data (without saving to disk)
    splits, loader = load_and_prepare_data(args)
    
    # Get the splits
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Model configuration
    model_config = {
        'num_classes': y_train.shape[1],
        'input_shape': (args.image_size, args.image_size, 3)
    }
    
    # Training configuration
    training_config = {
        'use_focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'stage1_lr': args.stage1_lr,
        'stage2_lr': args.stage2_lr,
        'stage3_lr': args.stage3_lr,
        'warmup_epochs': 3,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'early_stopping_patience': args.early_stopping_patience,
        'reduce_lr_patience': 5
    }
    
    # Create trainer
    trainer = RobustProgressiveTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir
    )
    
    # Run progressive training
    trainer.train_progressive(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    evaluation_results = trainer.evaluate_model(
        test_data=(X_test, y_test),
        batch_size=args.batch_size
    )
    
    print("\nTest Set Results:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Restore original function if it was replaced
    if original_save_splits:
        MultiDatasetLoader._save_splits = original_save_splits
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    args = setup_args()
    main(args)