#!/usr/bin/env python3
"""
Train using already processed data without reprocessing images
This avoids using additional disk space
"""

import os
import sys
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from training_robust import RobustProgressiveTrainer


def load_existing_processed_data():
    """Load the already processed data from splits directory"""
    print("Loading existing processed data...")
    
    # Check for compressed format first
    splits_dir = Path('./data/splits')
    splits = {}
    
    if splits_dir.exists():
        for split_name in ['train', 'val', 'test']:
            # Try compressed format first
            compressed_path = splits_dir / f'{split_name}_data.npz'
            if compressed_path.exists():
                print(f"Loading {split_name} from compressed format...")
                data = np.load(compressed_path)
                splits[split_name] = (data['X'], data['y'])
            else:
                # Try regular format
                X_path = splits_dir / f'X_{split_name}.npy'
                y_path = splits_dir / f'y_{split_name}.npy'
                if X_path.exists() and y_path.exists():
                    print(f"Loading {split_name} from regular format...")
                    X = np.load(X_path)
                    y = np.load(y_path)
                    splits[split_name] = (X, y)
                else:
                    # Try to find any numpy files
                    print(f"Could not find {split_name} data in standard locations")
    
    # If no splits found, try to load from the last successful run
    if not splits:
        print("\nNo splits found. Checking for cached processed data...")
        
        # Look for any .npy files in data directory
        data_files = list(Path('./data').glob('X_*.npy'))
        if data_files:
            print(f"Found {len(data_files)} data files")
            # Load the most recent ones
            X = np.load('./data/X_processed.npy')
            y = np.load('./data/y_processed.npy')
            
            # Create splits manually
            from sklearn.model_selection import train_test_split
            
            # First split: 70% train, 30% temp
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
            )
            
            # Second split: 50% val, 50% test from temp (15% each of total)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
            )
            
            splits = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test)
            }
            
            print(f"Created splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    if not splits:
        raise ValueError("No processed data found! Please run data processing first.")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Train with existing processed data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--stage1-epochs', type=int, default=15, help='Stage 1 epochs')
    parser.add_argument('--stage2-epochs', type=int, default=20, help='Stage 2 epochs')
    parser.add_argument('--stage3-epochs', type=int, default=10, help='Stage 3 epochs')
    parser.add_argument('--output-dir', type=str, default='./models/rgb_robust', help='Output directory')
    parser.add_argument('--stage1-lr', type=float, default=0.001, help='Stage 1 learning rate')
    parser.add_argument('--stage2-lr', type=float, default=0.0001, help='Stage 2 learning rate')
    parser.add_argument('--stage3-lr', type=float, default=0.00001, help='Stage 3 learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("TRAINING WITH EXISTING PROCESSED DATA")
    print("="*50)
    
    # Load existing data
    try:
        splits = load_existing_processed_data()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTrying alternative: Looking for the balanced dataset that was just created...")
        
        # The balanced dataset should be in memory or recently saved
        # Let's check for the most recent files
        import glob
        recent_files = glob.glob('./data/*.npy')
        if recent_files:
            print(f"Found {len(recent_files)} numpy files")
            for f in recent_files:
                print(f"  - {f}")
        
        # If we can't find the data, provide instructions
        print("\nTo avoid reprocessing, you need to:")
        print("1. Find where the balanced 7000-sample dataset was saved")
        print("2. Load it directly without reprocessing")
        print("\nAlternatively, check if the data is in:")
        print("  - ./data/splits/")
        print("  - ./data/cache/")
        print("  - ./data/augmented/")
        return
    
    # Get the splits
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"\nLoaded data successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {y_train.shape[1]}")
    
    # Model configuration
    model_config = {
        'num_classes': y_train.shape[1],
        'input_shape': (224, 224, 3)
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
    print("\nStarting training...")
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
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()