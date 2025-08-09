#!/usr/bin/env python3
"""
Memory-efficient training script that uses cached data and batch processing
No re-downloading or reprocessing needed!
"""

import os
import sys
import gc
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

from data_loader import MultiDatasetLoader
from model import UniversalDiseaseDetector
from training import ProgressiveTrainer


def free_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()


def load_cached_or_process(data_dir='./data', samples_per_class=5000):
    """Load from cache if available, otherwise process and save"""
    
    cache_dir = Path(data_dir) / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    # Check for cached processed data
    cache_file = cache_dir / f'processed_{samples_per_class}.npz'
    
    if cache_file.exists():
        print(f"✅ Loading cached processed data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['labels'].item()
    
    print("Processing data (this will be cached for next time)...")
    
    # Load datasets with caching
    loader = MultiDatasetLoader(base_data_dir=data_dir)
    
    # Load datasets (uses cache if available)
    all_datasets = loader.load_all_datasets(
        use_cache=True,  # This uses cached file lists
        plantvillage_subset=1.0,  # Use all PlantVillage
        include_augmented=False
    )
    
    # Add limited PlantNet for balance
    if 'PlantNet' not in all_datasets:
        pn_images, pn_labels = loader.load_plantnet(max_samples=samples_per_class // 2)
        if pn_images:
            all_datasets['PlantNet_Balance'] = (pn_images, pn_labels)
    
    # Create balanced dataset
    X, y = loader.create_balanced_dataset(all_datasets, samples_per_class)
    
    # Get label mapping
    label_path = Path(data_dir) / 'label_mapping.json'
    if label_path.exists():
        with open(label_path, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    
    # Save processed data for next time
    print(f"💾 Caching processed data to {cache_file}")
    np.savez_compressed(cache_file, X=X, y=y, labels=labels)
    
    return X, y, labels


def create_data_splits_memory_efficient(X, y, val_split=0.15, test_split=0.15):
    """Create train/val/test splits without memory duplication"""
    
    print("\nCreating data splits (memory-efficient)...")
    
    # Use indices instead of copying arrays
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Get class labels for stratification
    y_classes = np.argmax(y, axis=1)
    
    # First split: separate test set
    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=test_split,
        random_state=42,
        stratify=y_classes
    )
    
    # Second split: separate train and validation
    y_trainval = y_classes[idx_trainval]
    val_size_adjusted = val_split / (1 - test_split)
    
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=y_trainval
    )
    
    print(f"✅ Split sizes - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    
    # Return indices instead of arrays to save memory
    return idx_train, idx_val, idx_test


def train_with_generators(model, X, y, idx_train, idx_val, idx_test, config, args):
    """Train using data generators to avoid memory issues"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data generators that load batches on-the-fly
    def create_generator(X, y, indices, batch_size=32, shuffle=True):
        """Generator that yields batches without loading all data"""
        def generator():
            batch_indices = indices.copy()
            
            while True:
                if shuffle:
                    np.random.shuffle(batch_indices)
                
                for i in range(0, len(batch_indices), batch_size):
                    batch_idx = batch_indices[i:i+batch_size]
                    
                    # Load only the batch we need
                    batch_x = X[batch_idx]
                    batch_y = y[batch_idx]
                    
                    # Apply augmentation here if needed
                    if shuffle and np.random.random() < 0.5:
                        # Simple augmentation
                        batch_x = batch_x + np.random.normal(0, 0.01, batch_x.shape)
                        batch_x = np.clip(batch_x, 0, 1)
                    
                    yield batch_x, batch_y
        
        return generator()
    
    # Create generators
    train_gen = create_generator(X, y, idx_train, args.batch_size, shuffle=True)
    val_gen = create_generator(X, y, idx_val, args.batch_size, shuffle=False)
    
    # Calculate steps
    steps_per_epoch = len(idx_train) // args.batch_size
    validation_steps = len(idx_val) // args.batch_size
    
    print(f"\nTraining with {steps_per_epoch} steps per epoch")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / 'training.csv')
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Test evaluation (batch-wise to save memory)
    print("\n📊 Evaluating on test set...")
    test_gen = create_generator(X, y, idx_test, args.batch_size, shuffle=False)
    test_steps = len(idx_test) // args.batch_size
    
    test_results = model.evaluate(test_gen, steps=test_steps, verbose=1)
    
    return history, test_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Memory-efficient training')
    
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--samples-per-class', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=16)  # Smaller batch for memory
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default='./models/efficient')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Memory-Efficient RGB Model Training")
    print("Using cached data - no re-downloading needed!")
    print("="*60)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured for memory growth")
        except:
            pass
    
    # Load or process data (uses cache)
    X, y, labels = load_cached_or_process(args.data_dir, args.samples_per_class)
    print(f"\n📊 Data shape: {X.shape}")
    
    # Create splits (memory-efficient)
    idx_train, idx_val, idx_test = create_data_splits_memory_efficient(X, y)
    
    # Free some memory
    free_memory()
    
    # Build model
    print("\n🔨 Building model...")
    detector = UniversalDiseaseDetector(
        num_classes=7,
        input_shape=(224, 224, 3),
        dropout_rate=0.4,
        l2_regularization=0.0005
    )
    
    detector.compile_model(
        learning_rate=args.learning_rate,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0
    )
    
    # Train with generators (memory-efficient)
    print("\n🚀 Starting training...")
    history, test_results = train_with_generators(
        detector.model, X, y,
        idx_train, idx_val, idx_test,
        {}, args
    )
    
    # Print results
    test_loss, test_acc, test_precision, test_recall, test_auc = test_results
    
    print("\n" + "="*60)
    print("📈 Final Results:")
    print(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"  Test Precision: {test_precision:.3f}")
    print(f"  Test Recall: {test_recall:.3f}")
    print(f"  Test AUC: {test_auc:.3f}")
    
    if test_acc >= 0.85:
        print("\n🎉 SUCCESS! Achieved 85%+ accuracy!")
    elif test_acc >= 0.80:
        print("\n✅ Good! Achieved 80%+ accuracy!")
    else:
        print(f"\n📊 Current accuracy: {test_acc*100:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()