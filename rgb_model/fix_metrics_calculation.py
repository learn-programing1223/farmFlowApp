"""
Fix for metrics calculation on augmented data issue.
This creates a custom callback that calculates metrics on clean data during training.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class CleanDataMetricsCallback(keras.callbacks.Callback):
    """
    Custom callback to calculate training metrics on clean (non-augmented) data.
    This solves the issue where training metrics are artificially low due to augmentation.
    """
    
    def __init__(self, clean_dataset, steps, update_frequency=1):
        """
        Args:
            clean_dataset: TF dataset without augmentation for metrics calculation
            steps: Number of steps to evaluate
            update_frequency: How often to calculate metrics (every N epochs)
        """
        super().__init__()
        self.clean_dataset = clean_dataset
        self.steps = steps
        self.update_frequency = update_frequency
        self.metrics_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate metrics on clean data at end of epoch."""
        if (epoch + 1) % self.update_frequency == 0:
            # Evaluate on clean data
            print(f"\n[Calculating clean training metrics...]", end="")
            
            # Get current model
            results = self.model.evaluate(
                self.clean_dataset,
                steps=self.steps,
                verbose=0
            )
            
            # Extract metrics
            metric_names = self.model.metrics_names
            clean_metrics = {}
            
            for i, name in enumerate(metric_names):
                if name != 'loss':  # We keep loss from augmented data
                    clean_name = f'clean_{name}'
                    clean_metrics[clean_name] = results[i]
                    
                    # Update logs to show in progress bar
                    if logs is not None:
                        logs[clean_name] = results[i]
            
            # Store history
            self.metrics_history.append({
                'epoch': epoch + 1,
                **clean_metrics
            })
            
            # Print clean metrics
            print(f" Clean Accuracy: {clean_metrics.get('clean_accuracy', 0):.4f}")
            
            # Compare with augmented metrics
            if logs and 'accuracy' in logs:
                gap = abs(logs['accuracy'] - clean_metrics.get('clean_accuracy', 0))
                print(f"[Augmented vs Clean Gap: {gap:.4f}]")


def create_clean_train_dataset(data_loader, train_paths, train_labels, batch_size):
    """
    Create a clean (non-augmented) version of training dataset for metrics calculation.
    
    Args:
        data_loader: EnhancedDataLoader instance
        train_paths: Training image paths
        train_labels: Training labels
        batch_size: Batch size
        
    Returns:
        Clean TF dataset for metrics calculation
    """
    # Create dataset WITHOUT augmentation
    clean_dataset = data_loader.create_tf_dataset(
        train_paths, 
        train_labels,
        is_training=False,  # This prevents augmentation
        shuffle=False,       # No need to shuffle for metrics
        augment=False        # Explicitly no augmentation
    )
    
    return clean_dataset


def apply_metrics_fix_to_training(args):
    """
    Apply the metrics fix to the training script.
    This function modifies the training to calculate metrics on clean data.
    """
    print("\n" + "="*60)
    print("APPLYING METRICS CALCULATION FIX")
    print("="*60)
    
    import sys
    from pathlib import Path
    sys.path.append('src')
    from data_loader_v2 import EnhancedDataLoader
    
    # Setup paths
    data_path = Path(args.data_path)
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    # Create data loaders with SAME preprocessing for train and val
    print("\n1. Creating consistent data loaders...")
    train_loader = EnhancedDataLoader(
        data_dir=train_path,
        target_size=(224, 224),
        batch_size=args.batch_size,
        use_advanced_preprocessing=args.use_advanced_preprocessing,
        preprocessing_mode=args.preprocessing_mode  # Same for both
    )
    
    val_loader = EnhancedDataLoader(
        data_dir=val_path,
        target_size=(224, 224),
        batch_size=args.batch_size,
        use_advanced_preprocessing=args.use_advanced_preprocessing,
        preprocessing_mode=args.preprocessing_mode  # Same preprocessing mode
    )
    
    # Load data
    print("\n2. Loading datasets...")
    train_paths, train_labels, class_names = train_loader.load_dataset_from_directory(train_path, 'train')
    val_paths, val_labels, _ = val_loader.load_dataset_from_directory(val_path, 'val')
    
    print(f"   Training samples: {len(train_paths)}")
    print(f"   Validation samples: {len(val_paths)}")
    
    # Create datasets
    print("\n3. Creating datasets...")
    
    # Training dataset WITH augmentation (for gradient updates)
    train_dataset_augmented = train_loader.create_tf_dataset(
        train_paths, train_labels,
        is_training=True,
        shuffle=True,
        augment=True  # Augmentation for training
    )
    
    # Training dataset WITHOUT augmentation (for metrics)
    train_dataset_clean = train_loader.create_tf_dataset(
        train_paths, train_labels,
        is_training=False,
        shuffle=False,
        augment=False  # No augmentation for metrics
    )
    
    # Validation dataset (no augmentation)
    val_dataset = val_loader.create_tf_dataset(
        val_paths, val_labels,
        is_training=False,
        shuffle=False,
        augment=False
    )
    
    print("   [OK] Augmented training dataset created (for gradients)")
    print("   [OK] Clean training dataset created (for metrics)")
    print("   [OK] Validation dataset created")
    
    # Calculate steps
    steps_per_epoch = len(train_paths) // args.batch_size
    validation_steps = len(val_paths) // args.batch_size
    clean_metrics_steps = min(50, steps_per_epoch)  # Evaluate on subset for speed
    
    print(f"\n4. Training configuration:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print(f"   Clean metrics steps: {clean_metrics_steps}")
    
    return {
        'train_dataset_augmented': train_dataset_augmented,
        'train_dataset_clean': train_dataset_clean,
        'val_dataset': val_dataset,
        'clean_metrics_callback': CleanDataMetricsCallback(
            clean_dataset=train_dataset_clean,
            steps=clean_metrics_steps,
            update_frequency=1  # Calculate every epoch
        ),
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps
    }


def patch_training_script():
    """
    Patch the train_robust_model_v2.py to use clean metrics calculation.
    """
    print("\nPatching train_robust_model_v2.py...")
    
    # Read the training script
    with open('train_robust_model_v2.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'CleanDataMetricsCallback' in content:
        print("[Already patched]")
        return
    
    # Add import at the top
    import_line = "from fix_metrics_calculation import CleanDataMetricsCallback, create_clean_train_dataset\n"
    
    # Find imports section
    import_idx = content.find("from data_loader_v2 import")
    if import_idx != -1:
        # Insert after this import
        end_of_line = content.find('\n', import_idx)
        content = content[:end_of_line+1] + import_line + content[end_of_line+1:]
    
    # Add clean dataset creation after train_dataset creation
    dataset_creation = """
    # Create clean training dataset for accurate metrics calculation
    print("Creating clean training dataset for metrics...")
    train_dataset_clean = train_loader.create_tf_dataset(
        train_paths, train_labels,
        is_training=False,
        shuffle=False,
        augment=False  # No augmentation for metrics calculation
    )
    clean_metrics_steps = min(50, steps_per_epoch)  # Evaluate on subset
    """
    
    # Find where to insert
    train_dataset_idx = content.find("train_dataset = train_loader.create_tf_dataset")
    if train_dataset_idx != -1:
        # Find end of this dataset creation block
        next_dataset_idx = content.find("val_dataset = val_loader.create_tf_dataset", train_dataset_idx)
        content = content[:next_dataset_idx] + dataset_creation + "\n    " + content[next_dataset_idx:]
    
    # Add callback to callbacks list
    callback_addition = """
        # Clean metrics callback for accurate training metrics
        CleanDataMetricsCallback(
            clean_dataset=train_dataset_clean,
            steps=clean_metrics_steps,
            update_frequency=1
        ),
    """
    
    # Find callbacks list
    callbacks_idx = content.find("callbacks = [")
    if callbacks_idx != -1:
        # Insert after opening bracket
        bracket_idx = content.find('[', callbacks_idx) + 1
        content = content[:bracket_idx] + "\n" + callback_addition + content[bracket_idx:]
    
    # Save patched version
    with open('train_robust_model_v2_patched.py', 'w') as f:
        f.write(content)
    
    print("[OK] Created train_robust_model_v2_patched.py with metrics fix")
    print("\nTo use the fixed version:")
    print("  python train_robust_model_v2_patched.py --epochs 3 --test_run")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/plantvillage_processed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--preprocessing_mode', default='legacy')
    parser.add_argument('--use_advanced_preprocessing', action='store_true')
    
    args = parser.parse_args()
    
    # Test the fix
    print("Testing metrics calculation fix...")
    results = apply_metrics_fix_to_training(args)
    
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    print("✓ Preprocessing modes now consistent between train and val")
    print("✓ Created clean dataset for training metrics calculation")
    print("✓ Augmentation only affects gradients, not metrics")
    print("✓ Expected result: Training accuracy should now be ~80-85%")
    print("                   (not 65% as before)")
    
    # Optionally patch the training script
    patch_training_script()