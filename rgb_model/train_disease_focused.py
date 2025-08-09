#!/usr/bin/env python3
"""
Disease-focused training script for 85% accuracy target
Uses ALL available disease images, PlantNet only for healthy balance
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime

from data_loader import MultiDatasetLoader
from model import UniversalDiseaseDetector
from training import ProgressiveTrainer


def check_available_datasets(data_dir='./data'):
    """Check which disease datasets are available"""
    data_dir = Path(data_dir)
    available = {}
    
    # Check PlantVillage
    pv_dir = data_dir / 'PlantVillage'
    if pv_dir.exists():
        # Check for actual disease folders
        disease_folders = [d for d in pv_dir.rglob('*') if d.is_dir() and 'healthy' not in d.name.lower()]
        if disease_folders:
            image_count = sum(1 for _ in pv_dir.rglob('*.jpg'))
            available['PlantVillage'] = {
                'path': str(pv_dir),
                'images': image_count,
                'has_diseases': True
            }
            print(f"✓ PlantVillage: {image_count} images with diseases")
    
    # Check PlantDoc
    pd_dir = data_dir / 'PlantDoc'
    if pd_dir.exists():
        image_count = sum(1 for _ in pd_dir.rglob('*.jpg'))
        if image_count > 0:
            available['PlantDoc'] = {
                'path': str(pd_dir),
                'images': image_count,
                'has_diseases': True
            }
            print(f"✓ PlantDoc: {image_count} images with diseases")
    
    # Check Kaggle
    kg_dir = data_dir / 'KagglePlantPathology'
    if kg_dir.exists() and (kg_dir / 'train_images').exists():
        image_count = sum(1 for _ in kg_dir.rglob('*.jpg'))
        if image_count > 0:
            available['Kaggle'] = {
                'path': str(kg_dir),
                'images': image_count,
                'has_diseases': True
            }
            print(f"✓ Kaggle: {image_count} images with diseases")
    
    # Check PlantNet (only for healthy samples)
    pn_zip = data_dir / 'plantnet_300K.zip'
    if not pn_zip.exists():
        pn_zip = Path(__file__).parent / 'src' / 'data' / 'plantnet_300K.zip'
    
    if pn_zip.exists():
        available['PlantNet'] = {
            'path': str(pn_zip),
            'images': 306146,
            'has_diseases': False  # Only healthy plants
        }
        print(f"✓ PlantNet: 306,146 healthy plant images (for balance)")
    
    return available


def setup_training_data(data_dir='./data', use_plantnet_for_balance=True, max_healthy_ratio=0.3):
    """
    Setup training data with focus on disease images
    
    Args:
        data_dir: Base data directory
        use_plantnet_for_balance: Use PlantNet to balance healthy class
        max_healthy_ratio: Maximum ratio of healthy samples (0.3 = 30% healthy)
    """
    
    print("\n" + "="*60)
    print("Loading Disease Datasets")
    print("="*60)
    
    loader = MultiDatasetLoader(base_data_dir=data_dir)
    
    # Load all disease datasets
    all_datasets = {}
    
    # Load PlantVillage (main disease dataset)
    print("\nLoading PlantVillage (54,306 disease images)...")
    pv_images, pv_labels = loader.load_plantvillage(subset_percent=1.0)  # Use ALL images
    if pv_images:
        all_datasets['PlantVillage'] = (pv_images, pv_labels)
        print(f"  Loaded: {len(pv_images)} images")
        
        # Count disease distribution
        from collections import Counter
        disease_counts = Counter(pv_labels)
        print("  Disease distribution:")
        for disease, count in disease_counts.most_common():
            print(f"    {disease}: {count}")
    
    # Load PlantDoc
    print("\nLoading PlantDoc (2,598 annotated images)...")
    pd_images, pd_labels = loader.load_plantdoc()
    if pd_images:
        all_datasets['PlantDoc'] = (pd_images, pd_labels)
        print(f"  Loaded: {len(pd_images)} images")
    
    # Load Kaggle
    print("\nLoading Kaggle Plant Pathology...")
    kg_images, kg_labels = loader.load_kaggle_plant_pathology()
    if kg_images:
        all_datasets['Kaggle'] = (kg_images, kg_labels)
        print(f"  Loaded: {len(kg_images)} images")
    
    # Calculate current healthy ratio
    all_labels = []
    for dataset_name, (images, labels) in all_datasets.items():
        all_labels.extend(labels)
    
    healthy_count = sum(1 for label in all_labels if label.lower() == 'healthy')
    disease_count = len(all_labels) - healthy_count
    current_healthy_ratio = healthy_count / max(len(all_labels), 1)
    
    print(f"\nCurrent distribution:")
    print(f"  Disease images: {disease_count}")
    print(f"  Healthy images: {healthy_count}")
    print(f"  Healthy ratio: {current_healthy_ratio:.1%}")
    
    # Add PlantNet for healthy balance if needed
    if use_plantnet_for_balance and current_healthy_ratio < max_healthy_ratio:
        needed_healthy = int(disease_count * max_healthy_ratio / (1 - max_healthy_ratio)) - healthy_count
        needed_healthy = max(0, needed_healthy)
        
        if needed_healthy > 0:
            print(f"\nAdding {needed_healthy} PlantNet images for healthy balance...")
            pn_images, pn_labels = loader.load_plantnet(max_samples=needed_healthy)
            if pn_images:
                all_datasets['PlantNet_Balance'] = (pn_images, pn_labels)
                print(f"  Added: {len(pn_images)} healthy images from PlantNet")
    
    if not all_datasets:
        raise ValueError("No datasets found! Please run setup_all_disease_datasets.py first")
    
    return loader, all_datasets


def train_disease_model(args):
    """Main training function focused on disease detection"""
    
    print("="*60)
    print("Disease-Focused RGB Model Training")
    print("Target: 85% Accuracy on Universal Disease Detection")
    print("="*60)
    
    # Check available datasets
    available = check_available_datasets(args.data_dir)
    
    if not available:
        print("\n❌ No datasets found!")
        print("Please run: python setup_all_disease_datasets.py")
        return
    
    total_images = sum(d['images'] for d in available.values())
    disease_images = sum(d['images'] for d in available.values() if d['has_diseases'])
    
    print(f"\nTotal images available: {total_images:,}")
    print(f"Disease images: {disease_images:,}")
    
    if disease_images < 10000:
        print("\n⚠️  Warning: Limited disease data. Download more datasets for 85% accuracy.")
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n✓ Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model_config': {
                'num_classes': 7,
                'input_shape': [224, 224, 3],
                'dropout_rate': 0.4,
                'l2_regularization': 0.0005
            },
            'training_config': {
                'samples_per_class': args.samples_per_class,
                'batch_size': args.batch_size,
                'use_focal_loss': True,
                'focal_alpha': 0.75,
                'focal_gamma': 2.0,
                'use_mixup': True,
                'use_cutmix': True
            }
        }
    
    # Setup training data
    print("\nPreparing training data...")
    loader, all_datasets = setup_training_data(
        args.data_dir,
        use_plantnet_for_balance=not args.no_plantnet,
        max_healthy_ratio=args.healthy_ratio
    )
    
    # Create balanced dataset with maximum samples
    print(f"\nCreating balanced dataset ({args.samples_per_class} samples per class)...")
    X, y = loader.create_balanced_dataset(
        all_datasets, 
        samples_per_class=args.samples_per_class
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {y.shape[1]}")
    
    # Check class distribution
    class_sums = y.sum(axis=0)
    print("\nClass distribution:")
    categories = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew', 
                 'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    for i, (cat, count) in enumerate(zip(categories, class_sums)):
        print(f"  {cat}: {int(count)} samples")
    
    # Split data with leakage prevention
    print("\nSplitting data (with leakage prevention)...")
    splits = loader.prepare_train_val_test_split(
        X, y,
        val_split=0.15,
        test_split=0.15,
        ensure_no_leakage=True
    )
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    print("\nInitializing progressive trainer...")
    trainer = ProgressiveTrainer(
        model_config=config['model_config'],
        training_config=config['training_config'],
        output_dir=str(output_dir)
    )
    
    # Progressive training
    print("\n" + "="*50)
    print("Starting Progressive Training")
    print("="*50)
    
    # Stage 1: Feature extraction
    history_stage1 = trainer.train_stage_1(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=args.stage1_epochs,
        batch_size=args.batch_size
    )
    
    if not args.stage1_only:
        # Stage 2: Partial fine-tuning
        history_stage2 = trainer.train_stage_2(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            num_layers_to_unfreeze=20
        )
        
        if args.full_training:
            # Stage 3: Full fine-tuning
            history_stage3 = trainer.train_stage_3(
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                epochs=args.stage3_epochs,
                batch_size=args.batch_size
            )
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    final_model = trainer.detector.model
    
    # Test evaluation
    test_results = final_model.evaluate(X_test, y_test, verbose=1)
    test_loss, test_acc, test_precision, test_recall, test_auc = test_results
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall: {test_recall:.3f}")
    print(f"  AUC: {test_auc:.3f}")
    print(f"  Loss: {test_loss:.3f}")
    
    # Per-class evaluation
    y_pred = final_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n📈 Per-Class Accuracy:")
    for i, cat in enumerate(categories):
        mask = y_true_classes == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"  {cat}: {class_acc:.3f} ({class_acc*100:.1f}%)")
    
    # Save model
    model_path = output_dir / 'final_model.keras'
    trainer.detector.save_model(str(model_path))
    print(f"\n💾 Model saved to {model_path}")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'datasets_used': list(all_datasets.keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Success check
    print("\n" + "="*60)
    if test_acc >= 0.85:
        print("🎉 SUCCESS! Achieved 85%+ accuracy target!")
    elif test_acc >= 0.80:
        print("✅ Good! Achieved 80%+ accuracy. Close to target!")
        print("   Consider: More epochs, more data, or hyperparameter tuning")
    else:
        print(f"📈 Current accuracy: {test_acc*100:.1f}%")
        print("   Recommendations:")
        print("   - Download more disease datasets")
        print("   - Increase samples_per_class")
        print("   - Try more training epochs")
        print("   - Adjust augmentation parameters")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Train disease-focused RGB model for 85% accuracy'
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Base data directory')
    parser.add_argument('--samples-per-class', type=int, default=5000,
                       help='Maximum samples per class (default: 5000)')
    parser.add_argument('--healthy-ratio', type=float, default=0.3,
                       help='Maximum ratio of healthy samples (default: 0.3)')
    parser.add_argument('--no-plantnet', action='store_true',
                       help='Do not use PlantNet for healthy balance')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--stage1-epochs', type=int, default=15,
                       help='Stage 1 epochs')
    parser.add_argument('--stage2-epochs', type=int, default=10,
                       help='Stage 2 epochs')
    parser.add_argument('--stage3-epochs', type=int, default=5,
                       help='Stage 3 epochs')
    
    # Training control
    parser.add_argument('--stage1-only', action='store_true',
                       help='Only train stage 1')
    parser.add_argument('--full-training', action='store_true',
                       help='Include stage 3 full fine-tuning')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                       default='./models/disease_focused',
                       help='Output directory')
    parser.add_argument('--config', type=str,
                       default='./improved_config.json',
                       help='Config file')
    
    args = parser.parse_args()
    
    # Run training
    train_disease_model(args)


if __name__ == "__main__":
    main()