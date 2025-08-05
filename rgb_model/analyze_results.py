#!/usr/bin/env python3
"""
Analyzes training results and provides recommendations for improving model performance.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_training_results(model_dir: str = './models/rgb_model'):
    """Analyzes the training results and identifies issues."""
    
    model_path = Path(model_dir)
    
    # Load training report
    report_path = model_path / 'training_report.json'
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\n=== TRAINING ANALYSIS ===\n")
        
        # Analyze each stage
        for stage, metrics in report['stages'].items():
            print(f"{stage.upper()}:")
            print(f"  Final Training Accuracy: {metrics['final_accuracy']:.2%}")
            print(f"  Final Validation Accuracy: {metrics['final_val_accuracy']:.2%}")
            print(f"  Best Validation Accuracy: {metrics['best_val_accuracy']:.2%}")
            print(f"  Final Loss: {metrics['final_loss']:.4f}")
            print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
            print()
    
    # Load label mapping to check class distribution
    label_path = Path('./data/label_mapping.json')
    if label_path.exists():
        with open(label_path, 'r') as f:
            label_info = json.load(f)
        
        print("\n=== CLASS DISTRIBUTION ===")
        print(f"Total classes: {label_info['num_classes']}")
        print("Classes:", list(label_info['idx_to_label'].values()))
    
    print("\n=== IDENTIFIED ISSUES ===\n")
    
    issues = []
    
    # Issue 1: Dataset imbalance
    print("1. DATASET IMBALANCE:")
    print("   - Some classes may have significantly fewer samples than others")
    print("   - This can cause the model to be biased towards well-represented classes")
    print("   - Need to balance dataset for optimal performance")
    issues.append("dataset_imbalance")
    
    # Issue 2: Model architecture
    print("\n2. MODEL ARCHITECTURE:")
    print("   - Using fallback CNN instead of EfficientNet")
    print("   - EfficientNet would provide better feature extraction")
    print("   - Current CNN may be too simple for complex disease patterns")
    issues.append("simple_architecture")
    
    # Issue 3: Low initial performance
    print("\n3. TRAINING DYNAMICS:")
    print("   - Initial validation accuracy was very low (~19%)")
    print("   - This suggests the model struggled to generalize initially")
    print("   - Final validation accuracy (~72%) is decent but could be better")
    issues.append("training_dynamics")
    
    # Issue 4: Hardware limitations
    print("\n4. HARDWARE LIMITATIONS:")
    print("   - Training on CPU is extremely slow")
    print("   - GPU would speed up training 10-50x")
    print("   - Slow training limits experimentation")
    issues.append("cpu_training")
    
    print("\n=== RECOMMENDATIONS ===\n")
    
    print("1. IMMEDIATE FIXES:")
    print("   a) Balance dataset using CycleGAN augmentation to 7000 samples per class")
    print("   b) Use MobileNetV3 instead of EfficientNet for better compatibility")
    print("   c) Increase samples_per_class to 7000 for balanced training")
    
    print("\n2. DATA IMPROVEMENTS:")
    print("   a) Download PlantDoc dataset for more pest damage examples")
    print("   b) Use data augmentation more aggressively")
    print("   c) Consider synthetic data generation for rare classes")
    
    print("\n3. MODEL IMPROVEMENTS:")
    print("   a) Try MobileNetV3 or ResNet50 as alternatives to EfficientNet")
    print("   b) Implement class weights to handle imbalance")
    print("   c) Use learning rate warmup in first stage")
    
    print("\n4. TRAINING IMPROVEMENTS:")
    print("   a) Train on Google Colab with free GPU")
    print("   b) Use mixed precision training for faster computation")
    print("   c) Implement gradient accumulation for larger effective batch size")
    
    print("\n5. EVALUATION:")
    print("   a) Use confusion matrix to identify problematic class pairs")
    print("   b) Calculate per-class precision/recall/F1 scores")
    print("   c) Visualize misclassified examples to understand errors")
    
    return issues

def create_improved_config():
    """Creates an improved configuration based on analysis."""
    
    improved_config = {
        "model_config": {
            "num_classes": 7,  # Updated to 7 classes
            "input_shape": [224, 224, 3],
            "dropout_rate": 0.4,  # Slightly lower dropout
            "l2_regularization": 0.0005  # Lower regularization
        },
        "training_config": {
            "samples_per_class": 7000,  # Balanced dataset with augmentation
            "batch_size": 64,  # Larger batch if memory allows
            "stage1_epochs": 10,  # More epochs for stage 1
            "stage2_epochs": 15,
            "stage3_epochs": 10,
            "stage1_lr": 0.0005,  # Lower initial learning rate
            "stage2_lr": 0.00005,
            "stage3_lr": 0.000005,
            "use_class_weights": True,  # Enable class weights
            "augmentation_strength": 1.5,  # Stronger augmentation
            "label_smoothing": 0.1,  # Add label smoothing
            "warmup_epochs": 3  # Add learning rate warmup
        }
    }
    
    # Save improved config
    with open('./rgb_model/improved_config.json', 'w') as f:
        json.dump(improved_config, f, indent=2)
    
    print("\n=== IMPROVED CONFIGURATION SAVED ===")
    print("Location: ./rgb_model/improved_config.json")
    
    return improved_config

def generate_training_curves():
    """Generates training curves from CSV logs."""
    
    import pandas as pd
    
    stages = ['stage1', 'stage2', 'stage3']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, stage in enumerate(stages):
        csv_path = Path(f'./models/rgb_model/{stage}/training_log.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Plot accuracy
            axes[0, i].plot(df['epoch'], df['accuracy'], label='Train')
            axes[0, i].plot(df['epoch'], df['val_accuracy'], label='Val')
            axes[0, i].set_title(f'{stage.upper()} Accuracy')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot loss
            axes[1, i].plot(df['epoch'], df['loss'], label='Train')
            axes[1, i].plot(df['epoch'], df['val_loss'], label='Val')
            axes[1, i].set_title(f'{stage.upper()} Loss')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig('./models/rgb_model/training_curves.png')
    print("\nTraining curves saved to: ./models/rgb_model/training_curves.png")

if __name__ == "__main__":
    print("Analyzing RGB Model Training Results...")
    issues = analyze_training_results()
    
    print("\n" + "="*50)
    improved_config = create_improved_config()
    
    try:
        generate_training_curves()
    except Exception as e:
        print(f"\nCould not generate training curves: {e}")
    
    print("\n=== NEXT STEPS ===")
    print("1. Run: python rgb_model/train_rgb_model.py --data-dir ./data --batch-size 64 --samples-per-class 1000")
    print("2. Consider training on Google Colab for GPU acceleration")
    print("3. Download additional datasets (PlantDoc, PlantNet) for better coverage")
    print("4. Implement the improvements suggested above")