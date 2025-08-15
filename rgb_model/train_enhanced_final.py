#!/usr/bin/env python3
"""
Final Enhanced Training Script
===============================

Production training with all enhancements for maximum real-world performance.

Configurations:
- Enhanced preprocessing with CLAHE
- Combined loss (Focal + Label Smoothing)
- Stochastic Weight Averaging
- Full augmentation pipeline
- Gradient clipping

Author: PlantPulse Team
Date: 2025
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def run_training(epochs=50, test_run=False):
    """
    Run training with optimal configuration.
    
    Args:
        epochs: Number of training epochs
        test_run: If True, runs quick test with fewer epochs
    """
    
    # Build command with all enhancements
    cmd = [
        sys.executable,
        "train_robust_model_v2.py",
        
        # Data configuration
        "--data_path", "datasets/plantvillage_processed",
        "--preprocessing_mode", "default",  # Full CLAHE enhancement
        "--use_advanced_preprocessing", "True",
        
        # Model configuration
        "--batch_size", "16",  # Conservative for CPU
        "--epochs", str(epochs),
        "--learning_rate", "0.001",
        
        # Loss configuration
        "--loss_type", "combined",
        "--focal_weight", "0.7",
        "--focal_gamma", "2.0",
        "--label_smoothing_epsilon", "0.1",
        
        # Training enhancements
        "--swa_start_epoch", "20" if epochs > 20 else str(max(epochs - 3, 1)),
        "--gradient_clip_norm", "1.0",
        "--mixup_alpha", "0.2",
        "--mixup_probability", "0.5",
        
        # Output configuration
        "--output_dir", "models",
    ]
    
    if test_run:
        cmd.append("--test_run")
        print("\n" + "=" * 70)
        print("RUNNING TEST TRAINING (3 epochs)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"RUNNING FULL TRAINING ({epochs} epochs)")
        print("=" * 70)
    
    print("\nConfiguration:")
    print("  - Preprocessing: default (full CLAHE)")
    print("  - Loss: Combined (70% Focal + 30% Label Smoothing)")
    print("  - SWA: Starting at epoch", "20" if epochs > 20 else max(epochs - 3, 1))
    print("  - Batch size: 16")
    print("  - MixUp augmentation: Enabled")
    print("  - Gradient clipping: 1.0")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n[OK] Training completed successfully in {elapsed_time/60:.1f} minutes")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
        return False

def verify_dataset():
    """Verify dataset is ready for training."""
    data_path = Path("datasets/plantvillage_processed")
    
    if not data_path.exists():
        print("[ERROR] Dataset not found. Please run prepare_plantvillage_data.py first")
        return False
    
    # Check for train/val/test splits
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            print(f"[ERROR] Missing {dir_name} directory")
            return False
        
        # Count classes
        classes = [d for d in dir_path.iterdir() if d.is_dir()]
        if len(classes) < 6:
            print(f"[WARNING] Only {len(classes)} classes found in {dir_name}")
    
    print("[OK] Dataset verified and ready")
    return True

def check_gpu():
    """Check GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU available: {gpus}")
            return True
        else:
            print("[WARNING] No GPU found, training will use CPU (slower)")
            return False
    except:
        print("[WARNING] Could not check GPU status")
        return False

def main():
    """Main training orchestration."""
    print("=" * 70)
    print("ENHANCED MODEL TRAINING - FINAL PRODUCTION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Pre-flight checks
    print("\n" + "-" * 70)
    print("Pre-flight checks...")
    
    if not verify_dataset():
        print("\n[ERROR] Please prepare the dataset first:")
        print("   python prepare_plantvillage_data.py")
        return
    
    gpu_available = check_gpu()
    
    # Ask user for training mode
    print("\n" + "-" * 70)
    print("Training options:")
    print("1. Quick test run (3 epochs) - Verify configuration")
    print("2. Short training (10 epochs) - Quick results")
    print("3. Standard training (30 epochs) - Good balance")
    print("4. Full training (50 epochs) - Maximum accuracy")
    print("5. Custom epochs")
    
    choice = input("\nSelect option (1-5) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        # Test run
        success = run_training(epochs=3, test_run=True)
    elif choice == "2":
        # Short training
        success = run_training(epochs=10, test_run=False)
    elif choice == "3":
        # Standard training
        success = run_training(epochs=30, test_run=False)
    elif choice == "4":
        # Full training
        if not gpu_available:
            confirm = input("\n[WARNING] 50 epochs on CPU may take several hours. Continue? (y/n): ")
            if confirm.lower() != 'y':
                print("Training cancelled")
                return
        success = run_training(epochs=50, test_run=False)
    elif choice == "5":
        # Custom epochs
        epochs = input("Enter number of epochs: ").strip()
        try:
            epochs = int(epochs)
            if epochs < 1:
                print("[ERROR] Invalid number of epochs")
                return
            success = run_training(epochs=epochs, test_run=False)
        except ValueError:
            print("[ERROR] Invalid input")
            return
    else:
        print("[ERROR] Invalid choice")
        return
    
    if success:
        print("\n" + "=" * 70)
        print("POST-TRAINING TASKS")
        print("=" * 70)
        
        # Check for saved models
        model_files = list(Path("models").glob("enhanced_*.h5"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"\n[OK] Latest model saved: {latest_model}")
            
            # Offer to evaluate
            evaluate = input("\nRun evaluation on the new model? (y/n) [default: y]: ").strip() or "y"
            if evaluate.lower() == 'y':
                print("\nRunning evaluation...")
                try:
                    subprocess.run([
                        sys.executable,
                        "evaluate_real_world.py",
                        "--model_path", str(latest_model),
                        "--output_dir", "evaluation_results_enhanced"
                    ], check=True)
                    print("\n[OK] Evaluation complete. Check evaluation_results_enhanced/")
                except:
                    print("\n[WARNING] Evaluation failed. Run manually:")
                    print(f"   python evaluate_real_world.py --model_path {latest_model}")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check models/ for saved checkpoints")
        print("2. Review logs/ for TensorBoard visualizations")
        print("3. Test on real images: python inference_real_world.py <image>")
        print("4. Compare with baseline using evaluate_real_world.py")

if __name__ == "__main__":
    main()