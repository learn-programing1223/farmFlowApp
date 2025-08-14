#!/usr/bin/env python3
"""
Monitor the robust PlantVillage model training progress
"""

import os
import time
import json
from pathlib import Path
import re

def monitor_training():
    """Monitor ongoing training and display progress"""
    
    print("=" * 70)
    print("ROBUST MODEL TRAINING MONITOR")
    print("=" * 70)
    
    # Check for training output
    log_files = [
        'robust_plantvillage_training.log',
        'training.log',
        'train_robust_plantvillage.log'
    ]
    
    active_log = None
    for log in log_files:
        if Path(log).exists():
            active_log = log
            break
    
    if not active_log:
        print("\nNo active training log found.")
        print("Training may not have started yet or is running elsewhere.")
        return
    
    print(f"\nMonitoring: {active_log}")
    print("-" * 50)
    
    # Read last lines of log
    try:
        with open(active_log, 'r') as f:
            lines = f.readlines()
            
        # Extract key metrics
        current_epoch = None
        best_val_acc = 0.0
        current_val_acc = 0.0
        current_loss = None
        
        for line in lines[-100:]:  # Check last 100 lines
            # Find epoch info
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = f"Epoch {epoch_match.group(1)}/{epoch_match.group(2)}"
            
            # Find validation accuracy
            val_acc_match = re.search(r'val_accuracy: ([\d.]+)', line)
            if val_acc_match:
                current_val_acc = float(val_acc_match.group(1))
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
            
            # Find loss
            loss_match = re.search(r'loss: ([\d.]+)', line)
            if loss_match:
                current_loss = float(loss_match.group(1))
        
        # Display status
        if current_epoch:
            print(f"\n📊 Training Progress:")
            print(f"  Current: {current_epoch}")
            if current_loss:
                print(f"  Loss: {current_loss:.4f}")
            if current_val_acc > 0:
                print(f"  Current Val Accuracy: {current_val_acc:.2%}")
            if best_val_acc > 0:
                print(f"  Best Val Accuracy: {best_val_acc:.2%}")
        
        # Show recent output
        print(f"\n📜 Recent Training Output:")
        print("-" * 50)
        for line in lines[-10:]:
            print(line.strip())
        
    except Exception as e:
        print(f"Error reading log: {e}")
    
    # Check for saved models
    print("\n" + "-" * 50)
    print("💾 Saved Models:")
    
    model_dir = Path('models')
    if model_dir.exists():
        models = list(model_dir.glob('*plantvillage*.h5')) + \
                list(model_dir.glob('*plantvillage*.tflite'))
        
        if models:
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  ✓ {model.name}: {size_mb:.2f} MB")
        else:
            print("  No PlantVillage models saved yet")
    
    # Check dataset
    print("\n" + "-" * 50)
    print("📁 Dataset Status:")
    
    processed_dir = Path('datasets/plantvillage_processed')
    if processed_dir.exists():
        # Count images
        train_count = sum(1 for _ in (processed_dir / 'train').rglob('*.jpg'))
        val_count = sum(1 for _ in (processed_dir / 'val').rglob('*.jpg'))
        test_count = sum(1 for _ in (processed_dir / 'test').rglob('*.jpg'))
        
        print(f"  Train: {train_count} images")
        print(f"  Val: {val_count} images")
        print(f"  Test: {test_count} images")
        print(f"  Total: {train_count + val_count + test_count} images")
    
    # Performance targets
    print("\n" + "-" * 50)
    print("🎯 Performance Targets:")
    print("  Target Accuracy: >85%")
    print("  Target Model Size: <10MB TFLite")
    print("  Target Inference: <100ms mobile")
    
    if best_val_acc >= 0.85:
        print(f"\n✅ TARGET ACHIEVED! Validation accuracy: {best_val_acc:.2%}")
    elif best_val_acc > 0:
        needed = 0.85 - best_val_acc
        print(f"\n⏳ Progress: {needed:.2%} more needed to reach target")

if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        monitor_training()
        print("\n" + "=" * 70)
        print("Refreshing in 30 seconds... (Ctrl+C to exit)")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break