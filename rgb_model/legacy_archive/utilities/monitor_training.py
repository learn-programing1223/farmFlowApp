#!/usr/bin/env python3
"""
Monitor training progress and extract key metrics
"""

import time
from pathlib import Path
import re

def monitor_training():
    log_file = Path("training_log.txt")
    
    if not log_file.exists():
        print("Training log not found. Is training running?")
        return
    
    print("=" * 60)
    print("TRAINING MONITOR")
    print("=" * 60)
    
    # Read last lines of log
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find latest epoch info
    epoch_pattern = r"Epoch (\d+)/(\d+)"
    val_acc_pattern = r"val_accuracy: ([\d.]+)"
    val_loss_pattern = r"val_loss: ([\d.]+)"
    
    latest_epoch = None
    best_val_acc = 0
    
    for line in lines:
        # Check for epoch
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            latest_epoch = f"Epoch {epoch_match.group(1)}/{epoch_match.group(2)}"
        
        # Check for validation accuracy
        val_acc_match = re.search(val_acc_pattern, line)
        if val_acc_match:
            val_acc = float(val_acc_match.group(1))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # Check if training completed
    if "TRAINING COMPLETE" in "".join(lines[-10:]):
        print("\n✓ TRAINING COMPLETE!")
        
        # Extract final results
        for line in lines[-20:]:
            if "Best validation accuracy:" in line:
                print(line.strip())
            if "Models saved:" in line:
                print("\n" + line.strip())
                # Print next 3 lines (model paths)
                idx = lines.index(line)
                for i in range(1, 4):
                    if idx + i < len(lines):
                        print(lines[idx + i].strip())
    else:
        print(f"\nStatus: TRAINING IN PROGRESS")
        if latest_epoch:
            print(f"Current: {latest_epoch}")
        print(f"Best validation accuracy so far: {best_val_acc:.2%}")
        
        # Show last few lines
        print("\nLatest output:")
        print("-" * 40)
        for line in lines[-5:]:
            print(line.strip())
    
    # Check model files
    print("\n" + "-" * 40)
    print("Model files:")
    model_dir = Path("models")
    if model_dir.exists():
        for model_file in model_dir.glob("robust_final*.h5"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name}: {size_mb:.2f} MB")
        for model_file in model_dir.glob("robust_final*.tflite"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    monitor_training()