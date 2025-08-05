"""
Automated thermal training runner
"""

import sys
import os

# Add the training directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_with_real_thermal import train_with_dataset

if __name__ == "__main__":
    print("AUTOMATED THERMAL TRAINING")
    print("=" * 60)
    
    # Check available datasets
    datasets = [
        ("data/quick_thermal_test", "generic_thermal", "Quick Test (400 images)"),
        ("data/test_thermal", "generic_thermal", "Test Dataset (15 images)"),
        ("data/synthetic_thermal_advanced", "combined", "Advanced Synthetic (10K images)"),
        ("data/eth_thermal", "eth_zurich", "ETH Zurich Professional"),
        ("data/all_thermal_datasets/combined", "combined", "All Combined Datasets")
    ]
    
    # Find first available dataset
    selected = None
    for path, dtype, name in datasets:
        if os.path.exists(path):
            print(f"✅ Found dataset: {name} at {path}")
            selected = (path, dtype, name)
            break
    
    if selected:
        path, dtype, name = selected
        print(f"\nTraining with: {name}")
        print("=" * 60)
        
        # Train with the dataset
        model, history = train_with_dataset(path, dtype)
        
        if model and history:
            print("\n✅ Training completed successfully!")
            print("\nModel is ready for deployment in your PlantPulse app!")
    else:
        print("\n❌ No datasets found!")
        print("\nPlease run one of these commands first:")
        print("  python generate_synthetic_thermal.py  # Quick synthetic data")
        print("  python download_eth_dataset.py       # Professional thermal data")