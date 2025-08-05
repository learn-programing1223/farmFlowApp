#!/usr/bin/env python3
"""
Train PlantPulse model with real disease data
Combines existing thermal data with disease-converted thermal images
"""

import os
import sys
from pathlib import Path
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_with_real_thermal import train_with_dataset
from download_disease_datasets import DiseaseDatasetDownloader

def main():
    print("\n" + "="*60)
    print("PLANTPULSE TRAINING WITH REAL DISEASE DATA")
    print("="*60)
    
    # Check if disease data exists
    disease_thermal_dir = Path("data/disease_datasets/thermal_diseases")
    combined_dir = Path("data/all_thermal_datasets/combined_with_diseases")
    
    if not disease_thermal_dir.exists():
        print("\n⚠️  Disease thermal data not found!")
        print("Would you like to download and prepare disease datasets?")
        response = input("This will take 30-60 minutes (y/n): ")
        
        if response.lower() == 'y':
            print("\nDownloading disease datasets...")
            downloader = DiseaseDatasetDownloader()
            downloader.download_and_prepare_all()
        else:
            print("\nPlease run this first:")
            print("  python download_disease_datasets.py")
            return
    
    if not combined_dir.exists():
        print("\n⚠️  Combined dataset not found!")
        print("Creating combined dataset...")
        downloader = DiseaseDatasetDownloader()
        downloader.create_combined_thermal_dataset(disease_thermal_dir)
    
    # Count images in each category
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    
    total_images = 0
    disease_counts = {}
    
    # Count disease images
    for disease_type in ["healthy", "bacterial", "fungal", "viral"]:
        count = len(list(combined_dir.glob(f"disease_{disease_type}_*.png")))
        disease_counts[disease_type] = count
        total_images += count
        print(f"{disease_type.capitalize()}: {count} images")
    
    # Count existing thermal images
    existing_count = len(list(combined_dir.glob("existing_*.png")))
    total_images += existing_count
    print(f"Existing thermal: {existing_count} images")
    print(f"TOTAL: {total_images} images")
    print("-" * 40)
    
    # Check balance
    if disease_counts.get("bacterial", 0) == 0 or disease_counts.get("fungal", 0) == 0:
        print("\n⚠️  WARNING: Missing bacterial or fungal disease examples!")
        print("The model may still achieve 100% accuracy on disease classification.")
        print("Consider downloading more disease datasets.")
    
    # Train model
    print(f"\n🚀 Starting training with {total_images} images...")
    print("This dataset includes REAL disease patterns!")
    print("Expected improvements:")
    print("- Disease classification should be 75-85% (not 100%)")
    print("- Better generalization to real-world diseases")
    print("- More robust multi-class disease detection")
    
    input("\nPress Enter to start training...")
    
    # Train with combined dataset
    model, history = train_with_dataset(str(combined_dir), "combined_diseases")
    
    if model and history:
        print("\n✅ Training completed successfully!")
        
        # Check final disease accuracy
        final_disease_acc = history.history.get('val_disease_accuracy', [0])[-1]
        
        if final_disease_acc >= 0.99:
            print("\n⚠️  WARNING: Disease accuracy is still suspiciously high!")
            print("Possible issues:")
            print("1. Disease images may not be diverse enough")
            print("2. Thermal conversion may be too uniform")
            print("3. Consider adding more augmentation")
        else:
            print(f"\n✅ Disease accuracy: {final_disease_acc:.1%}")
            print("This is a more realistic accuracy for disease detection!")
        
        # Find the latest model
        import glob
        tflite_files = glob.glob("thermal_model_combined_diseases_*.tflite")
        if tflite_files:
            latest_model = sorted(tflite_files)[-1]
            print(f"\n📱 Deploy this model to your app:")
            print(f"  cp {latest_model} ../src/ml/models/plant_health_thermal.tflite")
            
            # Show improvement over previous model
            print("\n📈 Improvements over previous model:")
            print("- Real disease detection (not just stress)")
            print("- Bacterial vs Fungal vs Viral classification")
            print("- More robust to real-world conditions")

if __name__ == "__main__":
    main()