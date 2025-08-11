#!/usr/bin/env python3
"""
START HERE - One-click solution to begin fixing your model
Run this script to start the improvement process
"""

import subprocess
import sys
from pathlib import Path
import time

def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def install_dependencies():
    """Install required packages"""
    print_banner("Installing Dependencies")
    
    packages = [
        'albumentations',
        'opencv-python',
        'Pillow',
        'requests',
        'selenium',
        'kaggle'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                      capture_output=True)
    
    print("✓ Dependencies installed")

def check_current_model():
    """Check current model status"""
    print_banner("Current Model Status")
    
    model_files = [
        ('models/best_cyclegan_model.h5', 'CycleGAN Model (88% lab accuracy)'),
        ('models/plant_disease_cyclegan_robust.tflite', 'TFLite Model'),
        ('../PlantPulse/assets/models/plant_disease_model.tflite', 'Deployed Model')
    ]
    
    for file_path, description in model_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"✓ {description}: {size_mb:.2f} MB")
        else:
            print(f"✗ {description}: Not found")

def create_quick_fix():
    """Create a quick fix script for immediate improvement"""
    print_banner("Creating Quick Fix Solution")
    
    quick_fix_content = '''#!/usr/bin/env python3
"""
Quick fix: Retrain with heavy augmentation
This will improve your model TODAY (4-6 hours)
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2

print("QUICK FIX: Retraining with extreme augmentation")
print("="*60)

# Load existing model
model_path = 'models/best_working_model.h5'
if not Path(model_path).exists():
    print("Error: Model not found!")
    print("Please ensure you have a trained model first")
    exit(1)

print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile with better optimizer
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load your existing data
print("Loading training data...")
# YOU NEED TO UPDATE THIS PATH
data_dir = Path('C:/Users/aayan/Downloads/PlantVillage')

if not data_dir.exists():
    print(f"Error: Data directory not found: {data_dir}")
    print("Please update the path to your PlantVillage dataset")
    exit(1)

# Create aggressive data augmentation
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    channel_shift_range=50,
    fill_mode='reflect',
    validation_split=0.2
)

# Load data
train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'models/quick_fix_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
]

print("Starting training with extreme augmentation...")
print("This will take 4-6 hours")
print("-"*60)

# Train
history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=callbacks
)

# Save final model
model.save('models/quick_fix_model_final.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/quick_fix_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\\n" + "="*60)
print("QUICK FIX COMPLETE!")
print("="*60)
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.2%}")
print("\\nModels saved:")
print("- models/quick_fix_model_best.h5")
print("- models/quick_fix_model.tflite")
print("\\nNext steps:")
print("1. Test on real images: python evaluate_real_world_performance.py")
print("2. Deploy if improved: cp models/quick_fix_model.tflite ../PlantPulse/assets/models/")
'''
    
    quick_fix_path = Path('quick_fix_training.py')
    quick_fix_path.write_text(quick_fix_content)
    print(f"✓ Created: {quick_fix_path}")
    
    return quick_fix_path

def show_menu():
    """Show interactive menu"""
    print_banner("PLANT DISEASE MODEL FIX")
    
    print("\nYour model works great on lab images but fails on real photos.")
    print("This tool will fix that problem.\n")
    
    print("Choose an option:")
    print("\n[1] QUICK FIX (4-6 hours)")
    print("    - Retrain with extreme augmentation")
    print("    - Uses your existing data")
    print("    - Expected improvement: 10-15%")
    
    print("\n[2] DOWNLOAD NEW DATA (1-2 hours)")
    print("    - Download PlantDoc (field images)")
    print("    - Scrape Google Images")
    print("    - Prepare for better training")
    
    print("\n[3] FULL SOLUTION (2-3 days)")
    print("    - Complete data collection")
    print("    - Modern architecture (EfficientNetV2)")
    print("    - Expected accuracy: 85%+ on real images")
    
    print("\n[4] TEST CURRENT MODEL")
    print("    - Evaluate on real internet images")
    print("    - See what's failing")
    
    print("\n[5] READ DOCUMENTATION")
    print("    - Understand the problem")
    print("    - See implementation details")
    
    print("\n[0] EXIT")
    
    return input("\nEnter choice (0-5): ")

def main():
    """Main execution"""
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7+ required")
        return
    
    # Install dependencies first
    install_dependencies()
    
    # Check current status
    check_current_model()
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Quick fix
            print_banner("Starting Quick Fix")
            quick_fix_script = create_quick_fix()
            
            print("\nReady to start quick fix training!")
            print(f"Run: python {quick_fix_script}")
            print("\nThis will:")
            print("1. Load your existing model")
            print("2. Apply extreme augmentation")
            print("3. Retrain for better real-world performance")
            print("4. Take 4-6 hours")
            
            start_now = input("\nStart training now? (y/n): ")
            if start_now.lower() == 'y':
                subprocess.run([sys.executable, str(quick_fix_script)])
            
        elif choice == '2':
            # Download data
            print_banner("Downloading Field Image Datasets")
            
            print("\nThis will help you download:")
            print("- PlantDoc (2,598 field images)")
            print("- Plant Pathology (23,000+ images)")
            print("- Google Images")
            
            subprocess.run([sys.executable, "download_field_datasets.py"])
            
        elif choice == '3':
            # Full solution
            print_banner("Full Solution Setup")
            
            print("\nSteps for full solution:")
            print("1. Run: python download_field_datasets.py")
            print("2. Download datasets manually (see instructions)")
            print("3. Run: python ultimate_augmentation_pipeline.py")
            print("4. Run: python train_ultimate_model.py")
            print("5. Run: python evaluate_real_world_performance.py")
            
            print("\nExpected timeline:")
            print("- Day 1: Data collection")
            print("- Day 2: Training")
            print("- Day 3: Evaluation & deployment")
            
            print("\nDocumentation:")
            print("- Read: ULTIMATE_TRAINING_PLAN.md")
            print("- Read: IMPLEMENTATION_ROADMAP.md")
            
        elif choice == '4':
            # Test model
            print_banner("Testing Current Model")
            
            if Path('evaluate_real_world_performance.py').exists():
                subprocess.run([sys.executable, "evaluate_real_world_performance.py"])
            else:
                print("Evaluation script not found!")
            
        elif choice == '5':
            # Documentation
            print_banner("Documentation")
            
            docs = [
                'ULTIMATE_TRAINING_PLAN.md',
                'IMPLEMENTATION_ROADMAP.md',
                'DEBUGGING_REPORT.md'
            ]
            
            for doc in docs:
                if Path(doc).exists():
                    print(f"✓ {doc}")
            
            print("\nOpen these files to understand:")
            print("- Why your model fails on real images")
            print("- How to fix it properly")
            print("- Expected results")
            
        elif choice == '0':
            print("\nGood luck with your model improvement!")
            break
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("="*60)
    print("  PLANT DISEASE MODEL - REAL WORLD FIX")
    print("  Your solution to poor real-world performance")
    print("="*60)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please report this issue")