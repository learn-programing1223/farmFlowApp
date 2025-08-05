"""
Train PlantPulse model with real thermal datasets
Uses ETH Zurich or other thermal datasets for improved accuracy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Import our thermal data loader
from thermal_data_loader import ThermalDatasetLoader

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
L2_WEIGHT = 0.01
DROPOUT_RATE = 0.5

print(f"TensorFlow version: {tf.__version__}")
print("Training with REAL thermal data!")

def create_thermal_model(dataset_type: str = "generic_thermal") -> keras.Model:
    """Create model optimized for thermal image analysis"""
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='thermal_input')
    
    # Data augmentation (built-in)
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    
    # Thermal-specific preprocessing
    # Add gaussian noise to simulate sensor variations
    x = layers.GaussianNoise(0.01)(x)
    
    # Feature extraction backbone
    x = layers.Conv2D(32, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.3)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.4)(x)
    
    # Thermal pattern detection layers
    x = layers.Conv2D(256, 3, activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    
    # Global feature aggregation
    global_features = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with strong regularization
    features = layers.Dense(256, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT))(global_features)
    features = layers.Dropout(DROPOUT_RATE)(features)
    features = layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    features = layers.Dropout(DROPOUT_RATE)(features)
    
    # Multi-task outputs
    water_stress_output = layers.Dense(1, activation='sigmoid', 
                                     name='water_stress',
                                     kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    # Determine number of disease classes based on dataset
    num_disease_classes = 4 if dataset_type != "combined" else 13  # More classes for combined
    disease_output = layers.Dense(num_disease_classes, activation='softmax', 
                                 name='disease',
                                 kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    nutrients_output = layers.Dense(3, activation='sigmoid', 
                                   name='nutrients',
                                   kernel_regularizer=regularizers.l2(L2_WEIGHT))(features)
    
    # Segmentation decoder (simplified)
    seg_features = layers.Dense(28 * 28 * 32)(features)
    seg = layers.Reshape((28, 28, 32))(seg_features)
    seg = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Conv2DTranspose(4, 3, strides=2, padding='same', activation='relu')(seg)
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(seg)
    
    model = keras.Model(
        inputs=inputs,
        outputs=[water_stress_output, disease_output, nutrients_output, segmentation_output]
    )
    
    return model

def train_with_dataset(dataset_path: str, dataset_type: str = "eth_zurich"):
    """Train model with real thermal dataset"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH {dataset_type.upper()} DATASET")
    print(f"{'='*60}")
    
    # Load dataset
    loader = ThermalDatasetLoader(dataset_path, dataset_type)
    
    try:
        train_data, val_data = loader.load_dataset(split=0.8)
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        print("\nMake sure you have downloaded the dataset first!")
        print("Run: python download_eth_dataset.py")
        return None
    
    # Create TensorFlow datasets
    train_ds = loader.create_tf_dataset(
        train_data[0], train_data[1],
        batch_size=BATCH_SIZE,
        augment=True
    )
    
    val_ds = loader.create_tf_dataset(
        val_data[0], val_data[1],
        batch_size=BATCH_SIZE,
        augment=False
    )
    
    # Build model
    print("\nBuilding thermal-optimized model...")
    model = create_thermal_model(dataset_type)
    
    # Count parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Model size estimate: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            'water_stress': 'mse',
            'disease': 'categorical_crossentropy',
            'nutrients': 'mse',
            'segmentation': 'binary_crossentropy'
        },
        loss_weights={
            'water_stress': 2.0,  # Important for thermal
            'disease': 2.0,
            'nutrients': 1.0,
            'segmentation': 0.5
        },
        metrics={
            'water_stress': ['mae'],
            'disease': ['accuracy'],
            'nutrients': ['mae'],
            'segmentation': ['accuracy']
        }
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"thermal_model_{dataset_type}_{timestamp}"
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_disease_loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_disease_loss',
            factor=0.5,
            patience=5,
            mode='min',
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_disease_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1
        )
    ]
    
    # Train
    print(f"\nTraining for up to {EPOCHS} epochs...")
    print("This uses REAL thermal data - expect better results than synthetic!")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(f'{model_name}_final.h5')
    
    # Save training history
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Plot results
    plot_training_results(history.history, model_name)
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Add representative dataset for quantization
    def representative_dataset():
        for images, _ in train_ds.take(100):
            yield [images]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    tflite_model = converter.convert()
    
    with open(f'{model_name}.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Training complete!")
    print(f"Models saved:")
    print(f"  - Best checkpoint: {model_name}_best.h5")
    print(f"  - Final model: {model_name}_final.h5")
    print(f"  - TFLite model: {model_name}.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    print(f"  - Training history: {model_name}_history.json")
    
    return model, history

def plot_training_results(history: dict, model_name: str):
    """Plot comprehensive training results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Disease accuracy
    ax = axes[0, 0]
    ax.plot(history.get('disease_accuracy', []), label='Train', linewidth=2)
    ax.plot(history.get('val_disease_accuracy', []), label='Validation', linewidth=2)
    ax.set_title('Disease Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Water stress MAE
    ax = axes[0, 1]
    ax.plot(history.get('water_stress_mae', []), label='Train', linewidth=2)
    ax.plot(history.get('val_water_stress_mae', []), label='Validation', linewidth=2)
    ax.set_title('Water Stress Prediction Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total loss
    ax = axes[0, 2]
    ax.plot(history.get('loss', []), label='Train', linewidth=2)
    ax.plot(history.get('val_loss', []), label='Validation', linewidth=2)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Segmentation accuracy
    ax = axes[1, 0]
    ax.plot(history.get('segmentation_accuracy', []), label='Train', linewidth=2)
    ax.plot(history.get('val_segmentation_accuracy', []), label='Validation', linewidth=2)
    ax.set_title('Segmentation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Nutrients MAE
    ax = axes[1, 1]
    ax.plot(history.get('nutrients_mae', []), label='Train', linewidth=2)
    ax.plot(history.get('val_nutrients_mae', []), label='Validation', linewidth=2)
    ax.set_title('Nutrient Prediction Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 2]
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'], linewidth=2, color='green')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Results: {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training pipeline"""
    
    print("PLANTPULSE THERMAL MODEL TRAINING")
    print("=" * 60)
    print("This script trains models with REAL thermal datasets")
    print("for improved accuracy over synthetic data.\n")
    
    # Check for available datasets
    datasets = {
        "1": ("data/eth_thermal", "eth_zurich"),
        "2": ("data/date_palm_thermal", "date_palm"),
        "3": ("data/test_thermal", "generic_thermal"),
        "4": ("data/custom_thermal", "generic_thermal"),
        "5": ("data/synthetic_thermal_advanced", "combined"),
        "6": ("data/quick_thermal_test", "generic_thermal"),
        "7": ("data/all_thermal_datasets/combined", "combined")
    }
    
    print("Available dataset options:")
    print("1. ETH Zurich Thermal (stress conditions)")
    print("2. Date Palm Thermal (pest damage)")
    print("3. Test Dataset (for validation)")
    print("4. Custom Dataset (your own thermal images)")
    print("5. Synthetic Thermal Advanced (10K images)")
    print("6. Quick Thermal Test (400 images)")
    print("7. Combined All Datasets (maximum robustness)")
    
    # Check which datasets exist
    for key, (path, _) in datasets.items():
        if Path(path).exists():
            print(f"   ✅ Option {key}: {path} EXISTS")
        else:
            print(f"   ❌ Option {key}: {path} NOT FOUND")
    
    # Allow command line argument or interactive selection
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        dataset_type = sys.argv[2] if len(sys.argv) > 2 else "generic_thermal"
    else:
        choice = input("\nSelect dataset (1-4) or enter custom path: ").strip()
        
        if choice in datasets:
            dataset_path, dataset_type = datasets[choice]
        else:
            dataset_path = choice
            dataset_type = input("Dataset type (eth_zurich/date_palm/generic_thermal): ").strip()
    
    # Train with selected dataset
    if Path(dataset_path).exists():
        train_with_dataset(dataset_path, dataset_type)
    else:
        print(f"\n❌ Dataset not found: {dataset_path}")
        print("\nPlease download a dataset first:")
        print("  python download_eth_dataset.py")
        print("\nOr create a test dataset:")
        print("  python thermal_data_loader.py")

if __name__ == "__main__":
    main()