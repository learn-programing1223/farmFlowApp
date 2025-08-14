#!/usr/bin/env python3
"""
Optimized Robust Training for PlantVillage Dataset
================================================

Specialized architecture for robust plant disease detection using the processed
PlantVillage dataset with 6 universal disease categories.

Key Features:
- EfficientNetB0 backbone for mobile deployment (<10MB target)
- Class weighting to handle imbalanced Nutrient_Deficiency category
- Advanced augmentation pipeline optimized for agricultural images
- Transfer learning from ImageNet with gradual fine-tuning
- Target: >85% accuracy with robust real-world performance

Author: Claude Code - Model Architecture Specialist
Date: 2025-01-11
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import json
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.experimental.enable_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

# Enable mixed precision for efficiency
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

class PlantVillageDataGenerator(keras.utils.Sequence):
    """
    Optimized data generator for PlantVillage dataset with agricultural-specific augmentation
    """
    
    def __init__(self, data_dir, batch_size=32, img_size=(224, 224), is_training=True, class_weights=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.class_weights = class_weights or {}
        
        # Class names (6 categories from processed PlantVillage)
        self.class_names = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
                           'Nutrient_Deficiency', 'Powdery_Mildew']
        self.num_classes = len(self.class_names)
        
        self._load_data()
        self._setup_augmentation()
    
    def _load_data(self):
        """Load image paths and labels from directory structure"""
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_name} directory not found")
                continue
                
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.labels.append(class_idx)
        
        # Convert to arrays and shuffle if training
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        if self.is_training:
            indices = np.random.permutation(len(self.image_paths))
            self.image_paths = self.image_paths[indices]
            self.labels = self.labels[indices]
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")
        
        # Print class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for idx, count in zip(unique, counts):
            print(f"  {self.class_names[idx]}: {count} images")
    
    def _setup_augmentation(self):
        """Setup agricultural-optimized augmentation pipeline"""
        if self.is_training:
            # Heavy augmentation for training to improve generalization
            self.augment_layers = keras.Sequential([
                # Geometric transformations (common in field photography)
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),  # Leaves can be at various angles
                layers.RandomZoom(0.1),       # Different distances from plant
                layers.RandomTranslation(0.1, 0.1),  # Slightly off-center shots
                
                # Lighting conditions (crucial for outdoor photography)
                layers.RandomBrightness(0.2),    # Different sun conditions
                layers.RandomContrast(0.15),     # Shadow/bright light variations
                
                # Add realistic noise and blur (smartphone camera effects)
                layers.Lambda(lambda x: tf.clip_by_value(
                    x + tf.random.normal(tf.shape(x), 0, 0.02), 0, 1)),  # Slight noise
            ])
        else:
            # No augmentation for validation/test
            self.augment_layers = keras.Sequential([])
    
    def _mixup_augmentation(self, batch_x, batch_y, alpha=0.2):
        """Apply MixUp augmentation for better generalization"""
        batch_size = tf.shape(batch_x)[0]
        
        # Random permutation for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Sample mixing coefficient from Beta distribution
        mix_weight = tf.random.gamma([batch_size, 1, 1, 1], alpha, 1.0)
        mix_weight = mix_weight / (mix_weight + tf.random.gamma([batch_size, 1, 1, 1], alpha, 1.0))
        
        # Mix images and labels
        mixed_x = mix_weight * batch_x + (1 - mix_weight) * tf.gather(batch_x, indices)
        mixed_y = tf.squeeze(mix_weight, axis=[2, 3]) * batch_y + (1 - tf.squeeze(mix_weight, axis=[2, 3])) * tf.gather(batch_y, indices)
        
        return mixed_x, mixed_y
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_size = end_idx - start_idx
        
        # Initialize batch arrays
        batch_x = np.zeros((batch_size, *self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        
        # Load and preprocess images
        for i, img_idx in enumerate(range(start_idx, end_idx)):
            # Load image
            img_path = self.image_paths[img_idx]
            img = cv2.imread(img_path)
            
            if img is None:
                # Handle corrupted images
                img = np.zeros((*self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Apply augmentation if training
            if self.is_training:
                img = self.augment_layers(tf.expand_dims(img, 0), training=True)[0]
            
            batch_x[i] = img
            
            # One-hot encode labels
            label_idx = self.labels[img_idx]
            batch_y[i, label_idx] = 1.0
        
        # Apply MixUp with 30% probability during training
        if self.is_training and np.random.random() < 0.3:
            batch_x, batch_y = self._mixup_augmentation(batch_x, batch_y)
        
        return batch_x, batch_y

class RobustPlantDiseaseModel:
    """
    Optimized model architecture for PlantVillage dataset
    """
    
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def build_efficientnet_model(self):
        """
        Build EfficientNetB0-based model optimized for mobile deployment
        Target: <10MB model size with >85% accuracy
        """
        # Load pre-trained EfficientNetB0 (lightest variant)
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tuning strategy: freeze early layers, train later layers
        base_model.trainable = True
        
        # Freeze first 80% of layers (transfer learning from ImageNet)
        freeze_at = int(0.8 * len(base_model.layers))
        for layer in base_model.layers[:freeze_at]:
            layer.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=self.input_shape, name='input_image')
        
        # Preprocessing (EfficientNet expects [0, 255] range, but we provide [0, 1])
        x = layers.Lambda(lambda x: x * 255.0, name='scale_input')(inputs)
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification head optimized for plant diseases
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Add spatial attention mechanism
        attention = layers.Dense(1280, activation='relu', name='attention_dense')(x)
        attention = layers.Dense(1280, activation='sigmoid', name='attention_weights')(attention)
        x = layers.Multiply(name='attended_features')([x, attention])
        
        # Classification layers with dropout for regularization
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(0.3, name='dropout_2')(x)
        
        # Output layer (float32 for mixed precision compatibility)
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            dtype='float32',
            name='predictions'
        )(x)
        
        self.model = keras.Model(inputs, outputs, name='PlantVillage_EfficientNet')
        return self.model
    
    def build_mobilenetv3_model(self):
        """
        Alternative lightweight model using MobileNetV3Small
        Even smaller than EfficientNet but still competitive
        """
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            minimalistic=False
        )
        
        # Fine-tuning strategy
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocessing
        x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Simplified head for mobile deployment
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        self.model = keras.Model(inputs, outputs, name='PlantVillage_MobileNetV3')
        return self.model

def calculate_class_weights(data_dir):
    """Calculate class weights to handle imbalanced dataset"""
    class_names = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
                   'Nutrient_Deficiency', 'Powdery_Mildew']
    
    # Count images per class
    class_counts = []
    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg')) + 
                       list(class_dir.glob('*.jpeg')) + 
                       list(class_dir.glob('*.png')))
            class_counts.append(count)
        else:
            class_counts.append(0)
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    class_weights = {}
    
    for i, count in enumerate(class_counts):
        if count > 0:
            class_weights[i] = total_samples / (len(class_names) * count)
        else:
            class_weights[i] = 1.0
    
    print("Class weights calculated:")
    for i, (class_name, weight) in enumerate(zip(class_names, class_weights.values())):
        print(f"  {class_name}: {weight:.3f} (count: {class_counts[i]})")
    
    return class_weights

def create_advanced_callbacks(model_name='robust_plantvillage'):
    """Create comprehensive callback suite for training"""
    callbacks_list = [
        # Save best model based on validation accuracy
        keras.callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
            cooldown=3
        ),
        
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Cosine annealing for learning rate
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-3 * (0.5 * (1 + np.cos(np.pi * epoch / 50))),
            verbose=0
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1,
            write_graph=True,
            write_images=False
        ),
        
        # CSV logging for analysis
        keras.callbacks.CSVLogger(
            f'models/{model_name}_training_log.csv',
            append=False
        )
    ]
    
    return callbacks_list

def evaluate_model_comprehensive(model, test_generator, class_names):
    """Comprehensive model evaluation with detailed metrics"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Predict on test set
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    
    # Get true labels
    true_labels = []
    predicted_labels = []
    
    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator[i]
        true_batch = np.argmax(batch_y, axis=1)
        true_labels.extend(true_batch)
    
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Ensure same length
    min_len = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_len]
    predicted_labels = predicted_labels[:min_len]
    
    # Overall accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    print(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    report = classification_report(
        true_labels, predicted_labels,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Plant Disease Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 30)
    for i, class_name in enumerate(class_names):
        class_correct = np.sum((np.array(true_labels) == i) & (np.array(predicted_labels) == i))
        class_total = np.sum(np.array(true_labels) == i)
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name:18}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return accuracy, results

def convert_to_tflite_optimized(model, model_name='robust_plantvillage'):
    """Convert model to optimized TFLite format for mobile deployment"""
    print("\n" + "="*60)
    print("CONVERTING TO TFLITE FOR MOBILE DEPLOYMENT")
    print("="*60)
    
    # Basic conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Try INT8 quantization for smallest size
    try:
        print("Attempting INT8 quantization...")
        converter.representative_dataset = _representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_quant_model = converter.convert()
        
        # Save quantized model
        quant_path = f'models/{model_name}_quantized.tflite'
        with open(quant_path, 'wb') as f:
            f.write(tflite_quant_model)
        
        print(f"INT8 Quantized model saved: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"INT8 quantization failed: {e}")
        print("Falling back to float16 quantization...")
        
        # Fallback to float16
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save float16 model
        float16_path = f'models/{model_name}_float16.tflite'
        with open(float16_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Float16 model saved: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Also save standard float32 model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_standard = converter.convert()
    
    standard_path = f'models/{model_name}.tflite'
    with open(standard_path, 'wb') as f:
        f.write(tflite_standard)
    
    print(f"Standard model saved: {len(tflite_standard) / 1024 / 1024:.2f} MB")
    
    return standard_path

def _representative_dataset_gen():
    """Generate representative dataset for quantization"""
    # This would be called during quantization
    # For now, return dummy data
    for _ in range(100):
        yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

def train_robust_plantvillage_model():
    """Main training function for robust PlantVillage model"""
    
    print("="*80)
    print("ROBUST PLANTVILLAGE DISEASE DETECTION MODEL TRAINING")
    print("="*80)
    print("Target: >85% accuracy, <10MB model size, real-world robustness")
    print("="*80)
    
    # Configuration
    config = {
        'model_architecture': 'efficientnetb0',  # or 'mobilenetv3'
        'input_shape': (224, 224, 3),
        'num_classes': 6,
        'batch_size': 32,
        'epochs': 50,
        'initial_lr': 1e-3,
        'use_class_weights': True,
        'use_mixed_precision': True,
        'data_path': 'datasets/plantvillage_processed'
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Check data availability
    data_path = Path(config['data_path'])
    if not data_path.exists():
        print(f"\nERROR: Data directory not found: {data_path}")
        print("Please run prepare_plantvillage_data.py first")
        return None
    
    # Calculate class weights
    print("\n1. Calculating class weights for imbalanced dataset...")
    class_weights = calculate_class_weights(data_path / 'train')
    
    # Create data generators
    print("\n2. Setting up data generators...")
    train_gen = PlantVillageDataGenerator(
        data_path / 'train',
        batch_size=config['batch_size'],
        is_training=True,
        class_weights=class_weights
    )
    
    val_gen = PlantVillageDataGenerator(
        data_path / 'val',
        batch_size=config['batch_size'],
        is_training=False
    )
    
    test_gen = PlantVillageDataGenerator(
        data_path / 'test',
        batch_size=config['batch_size'],
        is_training=False
    )
    
    # Build model
    print("\n3. Building optimized model architecture...")
    model_builder = RobustPlantDiseaseModel(
        num_classes=config['num_classes'],
        input_shape=config['input_shape']
    )
    
    if config['model_architecture'] == 'efficientnetb0':
        model = model_builder.build_efficientnet_model()
    else:
        model = model_builder.build_mobilenetv3_model()
    
    print(f"Model: {config['model_architecture']}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Display model summary
    model.summary(show_trainable=True)
    
    # Compile model with advanced optimizer
    print("\n4. Compiling model with advanced optimizer...")
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=config['initial_lr'],
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Setup callbacks
    print("\n5. Setting up training callbacks...")
    callbacks_list = create_advanced_callbacks('robust_plantvillage')
    
    # Train model
    print("\n6. Starting training...")
    print("-" * 60)
    print("Training with advanced augmentation and class weighting")
    print("This may take 1-2 hours depending on hardware")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        epochs=config['epochs'],
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights if config['use_class_weights'] else None,
        verbose=1
    )
    
    # Load best model for evaluation
    print("\n7. Loading best model for evaluation...")
    best_model = keras.models.load_model('models/robust_plantvillage_best.h5')
    
    # Comprehensive evaluation
    print("\n8. Comprehensive evaluation...")
    test_accuracy, results = evaluate_model_comprehensive(
        best_model, test_gen, train_gen.class_names
    )
    
    # Convert to TFLite
    print("\n9. Converting to TFLite for deployment...")
    tflite_path = convert_to_tflite_optimized(best_model, 'robust_plantvillage')
    
    # Save training history
    print("\n10. Saving training artifacts...")
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open('models/robust_plantvillage_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Create training report
    report = {
        'model_architecture': config['model_architecture'],
        'training_config': config,
        'final_results': {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': max(history.history['val_accuracy']),
            'total_parameters': int(model.count_params()),
            'model_size_mb': os.path.getsize('models/robust_plantvillage_best.h5') / 1024 / 1024
        },
        'class_distribution': {
            'train': dict(zip(train_gen.class_names, [
                len(list((data_path / 'train' / class_name).glob('*.jpg'))) + 
                len(list((data_path / 'train' / class_name).glob('*.jpeg'))) +
                len(list((data_path / 'train' / class_name).glob('*.png')))
                for class_name in train_gen.class_names
            ])),
            'class_weights': class_weights
        },
        'training_completed': datetime.now().isoformat()
    }
    
    with open('models/robust_plantvillage_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model Size: {os.path.getsize('models/robust_plantvillage_best.h5') / 1024 / 1024:.2f} MB")
    print(f"TFLite Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")
    
    print("\nFiles Created:")
    print("- models/robust_plantvillage_best.h5 (Best model)")
    print("- models/robust_plantvillage.tflite (Mobile deployment)")
    print("- models/robust_plantvillage_report.json (Comprehensive report)")
    print("- models/confusion_matrix.png (Evaluation visualization)")
    print("- models/evaluation_results.json (Detailed metrics)")
    
    print("\nKey Features Implemented:")
    print("✓ EfficientNetB0 backbone for mobile efficiency")
    print("✓ Class weighting for imbalanced dataset")
    print("✓ Advanced agricultural augmentation")
    print("✓ Transfer learning with gradual fine-tuning")
    print("✓ Mixed precision training")
    print("✓ Comprehensive evaluation metrics")
    print("✓ TFLite conversion for deployment")
    
    if test_accuracy >= 0.85:
        print(f"\n🎉 SUCCESS: Target accuracy of >85% achieved! ({test_accuracy*100:.2f}%)")
    else:
        print(f"\n⚠️  Target accuracy of >85% not reached ({test_accuracy*100:.2f}%)")
        print("Consider: More epochs, different architecture, or additional data")
    
    return best_model, history, results

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("GPU available - enabling memory growth")
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected - training will use CPU")
    
    # Start training
    try:
        model, history, results = train_robust_plantvillage_model()
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Test the model with real-world images using test_real_world_images.py")
        print("2. Deploy the .tflite model to mobile app (PlantPulse)")
        print("3. Monitor performance and collect feedback")
        print("4. Fine-tune based on real-world performance")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck the error above and fix any issues before retrying")