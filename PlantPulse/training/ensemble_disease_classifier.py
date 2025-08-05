#!/usr/bin/env python3
"""
Ensemble disease classifier combining multiple approaches
Achieves higher accuracy through model diversity
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ThermalFeatureExtractor:
    """Extract hand-crafted thermal features for disease detection"""
    
    @staticmethod
    def extract_temperature_statistics(thermal_image):
        """Extract statistical features from thermal data"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(thermal_image),
            np.std(thermal_image),
            np.min(thermal_image),
            np.max(thermal_image),
            np.median(thermal_image),
            np.percentile(thermal_image, 25),
            np.percentile(thermal_image, 75)
        ])
        
        # Temperature gradients
        grad_x = cv2.Sobel(thermal_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(thermal_image, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.std(grad_x),
            np.std(grad_y)
        ])
        
        return features
    
    @staticmethod
    def extract_texture_features(thermal_image):
        """Extract texture features using GLCM"""
        # Convert to uint8 for texture analysis
        thermal_uint8 = ((thermal_image - np.min(thermal_image)) / 
                        (np.max(thermal_image) - np.min(thermal_image)) * 255).astype(np.uint8)
        
        # Calculate Local Binary Pattern
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(thermal_uint8, P=8, R=1, method='uniform')
        
        # LBP histogram as features
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        hist = hist.astype(float) / hist.sum()
        
        return hist.tolist()
    
    @staticmethod
    def extract_spatial_features(thermal_image):
        """Extract spatial distribution features"""
        h, w = thermal_image.shape
        features = []
        
        # Divide into quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            thermal_image[:mid_h, :mid_w],
            thermal_image[:mid_h, mid_w:],
            thermal_image[mid_h:, :mid_w],
            thermal_image[mid_h:, mid_w:]
        ]
        
        # Temperature variance between quadrants
        quad_means = [np.mean(q) for q in quadrants]
        features.append(np.std(quad_means))
        
        # Center vs edge temperature difference
        center_region = thermal_image[h//4:3*h//4, w//4:3*w//4]
        edge_mask = np.ones_like(thermal_image, dtype=bool)
        edge_mask[h//4:3*h//4, w//4:3*w//4] = False
        edge_region = thermal_image[edge_mask]
        
        features.extend([
            np.mean(center_region) - np.mean(edge_region),
            np.std(center_region),
            np.std(edge_region)
        ])
        
        return features

def create_model_v1():
    """CNN with focus on local patterns"""
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 1)),
        
        # Small kernels for local pattern detection
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    
    return model

def create_model_v2():
    """CNN with larger receptive fields"""
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 1)),
        
        # Larger kernels for global pattern detection
        layers.Conv2D(32, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    
    return model

def create_model_v3():
    """Feature fusion model"""
    inputs = layers.Input(shape=(224, 224, 1))
    
    # Path 1: Fine-grained features
    fine = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    fine = layers.Conv2D(32, 3, activation='relu', padding='same')(fine)
    fine = layers.MaxPooling2D(2)(fine)
    
    # Path 2: Coarse features
    coarse = layers.Conv2D(32, 7, activation='relu', padding='same', strides=2)(inputs)
    
    # Merge paths
    merged = layers.Concatenate()([fine, coarse])
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    
    combined = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

class EnsembleClassifier:
    """Ensemble of multiple models for improved accuracy"""
    
    def __init__(self):
        self.models = []
        self.feature_extractor = ThermalFeatureExtractor()
        self.use_features = True
        
    def create_models(self):
        """Create diverse models for ensemble"""
        print("Creating ensemble models...")
        
        # Model 1: Local pattern focus
        model1 = create_model_v1()
        model1.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Model 2: Global pattern focus
        model2 = create_model_v2()
        model2.compile(
            optimizer=keras.optimizers.Adam(0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Model 3: Multi-scale fusion
        model3 = create_model_v3()
        model3.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models = [model1, model2, model3]
        print(f"Created {len(self.models)} models for ensemble")
        
    def preprocess_thermal(self, thermal_image, augment=False):
        """Preprocess thermal image with optional augmentation"""
        
        # Normalize
        thermal_norm = (thermal_image - np.mean(thermal_image)) / (np.std(thermal_image) + 1e-7)
        
        if augment:
            # Random contrast adjustment
            if np.random.random() > 0.5:
                contrast_factor = np.random.uniform(0.8, 1.2)
                thermal_norm = thermal_norm * contrast_factor
            
            # Random gaussian noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.1, thermal_norm.shape)
                thermal_norm = thermal_norm + noise
            
            # Random shift
            if np.random.random() > 0.5:
                shift_pixels = np.random.randint(-10, 10)
                thermal_norm = np.roll(thermal_norm, shift_pixels, axis=np.random.choice([0, 1]))
        
        # Clip values
        thermal_norm = np.clip(thermal_norm, -3, 3)
        
        return thermal_norm
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train all models in the ensemble"""
        
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{len(self.models)}")
            print(f"{'='*60}")
            
            # Create augmented data generator for this model
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20 * (i + 1) / len(self.models),  # Different augmentation per model
                width_shift_range=0.1 * (i + 1) / len(self.models),
                height_shift_range=0.1 * (i + 1) / len(self.models),
                zoom_range=0.1,
                horizontal_flip=i % 2 == 0,  # Some models use flip, others don't
                vertical_flip=i % 2 == 1
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            histories.append(history)
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"Model {i+1} - Validation Accuracy: {val_acc:.3f}")
        
        return histories
    
    def predict_ensemble(self, X, use_voting='soft'):
        """Make predictions using ensemble"""
        
        if use_voting == 'soft':
            # Average probabilities
            predictions = []
            for model in self.models:
                pred = model.predict(X, verbose=0)
                predictions.append(pred)
            
            ensemble_pred = np.mean(predictions, axis=0)
            
        else:  # hard voting
            # Majority vote
            predictions = []
            for model in self.models:
                pred = model.predict(X, verbose=0)
                pred_classes = np.argmax(pred, axis=1)
                predictions.append(pred_classes)
            
            predictions = np.array(predictions)
            ensemble_pred = []
            
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                majority_vote = unique[np.argmax(counts)]
                ensemble_pred.append(majority_vote)
            
            ensemble_pred = keras.utils.to_categorical(ensemble_pred, 4)
        
        return ensemble_pred

def analyze_results(y_true, y_pred, class_names):
    """Analyze and visualize classification results"""
    
    # Get class predictions
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Disease Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('disease_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    for class_name in class_names:
        metrics = report[class_name]
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1-score']:.3f}")
    
    print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
    
    return report

def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("ENSEMBLE DISEASE CLASSIFIER")
    print("="*60)
    print("Using 3 diverse models for improved accuracy")
    print("Expected accuracy: 75-85%")
    print("="*60)
    
    # Load data
    from train_with_diseases_efficient import create_subset_dataset
    
    disease_dir = Path("data/disease_datasets/thermal_subset")
    if not disease_dir.exists():
        print("Creating balanced subset...")
        full_dir = Path("data/disease_datasets/thermal_diseases")
        create_subset_dataset(full_dir, disease_dir, samples_per_class=2000)
    
    # Prepare data
    print("\nLoading data...")
    X = []
    y = []
    class_names = ['healthy', 'bacterial', 'fungal', 'viral']
    
    for i, class_name in enumerate(class_names):
        class_dir = disease_dir / class_name
        if class_dir.exists():
            for img_path in list(class_dir.glob('*.png'))[:2000]:  # Limit per class
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    img = img.astype(np.float32) / 255.0
                    X.append(img)
                    y.append(i)
    
    X = np.array(X)
    X = np.expand_dims(X, -1)
    y = keras.utils.to_categorical(y, 4)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
    )
    
    print(f"\nDataset size:")
    print(f"Training: {X_train.shape[0]} images")
    print(f"Validation: {X_val.shape[0]} images")
    
    # Create and train ensemble
    ensemble = EnsembleClassifier()
    ensemble.create_models()
    
    # Train
    histories = ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate ensemble
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    ensemble_pred = ensemble.predict_ensemble(X_val, use_voting='soft')
    ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == np.argmax(y_val, axis=1))
    
    print(f"\nEnsemble Accuracy (Soft Voting): {ensemble_acc:.3f}")
    
    # Analyze results
    report = analyze_results(y_val, ensemble_pred, class_names)
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, model in enumerate(ensemble.models):
        model.save(f'ensemble_model_{i+1}_{timestamp}.h5')
    
    print(f"\n✅ Training complete!")
    print(f"Final ensemble accuracy: {ensemble_acc:.3f}")
    print(f"Models saved with timestamp: {timestamp}")
    
    # Save results
    results = {
        'ensemble_accuracy': float(ensemble_acc),
        'individual_accuracies': [h.history['val_accuracy'][-1] for h in histories],
        'classification_report': report,
        'timestamp': timestamp
    }
    
    with open(f'ensemble_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return ensemble, histories

if __name__ == "__main__":
    print("This ensemble approach combines multiple strategies:")
    print("1. Different model architectures")
    print("2. Various augmentation strategies")
    print("3. Soft voting for final predictions")
    
    response = input("\nStart ensemble training? (y/n): ")
    if response.lower() == 'y':
        main()