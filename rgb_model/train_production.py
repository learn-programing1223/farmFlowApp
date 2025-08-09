#!/usr/bin/env python3
"""
Production-ready RGB plant disease detection model training
Implements all research-based improvements for 80%+ accuracy target
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import time

# Disable verbose TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class ImprovedCNN:
    """Improved CNN architecture for plant disease detection"""
    
    @staticmethod
    def build(num_classes=7, input_shape=(224, 224, 3)):
        """Build the improved CNN model"""
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=input_shape),
            
            # Block 1: Initial feature extraction
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Block 2: Deeper features
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Block 3: Complex patterns
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Block 4: High-level features
            tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(512, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense classifier
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model


class DataPipeline:
    """Efficient data loading and augmentation pipeline"""
    
    @staticmethod
    def load_data(data_dir='./data/splits'):
        """Load preprocessed data"""
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        print("Loading data...")
        
        # Load arrays
        X_train = np.load(data_dir / 'X_train.npy')
        y_train = np.load(data_dir / 'y_train.npy')
        X_val = np.load(data_dir / 'X_val.npy')
        y_val = np.load(data_dir / 'y_val.npy')
        
        # Normalize to [0, 1] if needed
        if X_train.max() > 1.0:
            X_train = X_train.astype(np.float32) / 255.0
            X_val = X_val.astype(np.float32) / 255.0
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        
        # Load test if available
        X_test = None
        y_test = None
        if (data_dir / 'X_test.npy').exists():
            X_test = np.load(data_dir / 'X_test.npy')
            y_test = np.load(data_dir / 'y_test.npy')
            if X_test.max() > 1.0:
                X_test = X_test.astype(np.float32) / 255.0
            print(f"  Test: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @staticmethod
    def create_augmented_dataset(X, y, batch_size=32, training=True):
        """Create TF dataset with augmentation"""
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if training:
            # Shuffle with large buffer
            dataset = dataset.shuffle(buffer_size=min(5000, len(X)))
            
            # Data augmentation
            def augment(image, label):
                # Random transformations
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                
                # Random rotation (90 degree increments)
                k = tf.random.uniform([], 0, 4, dtype=tf.int32)
                image = tf.image.rot90(image, k)
                
                # Color augmentation
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.8, 1.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
                
                # Ensure values are in [0, 1]
                image = tf.clip_by_value(image, 0.0, 1.0)
                
                return image, label
            
            dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class TrainingManager:
    """Manages the training process with all optimizations"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.histories = []
        
    def compile_model(self, learning_rate=1e-3):
        """Compile model with optimal settings"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.config.get('label_smoothing', 0.1)
            ),
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
    
    def train_progressive(self, train_data, val_data):
        """Progressive training with multiple stages"""
        
        # Stage 1: Initial training
        print("\nStage 1: Initial training (higher LR)")
        print("-" * 40)
        
        self.compile_model(learning_rate=1e-3)
        
        callbacks_stage1 = self._get_callbacks('stage1', patience=10)
        
        history1 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['stage1_epochs'],
            callbacks=callbacks_stage1,
            verbose=2
        )
        self.histories.append(history1.history)
        
        # Stage 2: Fine-tuning
        print("\nStage 2: Fine-tuning (lower LR)")
        print("-" * 40)
        
        self.compile_model(learning_rate=1e-4)
        
        callbacks_stage2 = self._get_callbacks('stage2', patience=15)
        
        history2 = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['stage2_epochs'],
            callbacks=callbacks_stage2,
            verbose=2
        )
        self.histories.append(history2.history)
        
        # Stage 3: Final refinement
        if self.config.get('stage3_epochs', 0) > 0:
            print("\nStage 3: Final refinement (very low LR)")
            print("-" * 40)
            
            self.compile_model(learning_rate=1e-5)
            
            callbacks_stage3 = self._get_callbacks('stage3', patience=20)
            
            history3 = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=self.config['stage3_epochs'],
                callbacks=callbacks_stage3,
                verbose=2
            )
            self.histories.append(history3.history)
        
        return self.histories
    
    def _get_callbacks(self, stage_name, patience=10):
        """Get callbacks for training stage"""
        checkpoint_dir = Path('./models/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / f'{stage_name}_{timestamp}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callbacks


def evaluate_model(model, test_data, class_names):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Evaluate
    results = model.evaluate(test_data, verbose=2)
    
    # Display metrics
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        if 'loss' in name:
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value:.4f} ({value*100:.2f}%)")
    
    return results[1]  # Return accuracy


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("PRODUCTION RGB MODEL TRAINING")
    print("Target: 80%+ Validation Accuracy")
    print("="*60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_classes': 7,
        'stage1_epochs': 30,
        'stage2_epochs': 20,
        'stage3_epochs': 10,
        'label_smoothing': 0.1
    }
    
    # Class names
    class_names = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew',
                   'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    
    try:
        # Load data
        print("\n1. Loading data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = DataPipeline.load_data()
        
        # Create datasets
        print("\n2. Creating augmented datasets...")
        train_dataset = DataPipeline.create_augmented_dataset(
            X_train, y_train, config['batch_size'], training=True
        )
        val_dataset = DataPipeline.create_augmented_dataset(
            X_val, y_val, config['batch_size'], training=False
        )
        
        # Build model
        print("\n3. Building model...")
        model = ImprovedCNN.build(config['num_classes'])
        print(f"  Total parameters: {model.count_params():,}")
        
        # Train model
        print("\n4. Starting progressive training...")
        print("="*60)
        
        start_time = time.time()
        trainer = TrainingManager(model, config)
        histories = trainer.train_progressive(train_dataset, val_dataset)
        
        training_time = time.time() - start_time
        print(f"\nTotal training time: {training_time/60:.2f} minutes")
        
        # Get best validation accuracy
        all_val_acc = []
        for history in histories:
            if 'val_accuracy' in history:
                all_val_acc.extend(history['val_accuracy'])
        
        best_val_acc = max(all_val_acc) if all_val_acc else 0
        final_val_acc = histories[-1]['val_accuracy'][-1] if histories else 0
        
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        print(f"Final validation accuracy: {final_val_acc:.2%}")
        
        if best_val_acc >= 0.80:
            print("SUCCESS: Achieved 80%+ validation accuracy!")
        else:
            print(f"Current: {best_val_acc:.2%} (Target: 80%)")
        
        # Test evaluation
        if X_test is not None:
            print("\n5. Evaluating on test set...")
            test_dataset = DataPipeline.create_augmented_dataset(
                X_test, y_test, config['batch_size'], training=False
            )
            test_acc = evaluate_model(model, test_dataset, class_names)
            
            if test_acc >= 0.80:
                print("SUCCESS: Achieved 80%+ test accuracy!")
        
        # Save model
        print("\n6. Saving model...")
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f'rgb_production_{timestamp}.h5'
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training history
        history_path = model_dir / f'history_production_{timestamp}.json'
        with open(history_path, 'w') as f:
            combined_history = {}
            for i, history in enumerate(histories):
                for key, values in history.items():
                    stage_key = f"stage{i+1}_{key}"
                    combined_history[stage_key] = [float(v) for v in values]
            json.dump(combined_history, f, indent=2)
        print(f"History saved to: {history_path}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())