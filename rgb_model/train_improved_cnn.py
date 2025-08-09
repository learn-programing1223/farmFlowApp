#!/usr/bin/env python3
"""
Improved training script with custom CNN architecture
Implements all research-based improvements for 80%+ accuracy
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

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def build_improved_cnn(num_classes=7, input_shape=(224, 224, 3)):
    """Build an improved CNN architecture based on research findings"""
    
    model = tf.keras.Sequential([
        # Input normalization layer
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Block 1
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 2
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 3
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Block 4
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with warmup"""
    
    def __init__(self, initial_lr, warmup_epochs, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Set learning rate (compatible with newer TF versions)
        self.model.optimizer.learning_rate.assign(lr)
        print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.2e}")


def load_data(data_dir='./data/splits'):
    """Load preprocessed data"""
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found.")
    
    print("Loading preprocessed data...")
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Data range - Min: {X_train.min():.2f}, Max: {X_train.max():.2f}")
    
    # Ensure data is in [0, 1] range for standard CNN
    if X_train.max() > 1.0:
        print("Normalizing data to [0, 1] range...")
        X_train = X_train / 255.0
        X_val = X_val / 255.0
    
    # Load test data if available
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        if X_test.max() > 1.0:
            X_test = X_test / 255.0
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_augmented_generator(X, y, batch_size=32, training=True):
    """Create data generator with augmentation"""
    
    if training:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            shear_range=0.15,
            fill_mode='reflect'
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    return datagen.flow(X, y, batch_size=batch_size, shuffle=training)


def cutmix(images, labels, alpha=1.0, prob=0.5):
    """CutMix augmentation"""
    batch_size = images.shape[0]
    
    if np.random.random() > prob:
        return images, labels
    
    # Get mixing parameter
    lam = np.random.beta(alpha, alpha)
    
    # Get random index for mixing
    indices = np.random.permutation(batch_size)
    
    # Calculate cut size
    img_h, img_w = images.shape[1], images.shape[2]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(img_h * cut_ratio)
    cut_w = int(img_w * cut_ratio)
    
    # Get random position
    cy = np.random.randint(cut_h // 2, img_h - cut_h // 2)
    cx = np.random.randint(cut_w // 2, img_w - cut_w // 2)
    
    # Create mixed images
    mixed_images = images.copy()
    mixed_images[:, cy - cut_h // 2:cy + cut_h // 2, cx - cut_w // 2:cx + cut_w // 2, :] = \
        images[indices, cy - cut_h // 2:cy + cut_h // 2, cx - cut_w // 2:cx + cut_w // 2, :]
    
    # Mix labels
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_images, mixed_labels


def train_model_progressive(model, train_data, val_data, config):
    """Progressive training with multiple stages"""
    
    # Create directory for checkpoints
    checkpoint_dir = Path('./models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    histories = []
    
    # Stage 1: Initial training with higher learning rate
    print("\n" + "="*60)
    print("STAGE 1: Initial training")
    print("="*60)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'stage1_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CosineAnnealingScheduler(
            initial_lr=1e-3,
            warmup_epochs=3,
            total_epochs=30
        )
    ]
    
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=callbacks_stage1,
        verbose=1
    )
    histories.append(history1.history)
    
    # Stage 2: Fine-tuning with lower learning rate
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning")
    print("="*60)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'stage2_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks_stage2,
        verbose=1
    )
    histories.append(history2.history)
    
    return model, histories


def evaluate_model(model, test_data, class_names):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Evaluate on test generator
    X_test = []
    y_test = []
    
    # Collect all test data
    steps = len(test_data)
    for i in range(steps):
        X_batch, y_batch = next(test_data)
        X_test.append(X_batch)
        y_test.append(y_batch)
    
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_true_classes == y_pred_classes)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print("\nPer-Class Performance:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("IMPROVED CNN MODEL TRAINING")
    print("Targeting 80%+ accuracy with research-based improvements")
    print("="*60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_classes': 7
    }
    
    # Class names
    class_names = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew', 
                   'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    
    try:
        # Load data
        print("\n1. Loading data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
        
        # Apply CutMix augmentation to training data
        print("\n2. Applying CutMix augmentation...")
        X_train_aug, y_train_aug = cutmix(X_train, y_train, alpha=1.0, prob=0.3)
        
        # Create data generators
        print("\n3. Creating data generators...")
        train_gen = create_augmented_generator(X_train_aug, y_train_aug, config['batch_size'], training=True)
        val_gen = create_augmented_generator(X_val, y_val, config['batch_size'], training=False)
        
        # Build model
        print("\n4. Building improved CNN model...")
        model = build_improved_cnn(config['num_classes'])
        model.summary()
        
        # Progressive training
        print("\n5. Starting progressive training...")
        start_time = time.time()
        
        model, histories = train_model_progressive(model, train_gen, val_gen, config)
        
        training_time = time.time() - start_time
        print(f"\nTotal training time: {training_time/60:.2f} minutes")
        
        # Evaluate on test set if available
        if X_test is not None:
            print("\n6. Evaluating on test set...")
            test_gen = create_augmented_generator(X_test, y_test, config['batch_size'], training=False)
            test_accuracy = evaluate_model(model, test_gen, class_names)
        
        # Save final model
        print("\n7. Saving final model...")
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f'cnn_improved_{timestamp}.h5'
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training history
        history_path = model_dir / f'history_cnn_{timestamp}.json'
        with open(history_path, 'w') as f:
            # Combine histories
            combined_history = {}
            for i, history in enumerate(histories):
                for key, values in history.items():
                    stage_key = f"stage{i+1}_{key}"
                    combined_history[stage_key] = [float(v) for v in values]
            json.dump(combined_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        # Get best validation accuracy
        all_val_acc = []
        for history in histories:
            if 'val_accuracy' in history:
                all_val_acc.extend(history['val_accuracy'])
        
        if all_val_acc:
            best_val_acc = max(all_val_acc)
            print(f"Best validation accuracy: {best_val_acc:.2%}")
            
            if best_val_acc >= 0.80:
                print("✓ Successfully achieved target accuracy of 80%+!")
            else:
                print(f"Current best accuracy: {best_val_acc:.2%}")
                print("\nSuggestions for improvement:")
                print("  - Train for more epochs")
                print("  - Adjust learning rate schedule")
                print("  - Increase CutMix probability")
                print("  - Add more data augmentation")
        
        if X_test is not None and test_accuracy >= 0.80:
            print(f"\n✓ Test accuracy: {test_accuracy:.2%} - Target achieved!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())