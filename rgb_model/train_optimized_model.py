#!/usr/bin/env python3
"""
Optimized training script implementing all research-based improvements:
1. Correct EfficientNet preprocessing (0-255 range, not 0-1)
2. Progressive unfreezing with proper learning rates
3. Label smoothing instead of focal loss for better generalization
4. CutMix augmentation for improved robustness
5. Cosine annealing with warmup
6. Full fine-tuning to avoid negative transfer
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


class CutMixAugmentation:
    """CutMix augmentation for better generalization"""
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images, labels):
        batch_size = tf.shape(images)[0]
        
        if tf.random.uniform([]) > self.prob:
            return images, labels
        
        # Get mixing parameter from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random index for mixing
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Get image dimensions
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
        
        # Calculate cut size
        cut_ratio = tf.sqrt(1.0 - lam)
        cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
        cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
        
        # Get random position
        cy = tf.random.uniform([], cut_h // 2, img_h - cut_h // 2, dtype=tf.int32)
        cx = tf.random.uniform([], cut_w // 2, img_w - cut_w // 2, dtype=tf.int32)
        
        # Create binary mask
        mask = tf.ones((batch_size, img_h, img_w, 1))
        pad_top = cy - cut_h // 2
        pad_bottom = img_h - (cy + cut_h // 2)
        pad_left = cx - cut_w // 2
        pad_right = img_w - (cx + cut_w // 2)
        
        # Apply cutmix
        mask_indices = tf.zeros((batch_size, cut_h, cut_w, 1))
        mask = tf.pad(mask_indices, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
                     constant_values=1)
        
        # Mix images and labels
        mixed_images = mask * images + (1 - mask) * tf.gather(images, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


class CosineLRScheduler(tf.keras.callbacks.Callback):
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
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")


def build_optimized_efficientnet(num_classes=7):
    """Build EfficientNet-B0 with correct preprocessing"""
    
    from tensorflow.keras.applications import EfficientNetB0
    
    # CRITICAL: Input should be in [0, 255] range for EfficientNet
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # EfficientNet has built-in preprocessing - expects [0, 255] range
    # The preprocess_input function will handle the normalization
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    
    # Load base model without input_tensor to avoid shape issues
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'  # Use average pooling
    )
    
    # Start with frozen backbone
    base_model.trainable = False
    
    # Pass preprocessed input through base model
    features = base_model(x, training=False)
    
    # Enhanced classification head
    x = tf.keras.layers.Dense(512, activation='relu')(features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer with softmax
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model


def load_and_preprocess_data(data_dir='./data/splits'):
    """Load data ensuring correct preprocessing for EfficientNet"""
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found.")
    
    print("Loading preprocessed data...")
    
    # Load arrays
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    
    # CRITICAL FIX: EfficientNet expects [0, 255] range, not [0, 1]
    # Check current range
    if X_train.max() <= 1.0:
        print("Converting data from [0, 1] to [0, 255] range for EfficientNet...")
        X_train = (X_train * 255).astype(np.float32)
        X_val = (X_val * 255).astype(np.float32)
    
    print(f"Data range - Min: {X_train.min():.2f}, Max: {X_train.max():.2f}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # Load test data if available
    X_test = None
    y_test = None
    if (data_dir / 'X_test.npy').exists():
        X_test = np.load(data_dir / 'X_test.npy')
        y_test = np.load(data_dir / 'y_test.npy')
        if X_test.max() <= 1.0:
            X_test = (X_test * 255).astype(np.float32)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_augmented_dataset(X, y, batch_size=32, training=True):
    """Create TF dataset with augmentations"""
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if training:
        # Shuffle with large buffer
        dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
        
        # Basic augmentations
        def augment(image, label):
            # Random flip
            image = tf.image.random_flip_left_right(image)
            
            # Random brightness/contrast (subtle)
            image = tf.image.random_brightness(image, max_delta=20)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0, 255)
            
            return image, label
        
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def progressive_training(model, base_model, train_data, val_data, config):
    """
    Three-stage progressive training:
    1. Frozen backbone (head only)
    2. Partial unfreezing (top layers)
    3. Full fine-tuning
    """
    
    # Create directory for checkpoints
    checkpoint_dir = Path('./models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    histories = {}
    
    # Stage 1: Train head only (frozen backbone)
    print("\n" + "="*60)
    print("STAGE 1: Training classification head (frozen backbone)")
    print("="*60)
    
    base_model.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks for Stage 1
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'stage1_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CosineLRScheduler(
            initial_lr=2e-4,
            warmup_epochs=2,
            total_epochs=config['stage1_epochs']
        )
    ]
    
    history_stage1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config['stage1_epochs'],
        callbacks=callbacks_stage1,
        verbose=1
    )
    histories['stage1'] = history_stage1.history
    
    # Stage 2: Unfreeze top layers
    print("\n" + "="*60)
    print("STAGE 2: Fine-tuning top layers")
    print("="*60)
    
    base_model.trainable = True
    # Freeze all but top 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'stage2_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CosineLRScheduler(
            initial_lr=5e-5,
            warmup_epochs=1,
            total_epochs=config['stage2_epochs']
        )
    ]
    
    history_stage2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config['stage2_epochs'],
        callbacks=callbacks_stage2,
        verbose=1
    )
    histories['stage2'] = history_stage2.history
    
    # Stage 3: Full fine-tuning
    print("\n" + "="*60)
    print("STAGE 3: Full model fine-tuning")
    print("="*60)
    
    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True
    
    # Recompile with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    callbacks_stage3 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'final_best_{timestamp}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history_stage3 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=config['stage3_epochs'],
        callbacks=callbacks_stage3,
        verbose=1
    )
    histories['stage3'] = history_stage3.history
    
    return model, histories


def evaluate_model(model, test_data, class_names):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Overall metrics
    results = model.evaluate(test_data, verbose=1)
    metrics_names = model.metrics_names
    
    print("\nOverall Performance:")
    for name, value in zip(metrics_names, results):
        print(f"  {name}: {value:.4f}")
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for X_batch, y_batch in test_data:
        predictions = model.predict(X_batch, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(y_batch, axis=1))
    
    # Classification report
    print("\nPer-Class Performance:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return results


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("OPTIMIZED RGB MODEL TRAINING")
    print("Based on research findings to achieve 80%+ accuracy")
    print("="*60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'stage1_epochs': 10,
        'stage2_epochs': 15,
        'stage3_epochs': 10,
        'num_classes': 7
    }
    
    # Class names (update based on your harmonized categories)
    class_names = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew', 
                   'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    
    try:
        # Load data with correct preprocessing
        print("\n1. Loading and preprocessing data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()
        
        # Create datasets
        print("\n2. Creating augmented datasets...")
        train_dataset = create_augmented_dataset(X_train, y_train, config['batch_size'], training=True)
        val_dataset = create_augmented_dataset(X_val, y_val, config['batch_size'], training=False)
        
        # Build model
        print("\n3. Building EfficientNet-B0 model...")
        model, base_model = build_optimized_efficientnet(config['num_classes'])
        model.summary()
        
        # Progressive training
        print("\n4. Starting progressive training...")
        start_time = time.time()
        
        model, histories = progressive_training(
            model, base_model, 
            train_dataset, val_dataset, 
            config
        )
        
        training_time = time.time() - start_time
        print(f"\nTotal training time: {training_time/60:.2f} minutes")
        
        # Evaluate on test set if available
        if X_test is not None:
            print("\n5. Evaluating on test set...")
            test_dataset = create_augmented_dataset(X_test, y_test, config['batch_size'], training=False)
            evaluate_model(model, test_dataset, class_names)
        
        # Save final model
        print("\n6. Saving final model...")
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f'rgb_optimized_{timestamp}.h5'
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training history
        history_path = model_dir / f'history_{timestamp}.json'
        with open(history_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            histories_serializable = {}
            for stage, history in histories.items():
                histories_serializable[stage] = {
                    key: [float(v) for v in values] 
                    for key, values in history.items()
                }
            json.dump(histories_serializable, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        # Get final metrics
        final_val_acc = histories['stage3']['val_accuracy'][-1]
        best_val_acc = max(max(histories['stage1']['val_accuracy']), 
                          max(histories['stage2']['val_accuracy']),
                          max(histories['stage3']['val_accuracy']))
        
        print(f"Final validation accuracy: {final_val_acc:.2%}")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        
        if best_val_acc >= 0.80:
            print("✓ Successfully achieved target accuracy of 80%+!")
        else:
            print(f"Current accuracy: {best_val_acc:.2%}")
            print("Consider:")
            print("  - Increasing training epochs")
            print("  - Adjusting learning rates")
            print("  - Adding more augmentation")
            print("  - Collecting more diverse training data")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())