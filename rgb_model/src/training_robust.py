"""
Robust training pipeline with advanced techniques for 80%+ accuracy
"""

import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
from typing import Dict, Tuple, List, Optional
import os
import json
from datetime import datetime
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Note: wandb not installed. Training will proceed without experiment tracking.")

from model_robust import RobustDiseaseDetector, create_robust_model
from preprocessing_simple import CrossCropPreprocessor, MixUpAugmentation, CutMixAugmentation


class RobustProgressiveTrainer:
    """
    Enhanced progressive training with advanced techniques
    """
    
    def __init__(self, model_config: Dict, training_config: Dict, 
                 output_dir: str = './models'):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = None  # Will be created with class weights
        self.mixup = MixUpAugmentation(alpha=training_config.get('mixup_alpha', 0.2))
        self.cutmix = CutMixAugmentation(alpha=training_config.get('cutmix_alpha', 1.0))
        
        # Training history
        self.history = {'stage1': None, 'stage2': None, 'stage3': None}
        
        # Track best metrics
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        
    def initialize_model(self, y_train: np.ndarray):
        """Initialize model with class weights calculated from training data"""
        self.detector = create_robust_model(
            num_classes=self.model_config.get('num_classes', 8),
            input_shape=self.model_config.get('input_shape', (224, 224, 3)),
            y_train=y_train
        )
        
    def create_advanced_data_generator(self, x_data: np.ndarray, y_data: np.ndarray,
                                     batch_size: int, is_training: bool = True):
        """
        Creates an advanced data generator with multiple augmentation strategies
        """
        def data_generator():
            indices = np.arange(len(x_data))
            
            while True:
                if is_training:
                    np.random.shuffle(indices)
                
                for start_idx in range(0, len(indices), batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    batch_x = x_data[batch_indices].copy()
                    batch_y = y_data[batch_indices].copy()
                    
                    if is_training:
                        # Apply MixUp or CutMix randomly
                        augment_choice = np.random.random()
                        
                        if augment_choice < 0.25 and len(batch_x) > 1:
                            # Apply MixUp
                            batch_x, batch_y = self.mixup.mixup_batch(batch_x, batch_y)
                        elif augment_choice < 0.5 and len(batch_x) > 1:
                            # Apply CutMix
                            for i in range(0, len(batch_x) - 1, 2):
                                batch_x[i], batch_y[i] = self.cutmix.cutmix(
                                    batch_x[i], batch_y[i],
                                    batch_x[i+1], batch_y[i+1]
                                )
                        
                        # Additional augmentations
                        for i in range(len(batch_x)):
                            if np.random.random() < 0.3:
                                # Random brightness
                                factor = np.random.uniform(0.8, 1.2)
                                batch_x[i] = np.clip(batch_x[i] * factor, 0, 1)
                            
                            if np.random.random() < 0.2:
                                # Add Gaussian noise
                                noise = np.random.normal(0, 0.02, batch_x[i].shape)
                                batch_x[i] = np.clip(batch_x[i] + noise, 0, 1)
                    
                    yield batch_x, batch_y
        
        return data_generator()
    
    def get_advanced_callbacks(self, stage: str, monitor: str = 'val_loss'):
        """
        Returns advanced callbacks including learning rate scheduling
        """
        stage_dir = self.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        callback_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(stage_dir / 'best_model.weights.h5'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            
            # Early stopping with patience
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.training_config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1,
                min_delta=0.001
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=self.training_config.get('reduce_lr_patience', 5),
                min_lr=1e-7,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                filename=str(stage_dir / 'training_log.csv'),
                append=True
            ),
            
            # TensorBoard with profiling
            callbacks.TensorBoard(
                log_dir=str(stage_dir / 'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch',
                profile_batch=0
            ),
            
            # Custom callback for tracking best metrics
            BestMetricsCallback(self)
        ]
        
        # Add learning rate warmup for stage 1
        if stage == 'stage1':
            warmup_epochs = self.training_config.get('warmup_epochs', 3)
            initial_lr = self.training_config.get('stage1_lr', 0.001)
            callback_list.append(
                WarmupLearningRateScheduler(
                    initial_lr=initial_lr,
                    warmup_epochs=warmup_epochs
                )
            )
        
        return callback_list
    
    def train_stage_1(self, train_data: Tuple, val_data: Tuple, 
                     epochs: int = 15, batch_size: int = 32):
        """
        Stage 1: Feature extraction with frozen backbone and warmup
        """
        print("\n" + "="*50)
        print("STAGE 1: Feature Extraction with Learning Rate Warmup")
        print("="*50)
        
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        # Initialize model if not already done
        if self.detector is None:
            self.initialize_model(y_train)
        
        # Compile with stage 1 settings
        self.detector.compile_model(
            learning_rate=self.training_config.get('stage1_lr', 0.001),
            use_focal_loss=self.training_config.get('use_focal_loss', True),
            focal_alpha=self.training_config.get('focal_alpha', 0.75),
            focal_gamma=self.training_config.get('focal_gamma', 2.0),
            label_smoothing=0.1
        )
        
        # Ensure base model is frozen
        if self.detector.base_model is not None:
            self.detector.base_model.trainable = False
        
        # Create data generators
        train_gen = self.create_advanced_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_advanced_data_generator(x_val, y_val, batch_size, False)
        
        # Train
        history = self.detector.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=self.get_advanced_callbacks('stage1'),
            verbose=1
        )
        
        self.history['stage1'] = history.history
        
        # Load best weights
        self.detector.model.load_weights(str(self.output_dir / 'stage1' / 'best_model.weights.h5'))
        
        print(f"\nStage 1 completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history
    
    def train_stage_2(self, train_data: Tuple, val_data: Tuple,
                     epochs: int = 10, batch_size: int = 32,
                     num_layers_to_unfreeze: int = 30):
        """
        Stage 2: Partial fine-tuning with increased augmentation
        """
        print("\n" + "="*50)
        print(f"STAGE 2: Partial Fine-tuning (Top {num_layers_to_unfreeze} layers)")
        print("="*50)
        
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        # Unfreeze top layers
        self.detector.unfreeze_layers(num_layers_to_unfreeze)
        
        # Recompile with lower learning rate
        self.detector.compile_model(
            learning_rate=self.training_config.get('stage2_lr', 0.0001),
            use_focal_loss=self.training_config.get('use_focal_loss', True),
            focal_alpha=self.training_config.get('focal_alpha', 0.75),
            focal_gamma=self.training_config.get('focal_gamma', 2.0),
            label_smoothing=0.1
        )
        
        # Increase augmentation strength
        self.mixup.alpha = 0.3
        self.cutmix.alpha = 1.5
        
        # Create data generators
        train_gen = self.create_advanced_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_advanced_data_generator(x_val, y_val, batch_size, False)
        
        # Train
        history = self.detector.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=self.get_advanced_callbacks('stage2'),
            verbose=1
        )
        
        self.history['stage2'] = history.history
        
        # Load best weights
        self.detector.model.load_weights(str(self.output_dir / 'stage2' / 'best_model.weights.h5'))
        
        print(f"\nStage 2 completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history
    
    def train_stage_3(self, train_data: Tuple, val_data: Tuple,
                     epochs: int = 5, batch_size: int = 32):
        """
        Stage 3: Full fine-tuning with cosine annealing
        """
        print("\n" + "="*50)
        print("STAGE 3: Full Fine-tuning with Cosine Annealing")
        print("="*50)
        
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        # Unfreeze all layers
        self.detector.unfreeze_layers(-1)
        
        # Recompile with very low learning rate
        self.detector.compile_model(
            learning_rate=self.training_config.get('stage3_lr', 0.00001),
            use_focal_loss=self.training_config.get('use_focal_loss', True),
            focal_alpha=self.training_config.get('focal_alpha', 0.75),
            focal_gamma=self.training_config.get('focal_gamma', 2.0),
            label_smoothing=0.05  # Reduce label smoothing
        )
        
        # Create data generators
        train_gen = self.create_advanced_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_advanced_data_generator(x_val, y_val, batch_size, False)
        
        # Add cosine annealing
        cosine_scheduler = callbacks.LearningRateScheduler(
            lambda epoch: self.training_config.get('stage3_lr', 0.00001) * 
                        (1 + np.cos(np.pi * epoch / epochs)) / 2
        )
        
        # Get callbacks and add cosine scheduler
        callback_list = self.get_advanced_callbacks('stage3')
        callback_list.append(cosine_scheduler)
        
        # Train
        history = self.detector.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history['stage3'] = history.history
        
        # Load best weights
        self.detector.model.load_weights(str(self.output_dir / 'stage3' / 'best_model.weights.h5'))
        
        print(f"\nStage 3 completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
        
        return history
    
    def train_progressive(self, train_data: Tuple, val_data: Tuple,
                         stage1_epochs: int = 15,
                         stage2_epochs: int = 10,
                         stage3_epochs: int = 5,
                         batch_size: int = 32):
        """
        Runs all three training stages progressively
        """
        print("\nStarting Robust Progressive Training")
        print(f"Total epochs: {stage1_epochs + stage2_epochs + stage3_epochs}")
        print(f"Target: 80%+ validation accuracy")
        
        # Stage 1
        self.train_stage_1(train_data, val_data, stage1_epochs, batch_size)
        
        # Stage 2
        self.train_stage_2(train_data, val_data, stage2_epochs, batch_size)
        
        # Stage 3
        self.train_stage_3(train_data, val_data, stage3_epochs, batch_size)
        
        # Save final model
        self.save_final_model()
        
        # Generate comprehensive report
        self.generate_training_report()
        
        # Print final results
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.2%}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        if self.best_val_accuracy >= 0.80:
            print("\n✅ TARGET ACHIEVED: 80%+ validation accuracy!")
        else:
            print(f"\n⚠️  Target not reached. Current: {self.best_val_accuracy:.2%}, Target: 80%")
            print("Consider: 1) More training data, 2) Longer training, 3) GPU acceleration")
        
        return self.history
    
    def save_final_model(self):
        """Saves the final trained model in multiple formats"""
        final_dir = self.output_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in Keras format
        self.detector.save_model(str(final_dir / 'model'))
        
        # Save weights only
        self.detector.model.save_weights(str(final_dir / 'model.weights.h5'))
        
        # Save model architecture
        with open(final_dir / 'model_architecture.json', 'w') as f:
            json.dump(json.loads(self.detector.model.to_json()), f, indent=2)
        
        print(f"\nFinal model saved to {final_dir}")
    
    def generate_training_report(self):
        """Generates a comprehensive training report"""
        report_path = self.output_dir / 'training_report.json'
        
        report = {
            'training_completed': datetime.now().isoformat(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'best_metrics': {
                'validation_accuracy': float(self.best_val_accuracy),
                'validation_loss': float(self.best_val_loss)
            },
            'stages': {}
        }
        
        # Summarize each stage
        for stage, history in self.history.items():
            if history:
                report['stages'][stage] = {
                    'epochs': len(history['loss']),
                    'final_loss': float(history['loss'][-1]),
                    'final_val_loss': float(history['val_loss'][-1]),
                    'best_val_loss': float(min(history['val_loss'])),
                    'final_accuracy': float(history['accuracy'][-1]),
                    'final_val_accuracy': float(history['val_accuracy'][-1]),
                    'best_val_accuracy': float(max(history['val_accuracy']))
                }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTraining report saved to {report_path}")


class BestMetricsCallback(callbacks.Callback):
    """Tracks best metrics across all training stages"""
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            val_acc = logs.get('val_accuracy', 0)
            val_loss = logs.get('val_loss', float('inf'))
            
            if val_acc > self.trainer.best_val_accuracy:
                self.trainer.best_val_accuracy = val_acc
                
            if val_loss < self.trainer.best_val_loss:
                self.trainer.best_val_loss = val_loss


class WarmupLearningRateScheduler(callbacks.Callback):
    """Learning rate warmup scheduler"""
    
    def __init__(self, initial_lr: float, warmup_epochs: int):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            
            if logs is not None:
                logs['lr'] = lr


def test_robust_training():
    """Test the robust training pipeline"""
    # Model configuration
    model_config = {
        'num_classes': 7,
        'input_shape': (224, 224, 3)
    }
    
    # Training configuration
    training_config = {
        'use_focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'stage1_lr': 0.001,
        'stage2_lr': 0.0001,
        'stage3_lr': 0.00001,
        'warmup_epochs': 3,
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5
    }
    
    # Create dummy data
    num_samples = 100
    x_train = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 7, num_samples), 
        num_classes=7
    )
    
    x_val = np.random.rand(20, 224, 224, 3).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 7, 20), 
        num_classes=7
    )
    
    # Create trainer
    trainer = RobustProgressiveTrainer(model_config, training_config)
    
    # Run progressive training (with very few epochs for testing)
    trainer.train_progressive(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        stage1_epochs=2,
        stage2_epochs=2,
        stage3_epochs=1,
        batch_size=16
    )
    
    print("\nRobust progressive training test completed!")


if __name__ == "__main__":
    test_robust_training()