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

from model import UniversalDiseaseDetector, ModelMetrics
# Use simple preprocessing to avoid albumentations issues
from preprocessing_simple import CrossCropPreprocessor, MixUpAugmentation, CutMixAugmentation


class ProgressiveTrainer:
    """
    Implements progressive three-stage training for optimal performance.
    Stage 1: Feature extraction (frozen backbone)
    Stage 2: Partial fine-tuning (top 20 layers)
    Stage 3: Full fine-tuning (all layers)
    """
    
    def __init__(self, model_config: Dict, training_config: Dict, 
                 output_dir: str = './models'):
        """
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            output_dir: Directory to save models and logs
        """
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.detector = UniversalDiseaseDetector(
            num_classes=model_config.get('num_classes', 8),
            input_shape=model_config.get('input_shape', (224, 224, 3)),
            dropout_rate=model_config.get('dropout_rate', 0.5),
            l2_regularization=model_config.get('l2_regularization', 0.001)
        )
        
        # Initialize augmentation
        self.mixup = MixUpAugmentation(alpha=training_config.get('mixup_alpha', 0.2))
        self.cutmix = CutMixAugmentation(alpha=training_config.get('cutmix_alpha', 1.0))
        
        # Training history
        self.history = {'stage1': None, 'stage2': None, 'stage3': None}
        
    def create_data_generator(self, x_data: np.ndarray, y_data: np.ndarray,
                            batch_size: int, is_training: bool = True):
        """
        Creates a data generator with advanced augmentation.
        Applies MixUp and CutMix during training for better generalization.
        """
        def data_generator():
            indices = np.arange(len(x_data))
            
            while True:
                if is_training:
                    np.random.shuffle(indices)
                
                for start_idx in range(0, len(indices), batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    batch_x = x_data[batch_indices].copy()  # Copy to avoid modifying original
                    batch_y = y_data[batch_indices].copy()
                    
                    if is_training:
                        # Apply augmentation with probability
                        augmentation_prob = np.random.random()
                        
                        if augmentation_prob < 0.25 and self.training_config.get('use_mixup', True):
                            # Apply MixUp (25% chance)
                            batch_x, batch_y = self.mixup.mixup_batch(batch_x, batch_y)
                            
                        elif augmentation_prob < 0.5 and self.training_config.get('use_cutmix', True):
                            # Apply CutMix (25% chance)
                            # Shuffle for pairing
                            shuffle_indices = np.random.permutation(len(batch_x))
                            
                            for i in range(0, len(batch_x), 2):
                                if i + 1 < len(batch_x):
                                    batch_x[i], batch_y[i] = self.cutmix.cutmix(
                                        batch_x[i], batch_y[i],
                                        batch_x[shuffle_indices[i]], batch_y[shuffle_indices[i]]
                                    )
                        
                        # 50% chance of no augmentation (original images)
                    
                    yield batch_x, batch_y
        
        return data_generator()
    
    def get_callbacks(self, stage: str, monitor: str = 'val_loss'):
        """
        Returns callbacks for a training stage.
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
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.training_config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            
            # Reduce learning rate
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
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=str(stage_dir / 'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Add custom metrics callback if validation data provided
        if hasattr(self, 'validation_data') and self.validation_data is not None:
            callback_list.append(
                ModelMetrics(
                    self.validation_data,
                    self.model_config.get('class_names', [])
                )
            )
        
        return callback_list
    
    def train_stage_1(self, train_data: Tuple, val_data: Tuple, 
                     epochs: int = 15, batch_size: int = 32):
        """
        Stage 1: Feature extraction with frozen backbone.
        """
        print("\n" + "="*50)
        print("STAGE 1: Feature Extraction (Frozen Backbone)")
        print("="*50)
        
        x_train, y_train = train_data
        x_val, y_val = val_data
        self.validation_data = val_data
        
        # Compile model with high learning rate
        self.detector.compile_model(
            learning_rate=self.training_config.get('stage1_lr', 0.001),
            use_focal_loss=self.training_config.get('use_focal_loss', True),
            focal_alpha=self.training_config.get('focal_alpha', 0.75),
            focal_gamma=self.training_config.get('focal_gamma', 2.0)
        )
        
        # Ensure base model is frozen (if using EfficientNet)
        if self.detector.base_model is not None:
            self.detector.base_model.trainable = False
        else:
            print("Using custom CNN architecture - no base model to freeze")
        
        # Create data generators
        train_gen = self.create_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_data_generator(x_val, y_val, batch_size, False)
        
        # Train
        history = self.detector.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=self.get_callbacks('stage1'),
            verbose=1
        )
        
        self.history['stage1'] = history.history
        
        # Load best weights
        self.detector.model.load_weights(str(self.output_dir / 'stage1' / 'best_model.weights.h5'))
        
        print(f"\nStage 1 completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        
        return history
    
    def train_stage_2(self, train_data: Tuple, val_data: Tuple,
                     epochs: int = 10, batch_size: int = 32,
                     num_layers_to_unfreeze: int = 20):
        """
        Stage 2: Partial fine-tuning of top layers.
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
            focal_gamma=self.training_config.get('focal_gamma', 2.0)
        )
        
        # Create data generators with more augmentation
        train_gen = self.create_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_data_generator(x_val, y_val, batch_size, False)
        
        # Train
        history = self.detector.model.fit(
            train_gen,
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(x_val) // batch_size,
            callbacks=self.get_callbacks('stage2'),
            verbose=1
        )
        
        self.history['stage2'] = history.history
        
        # Load best weights
        self.detector.model.load_weights(str(self.output_dir / 'stage2' / 'best_model.weights.h5'))
        
        print(f"\nStage 2 completed. Best val_loss: {min(history.history['val_loss']):.4f}")
        
        return history
    
    def train_stage_3(self, train_data: Tuple, val_data: Tuple,
                     epochs: int = 5, batch_size: int = 32):
        """
        Stage 3: Full fine-tuning of all layers.
        """
        print("\n" + "="*50)
        print("STAGE 3: Full Fine-tuning (All layers)")
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
            focal_gamma=self.training_config.get('focal_gamma', 2.0)
        )
        
        # Create data generators
        train_gen = self.create_data_generator(x_train, y_train, batch_size, True)
        val_gen = self.create_data_generator(x_val, y_val, batch_size, False)
        
        # Train with cosine annealing
        cosine_scheduler = callbacks.LearningRateScheduler(
            lambda epoch: self.training_config.get('stage3_lr', 0.00001) * 
                        (1 + np.cos(np.pi * epoch / epochs)) / 2
        )
        
        # Get callbacks and add cosine scheduler
        callback_list = self.get_callbacks('stage3')
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
        
        return history
    
    def train_progressive(self, train_data: Tuple, val_data: Tuple,
                         stage1_epochs: int = 15,
                         stage2_epochs: int = 10,
                         stage3_epochs: int = 5,
                         batch_size: int = 32):
        """
        Runs all three training stages progressively.
        """
        print("\nStarting Progressive Training")
        print(f"Total epochs: {stage1_epochs + stage2_epochs + stage3_epochs}")
        
        # Stage 1
        self.train_stage_1(train_data, val_data, stage1_epochs, batch_size)
        
        # Stage 2
        self.train_stage_2(train_data, val_data, stage2_epochs, batch_size)
        
        # Stage 3
        self.train_stage_3(train_data, val_data, stage3_epochs, batch_size)
        
        # Save final model
        self.save_final_model()
        
        # Generate training report
        self.generate_training_report()
        
        return self.history
    
    def save_final_model(self):
        """Saves the final trained model in multiple formats."""
        final_dir = self.output_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in SavedModel format
        self.detector.save_model(str(final_dir / 'saved_model'))
        
        # Save weights only
        self.detector.model.save_weights(str(final_dir / 'model.weights.h5'))
        
        # Save model architecture
        with open(final_dir / 'model_architecture.json', 'w') as f:
            json.dump(json.loads(self.detector.model.to_json()), f, indent=2)
        
        print(f"\nFinal model saved to {final_dir}")
    
    def generate_training_report(self):
        """Generates a comprehensive training report."""
        report_path = self.output_dir / 'training_report.json'
        
        report = {
            'training_completed': datetime.now().isoformat(),
            'model_config': self.model_config,
            'training_config': self.training_config,
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
    
    def evaluate_model(self, test_data: Tuple, batch_size: int = 32):
        """
        Evaluates the model on test data.
        """
        x_test, y_test = test_data
        
        # Create test generator
        test_gen = self.create_data_generator(x_test, y_test, batch_size, False)
        
        # Evaluate
        results = self.detector.model.evaluate(
            test_gen,
            steps=len(x_test) // batch_size,
            verbose=1
        )
        
        # Create results dictionary
        metric_names = self.detector.model.metrics_names
        evaluation_results = dict(zip(metric_names, results))
        
        # Save evaluation results
        eval_path = self.output_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\nEvaluation results saved to {eval_path}")
        
        return evaluation_results


class WarmupCosineScheduler(callbacks.Callback):
    """
    Custom learning rate scheduler with warmup and cosine annealing.
    """
    
    def __init__(self, base_lr: float, warmup_epochs: int, total_epochs: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        
        if logs is not None:
            logs['lr'] = lr


def test_progressive_training():
    """Test the progressive training pipeline."""
    # Model configuration
    model_config = {
        'num_classes': 7,
        'input_shape': (224, 224, 3),
        'dropout_rate': 0.5,
        'l2_regularization': 0.001,
        'class_names': ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew',
                       'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
    }
    
    # Training configuration
    training_config = {
        'use_focal_loss': True,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'stage1_lr': 0.001,
        'stage2_lr': 0.0001,
        'stage3_lr': 0.00001,
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
        num_categories=7
    )
    
    x_val = np.random.rand(20, 224, 224, 3).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(
        np.random.randint(0, 7, 20), 
        num_categories=7
    )
    
    # Create trainer
    trainer = ProgressiveTrainer(model_config, training_config)
    
    # Run progressive training (with very few epochs for testing)
    trainer.train_progressive(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        stage1_epochs=2,
        stage2_epochs=2,
        stage3_epochs=1,
        batch_size=16
    )
    
    print("\nProgressive training test completed!")


if __name__ == "__main__":
    test_progressive_training()