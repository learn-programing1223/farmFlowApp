"""
Robust Universal Plant Disease Detection Model using MobileNetV3
Designed for high accuracy and mobile deployment
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
from typing import Tuple, Dict, Optional


class ImprovedFocalLoss(tf.keras.losses.Loss):
    """
    Improved Focal Loss with better numerical stability
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 class_weights: Optional[Dict[int, float]] = None,
                 from_logits: bool = False, 
                 label_smoothing: float = 0.1,
                 name: str = 'improved_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.epsilon = tf.keras.backend.epsilon()
        
    def call(self, y_true, y_pred):
        # Apply label smoothing
        num_classes = tf.shape(y_true)[-1]
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(num_classes, tf.float32)
        
        # Ensure numerical stability
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1. - self.epsilon)
        
        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply class weights if provided
        if self.class_weights:
            class_weight_tensor = tf.constant([self.class_weights.get(i, 1.0) 
                                             for i in range(num_classes)])
            ce = ce * class_weight_tensor
        
        # Apply alpha balancing
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


class RobustDiseaseDetector:
    """
    Robust plant disease detector using MobileNetV3 with advanced features
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 dropout_rate: float = 0.4,
                 l2_regularization: float = 0.0005,
                 use_mixed_precision: bool = False):
        """
        Initialize the robust disease detector
        
        Args:
            num_classes: Number of disease categories
            input_shape: Input image shape
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
            use_mixed_precision: Whether to use mixed precision training
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        
        if use_mixed_precision:
            set_global_policy('mixed_float16')
        
        self.model = self._build_model()
        self.base_model = None  # Will be set after building
        
    def _build_model(self) -> Model:
        """Build the robust model architecture"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Data augmentation layers (only active during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # Preprocessing for MobileNetV3
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        
        # Base model - MobileNetV3Small for efficiency and reliability
        base_model = MobileNetV3Small(
            input_tensor=x,
            weights='imagenet',
            include_top=False,
            pooling=None,
            include_preprocessing=False
        )
        
        # Freeze base model initially
        base_model.trainable = False
        self.base_model = base_model
        
        # Get features from base model
        x = base_model.output
        
        # Custom top layers with SE blocks
        # MobileNetV3Small outputs 576 channels
        x = self._add_se_block(x, filters=576, name='se_block_1')
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Multi-scale feature fusion
        # Also get features from an intermediate layer
        # Using the correct layer name for MobileNetV3Small
        intermediate_layer = base_model.get_layer('expanded_conv_10_project_bn')
        intermediate_features = intermediate_layer.output
        intermediate_features = layers.GlobalAveragePooling2D()(intermediate_features)
        intermediate_features = layers.Dense(256, activation='relu')(intermediate_features)
        
        # Concatenate multi-scale features
        x = layers.concatenate([x, intermediate_features])
        
        # Classification head with batch norm
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='dense_1'
        )(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.7)(x)
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='dense_2'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='predictions'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='robust_disease_detector')
        
        return model
    
    def _add_se_block(self, x, filters: int, ratio: int = 16, name: str = 'se'):
        """Add Squeeze-and-Excitation block for channel attention"""
        
        # Squeeze
        se = layers.GlobalAveragePooling2D(name=f'{name}_squeeze')(x)
        
        # Excitation
        se = layers.Dense(filters // ratio, activation='relu', name=f'{name}_reduce')(se)
        se = layers.Dense(filters, activation='sigmoid', name=f'{name}_expand')(se)
        
        # Scale
        se = layers.Reshape((1, 1, filters), name=f'{name}_reshape')(se)
        x = layers.multiply([x, se], name=f'{name}_multiply')
        
        return x
    
    def compile_model(self,
                     learning_rate: float = 0.001,
                     use_focal_loss: bool = True,
                     focal_alpha: float = 0.75,
                     focal_gamma: float = 2.0,
                     class_weights: Optional[Dict[int, float]] = None,
                     label_smoothing: float = 0.1,
                     use_ema: bool = True):
        """
        Compile the model with advanced optimization settings
        
        Args:
            learning_rate: Initial learning rate
            use_focal_loss: Whether to use focal loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            class_weights: Dictionary of class weights for imbalanced data
            label_smoothing: Label smoothing factor
            use_ema: Whether to use Exponential Moving Average
        """
        
        # Advanced optimizer with lookahead
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Apply EMA if requested
        if use_ema:
            optimizer = tf.keras.optimizers.EMA(optimizer, average_decay=0.999)
        
        # Loss function
        if use_focal_loss:
            loss = ImprovedFocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=class_weights,
                label_smoothing=label_smoothing
            )
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            )
        
        # Comprehensive metrics
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets
        
        Args:
            y_train: One-hot encoded training labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Convert one-hot to class indices
        y_indices = np.argmax(y_train, axis=1)
        
        # Calculate class frequencies
        unique, counts = np.unique(y_indices, return_counts=True)
        
        # Calculate weights (inverse frequency)
        total_samples = len(y_indices)
        n_classes = len(unique)
        
        class_weights = {}
        for cls, count in zip(unique, counts):
            weight = total_samples / (n_classes * count)
            class_weights[cls] = weight
            
        # Normalize weights
        max_weight = max(class_weights.values())
        for cls in class_weights:
            class_weights[cls] /= max_weight
            
        print("\nClass weights:")
        for cls, weight in class_weights.items():
            print(f"  Class {cls}: {weight:.3f}")
            
        return class_weights
    
    def unfreeze_layers(self, num_layers: int = -1):
        """Unfreeze layers for fine-tuning"""
        
        if self.base_model is None:
            print("No base model to unfreeze")
            return
            
        if num_layers == -1:
            # Unfreeze all layers
            self.base_model.trainable = True
        else:
            # Unfreeze only top layers
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-num_layers]:
                layer.trainable = False
        
        # Print trainable status
        trainable_count = sum([1 for layer in self.base_model.layers if layer.trainable])
        print(f"Trainable layers in base model: {trainable_count}/{len(self.base_model.layers)}")
    
    def get_model(self) -> Model:
        """Get the compiled model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
    
    def save_model(self, filepath: str):
        """Save model in Keras format"""
        if not filepath.endswith('.keras'):
            filepath = filepath + '.keras'
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")


def create_robust_model(num_classes: int = 7,
                       input_shape: Tuple[int, int, int] = (224, 224, 3),
                       y_train: Optional[np.ndarray] = None) -> RobustDiseaseDetector:
    """
    Factory function to create a robust model with optimal settings
    
    Args:
        num_classes: Number of disease categories
        input_shape: Input image shape
        y_train: Training labels for calculating class weights
        
    Returns:
        Configured RobustDiseaseDetector instance
    """
    
    # Create model
    model = RobustDiseaseDetector(
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=0.4,
        l2_regularization=0.0005,
        use_mixed_precision=False  # Set True if using GPU
    )
    
    # Calculate class weights if training data provided
    class_weights = None
    if y_train is not None:
        class_weights = model.get_class_weights(y_train)
    
    # Compile with optimal settings
    model.compile_model(
        learning_rate=0.001,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0,
        class_weights=class_weights,
        label_smoothing=0.1,
        use_ema=True
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Robust Disease Detector...")
    
    # Create model
    model = create_robust_model(num_classes=7)
    
    # Print summary
    model.summary()
    
    # Test with dummy data
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.get_model()(dummy_input)
    print(f"\nModel output shape: {output.shape}")
    print("Model test successful!")