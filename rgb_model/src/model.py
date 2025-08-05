import tensorflow as tf
from tensorflow.keras import layers, Model
try:
    from tensorflow.keras.applications import EfficientNetB0
except ImportError:
    from keras.applications import EfficientNetB0
from typing import Tuple, Optional
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in plant disease detection.
    Focuses learning on hard examples.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 from_logits: bool = False, name: str = 'focal_loss'):
        """
        Args:
            alpha: Balancing factor for positive/negative examples
            gamma: Focusing parameter for hard examples
            from_logits: Whether predictions are logits or probabilities
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        Computes focal loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
        """
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        # CE = -log(p_t)
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # p_t = p if y=1, 1-p otherwise
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Focal weight = (1 - p_t)^gamma
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha balancing
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Reduce mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class UniversalDiseaseDetector:
    """
    EfficientNet-B0 based model for universal plant disease detection.
    Optimized for mobile deployment with high accuracy.
    """
    
    def __init__(self, num_classes: int = 7, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 dropout_rate: float = 0.5,
                 l2_regularization: float = 0.001):
        """
        Args:
            num_classes: Number of universal disease categories
            input_shape: Input image shape
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        
        self.model = self._build_model()
        self.base_model = None  # Will be set if EfficientNet works
    
    def _build_custom_cnn(self, inputs):
        """Fallback CNN architecture when EfficientNet fails"""
        x = inputs
        
        # Conv Block 1
        x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 2
        x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 3
        x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Conv Block 4
        x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(512, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Create a pseudo base model for compatibility
        cnn_model = Model(inputs=inputs, outputs=x, name='custom_cnn_base')
        cnn_model.trainable = False  # Start with frozen layers
        self.base_model = cnn_model
        
        return x
    
    def _build_model(self) -> Model:
        """
        Builds the EfficientNet-B0 based model architecture.
        """
        # Create inputs
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        try:
            # Try to use EfficientNet-B0
            # Create a new input tensor for EfficientNet
            base_model = EfficientNetB0(
                input_tensor=inputs,
                weights='imagenet',
                include_top=False,
                pooling=None
            )
            
            # Freeze base model initially
            base_model.trainable = False
            
            # Store base model reference for fine-tuning
            self.base_model = base_model
            
            # Get the output from base model
            x = base_model.output
        except Exception as e:
            print(f"Warning: EfficientNet failed ({str(e)}). Using custom CNN instead.")
            # Fallback to custom CNN - this will set self.base_model
            x = self._build_custom_cnn(inputs)
        
        # Global pooling - GlobalMaxPooling better for disease features
        x = layers.GlobalMaxPooling2D(name='global_max_pool')(x)
        
        # Batch normalization
        x = layers.BatchNormalization(name='bn_1')(x)
        
        # First dense block with regularization
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            bias_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='dense_1'
        )(x)
        
        # Second batch normalization
        x = layers.BatchNormalization(name='bn_2')(x)
        
        # Second dropout
        x = layers.Dropout(self.dropout_rate * 0.6, name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
            name='predictions'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='universal_disease_detector')
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001,
                     use_focal_loss: bool = True,
                     focal_alpha: float = 0.75,
                     focal_gamma: float = 2.0):
        """
        Compiles the model with optimizer and loss function.
        
        Args:
            learning_rate: Initial learning rate
            use_focal_loss: Whether to use focal loss (vs categorical crossentropy)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss function
        if use_focal_loss:
            loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        # Metrics
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def unfreeze_layers(self, num_layers: int = -1):
        """
        Unfreezes layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top.
                       -1 means unfreeze all layers.
        """
        if self.base_model is None:
            print("No base model available")
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
        """Returns the compiled model."""
        return self.model
    
    def summary(self):
        """Prints model summary."""
        self.model.summary()
    
    def save_model(self, filepath: str):
        """
        Saves the model in Keras format.
        
        Args:
            filepath: Path to save the model
        """
        # Ensure filepath has .keras extension
        if not filepath.endswith('.keras'):
            filepath = filepath + '.keras'
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """Loads model weights."""
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")


class ModelMetrics(tf.keras.callbacks.Callback):
    """
    Custom callback for tracking per-class metrics during training.
    """
    
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate and print per-class metrics."""
        x_val, y_val = self.validation_data
        
        # Get predictions
        y_pred = self.model.predict(x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Calculate per-class accuracy
        print(f"\nPer-class metrics for epoch {epoch + 1}:")
        for i, class_name in enumerate(self.class_names):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred_classes[mask] == i)
                class_samples = np.sum(mask)
                print(f"  {class_name}: {class_acc:.3f} ({class_samples} samples)")


def create_model_with_augmentation():
    """
    Creates model with built-in augmentation layers for inference-time augmentation.
    """
    # Augmentation layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ], name='augmentation')
    
    # Create disease detector
    detector = UniversalDiseaseDetector()
    
    # Input
    inputs = layers.Input(shape=(224, 224, 3))
    
    # Apply augmentation
    x = data_augmentation(inputs)
    
    # Pass through the detector's layers manually
    # Get the base model output
    base_output = detector.base_model(x, training=False)
    
    # Apply the same processing as in the detector
    x = layers.GlobalMaxPooling2D(name='global_max_pool')(base_output)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(detector.dropout_rate, name='dropout_1')(x)
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(detector.l2_regularization),
        bias_regularizer=tf.keras.regularizers.l2(detector.l2_regularization),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(detector.dropout_rate * 0.6, name='dropout_2')(x)
    outputs = layers.Dense(
        detector.num_classes,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(detector.l2_regularization),
        name='predictions'
    )(x)
    
    # Create augmented model
    augmented_model = Model(inputs=inputs, outputs=outputs, 
                           name='augmented_disease_detector')
    
    return augmented_model


def test_model():
    """Test model creation and compilation."""
    # Create model
    detector = UniversalDiseaseDetector(num_classes=7)
    
    # Compile with focal loss
    detector.compile_model(
        learning_rate=0.001,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0
    )
    
    # Print summary
    detector.summary()
    
    # Test forward pass
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = detector.model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output sum: {tf.reduce_sum(output):.4f}")  # Should be ~1.0 for softmax
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    y_true = tf.one_hot([2], depth=7)  # One-hot encoded label
    y_pred = output
    loss_value = focal_loss(y_true, y_pred)
    print(f"Focal loss value: {loss_value:.4f}")


if __name__ == "__main__":
    test_model()