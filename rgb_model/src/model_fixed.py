"""
Fixed model implementation with corrected Focal Loss and architecture
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    """
    FIXED Focal Loss implementation for addressing class imbalance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        Fixed focal loss computation
        """
        # Apply softmax if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight: (1-p_t)^gamma
        # p_t is the probability of the true class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Apply focal weight to cross entropy
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha balancing (optional)
        if self.alpha is not None:
            alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            focal_loss = alpha_weight * focal_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def build_fixed_model(num_classes=7, input_shape=(224, 224, 3)):
    """
    Build a properly configured model for plant disease detection
    """
    
    try:
        # Try EfficientNet first
        from tensorflow.keras.applications import EfficientNetB0
        
        # Base model with proper input
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None
        )
        
        # Create input layer
        inputs = layers.Input(shape=input_shape)
        
        # Preprocessing for EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Pass through base model
        x = base_model(x, training=False)
        
        # Initially freeze base model
        base_model.trainable = False
        
        # Get features from the processed output
        features = x
        
    except ImportError:
        print("EfficientNet not available, using custom CNN")
        
        inputs = layers.Input(shape=input_shape)
        
        # Custom CNN architecture
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        features = x
        base_model = None
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(features)  # Changed from GlobalMaxPooling
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer - NO activation for use with from_logits=True
    outputs = layers.Dense(num_classes, activation=None, name='logits')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model


def compile_fixed_model(model, learning_rate=0.001, use_focal_loss=True):
    """
    Compile model with proper loss and optimizer settings
    """
    
    # Use Adam with proper learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if use_focal_loss:
        # Use focal loss with from_logits=True
        loss = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
    else:
        # Standard categorical crossentropy with label smoothing
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,  # Important: use from_logits=True
            label_smoothing=0.1
        )
    
    # Compile with proper metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def test_fixed_model():
    """Test the fixed model"""
    
    # Create model
    model, base_model = build_fixed_model(num_classes=7)
    
    # Compile
    model = compile_fixed_model(model, learning_rate=0.001)
    
    # Test with dummy data
    batch_size = 8
    dummy_x = np.random.random((batch_size, 224, 224, 3)).astype(np.float32)
    dummy_y = np.eye(7)[np.random.randint(0, 7, batch_size)]  # One-hot labels
    
    # Test forward pass
    output = model(dummy_x, training=False)
    print(f"Output shape: {output.shape}")
    
    # Test loss calculation
    loss = model.compiled_loss(dummy_y, output)
    print(f"Loss value: {loss.numpy():.4f}")
    
    # Test training step
    with tf.GradientTape() as tape:
        predictions = model(dummy_x, training=True)
        loss = model.compiled_loss(dummy_y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print("✓ Model working correctly!")
    
    return model


if __name__ == "__main__":
    test_fixed_model()