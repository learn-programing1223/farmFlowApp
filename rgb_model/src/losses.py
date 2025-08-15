"""
Advanced Loss Functions for Robust Plant Disease Detection
==========================================================

This module implements specialized loss functions designed to improve model
generalization on real-world, noisy internet images. These losses help prevent
overfitting to clean training data and improve performance on difficult samples.

Key Features:
- Focal Loss: Focuses learning on hard samples
- Label Smoothing Cross-Entropy: Prevents overconfidence
- Combined Loss: Weighted combination of multiple losses
- CORAL Loss: Domain adaptation for distribution alignment
- All losses compatible with TensorFlow/Keras

Author: PlantPulse Team
Date: 2025
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Union, Dict, List, Callable
import numpy as np


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance and hard sample mining.
    
    Focal loss applies a modulating term to the cross-entropy loss to focus
    learning on hard negative examples. It down-weights well-classified examples
    and focuses on misclassified ones.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        alpha: Balancing factor for positive/negative examples (default: 1.0)
        gamma: Focusing parameter for modulating loss (default: 2.0)
        from_logits: Whether predictions are logits or probabilities
        label_smoothing: Optional label smoothing factor
        
    Example:
        >>> import tensorflow as tf
        >>> # Create focal loss
        >>> focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        >>> 
        >>> # Generate sample data
        >>> y_true = tf.one_hot([0, 1, 2], depth=3)
        >>> y_pred = tf.constant([[0.9, 0.05, 0.05],
        ...                       [0.1, 0.8, 0.1],
        ...                       [0.1, 0.2, 0.7]])
        >>> 
        >>> # Calculate loss
        >>> loss = focal_loss(y_true, y_pred)
        >>> print(f"Focal loss: {loss.numpy():.4f}")
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'focal_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        """
        Calculate focal loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.shape(y_true)[-1]
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / tf.cast(num_classes, tf.float32))
        
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss components
        # CE = -y_true * log(y_pred)
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Focal term = (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_term = tf.pow(1.0 - p_t, self.gamma)
        
        # Combine with alpha weighting
        focal_loss = self.alpha * focal_term * ce_loss
        
        # Sum over classes and return
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config


class LabelSmoothingCrossEntropy(keras.losses.Loss):
    """
    Cross-entropy loss with label smoothing regularization.
    
    Label smoothing prevents the model from becoming overconfident by replacing
    hard labels with soft targets. Instead of [0, 1, 0], we use [0.05, 0.9, 0.05].
    
    Args:
        epsilon: Label smoothing factor (default: 0.1)
        from_logits: Whether predictions are logits or probabilities
        class_weights: Optional class weights for imbalanced datasets
        
    Example:
        >>> import tensorflow as tf
        >>> # Create label smoothing loss
        >>> ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
        >>> 
        >>> # Generate sample data
        >>> y_true = tf.one_hot([0, 1, 2], depth=3)
        >>> y_pred = tf.constant([[0.9, 0.05, 0.05],
        ...                       [0.1, 0.8, 0.1],
        ...                       [0.1, 0.2, 0.7]])
        >>> 
        >>> # Calculate loss
        >>> loss = ls_loss(y_true, y_pred)
        >>> print(f"Label smoothing CE loss: {loss.numpy():.4f}")
    """
    
    def __init__(
        self,
        epsilon: float = 0.1,
        from_logits: bool = False,
        class_weights: Optional[Union[Dict[int, float], List[float]]] = None,
        reduction: str = 'sum_over_batch_size',
        name: str = 'label_smoothing_ce'
    ):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon
        self.from_logits = from_logits
        self.class_weights = class_weights
        
        # Convert class weights to tensor if provided
        if class_weights is not None:
            if isinstance(class_weights, dict):
                # Convert dict to list
                num_classes = max(class_weights.keys()) + 1
                weights_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
                self.class_weights_tensor = tf.constant(weights_list, dtype=tf.float32)
            else:
                self.class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights_tensor = None
    
    def call(self, y_true, y_pred):
        """
        Calculate label smoothing cross-entropy loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Label smoothing cross-entropy loss
        """
        # Get number of classes
        num_classes = tf.shape(y_true)[-1]
        
        # Apply label smoothing
        y_true_smooth = y_true * (1.0 - self.epsilon) + (self.epsilon / tf.cast(num_classes, tf.float32))
        
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon_clip = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon_clip, 1.0 - epsilon_clip)
        
        # Calculate cross-entropy
        ce_loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        
        # Apply class weights if provided
        if self.class_weights_tensor is not None:
            # Get the class indices from one-hot labels
            class_indices = tf.argmax(y_true, axis=-1)
            sample_weights = tf.gather(self.class_weights_tensor, class_indices)
            ce_loss = ce_loss * sample_weights
        
        return tf.reduce_mean(ce_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'from_logits': self.from_logits,
            'class_weights': self.class_weights
        })
        return config


class CombinedLoss(keras.losses.Loss):
    """
    Combines multiple loss functions with configurable weights.
    
    This allows using multiple objectives simultaneously, such as combining
    focal loss for hard examples with label smoothing for regularization.
    
    Args:
        losses: List of loss functions or loss names
        weights: Weights for each loss function
        
    Example:
        >>> import tensorflow as tf
        >>> # Create combined loss with focal and label smoothing
        >>> combined_loss = CombinedLoss(
        ...     losses=[
        ...         FocalLoss(gamma=2.0),
        ...         LabelSmoothingCrossEntropy(epsilon=0.1)
        ...     ],
        ...     weights=[0.7, 0.3]
        ... )
        >>> 
        >>> # Generate sample data
        >>> y_true = tf.one_hot([0, 1, 2], depth=3)
        >>> y_pred = tf.constant([[0.9, 0.05, 0.05],
        ...                       [0.1, 0.8, 0.1],
        ...                       [0.1, 0.2, 0.7]])
        >>> 
        >>> # Calculate combined loss
        >>> loss = combined_loss(y_true, y_pred)
        >>> print(f"Combined loss: {loss.numpy():.4f}")
    """
    
    def __init__(
        self,
        losses: List[Union[keras.losses.Loss, str]],
        weights: Optional[List[float]] = None,
        reduction: str = 'sum_over_batch_size',
        name: str = 'combined_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        
        # Convert string loss names to loss objects
        self.losses = []
        for loss in losses:
            if isinstance(loss, str):
                self.losses.append(get_loss_by_name(loss))
            else:
                self.losses.append(loss)
        
        # Set equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(losses)] * len(losses)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def call(self, y_true, y_pred):
        """
        Calculate combined loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predictions
            
        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(y_true, y_pred)
            total_loss += weight * loss_value
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'losses': [loss.get_config() for loss in self.losses],
            'weights': self.weights
        })
        return config


class CORALLoss(keras.losses.Loss):
    """
    CORrelation ALignment (CORAL) loss for domain adaptation.
    
    CORAL aligns the second-order statistics (covariance) of source and target
    feature distributions. This helps when training on clean data but deploying
    on noisy real-world images.
    
    Reference: Sun et al., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (2016)
    
    Args:
        lambda_coral: Weight for CORAL loss term (default: 1.0)
        
    Example:
        >>> import tensorflow as tf
        >>> # Create CORAL loss for domain adaptation
        >>> coral_loss = CORALLoss(lambda_coral=0.5)
        >>> 
        >>> # Generate sample features from source and target domains
        >>> source_features = tf.random.normal((32, 128))  # 32 samples, 128 features
        >>> target_features = tf.random.normal((32, 128))
        >>> 
        >>> # Calculate CORAL loss
        >>> loss = coral_loss(source_features, target_features)
        >>> print(f"CORAL loss: {loss.numpy():.4f}")
    """
    
    def __init__(
        self,
        lambda_coral: float = 1.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'coral_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.lambda_coral = lambda_coral
    
    def _compute_covariance(self, features):
        """
        Compute covariance matrix of features.
        
        Args:
            features: Feature tensor (batch_size, num_features)
            
        Returns:
            Covariance matrix
        """
        n = tf.cast(tf.shape(features)[0], tf.float32)
        
        # Center the features
        features_mean = tf.reduce_mean(features, axis=0, keepdims=True)
        features_centered = features - features_mean
        
        # Compute covariance
        covariance = tf.matmul(
            features_centered,
            features_centered,
            transpose_a=True
        ) / (n - 1)
        
        return covariance
    
    def call(self, source_features, target_features):
        """
        Calculate CORAL loss between source and target features.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            CORAL loss value
        """
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Compute Frobenius norm of difference
        diff = source_cov - target_cov
        loss = tf.reduce_sum(tf.square(diff))
        
        # Normalize by 4 * d^2 where d is feature dimension
        d = tf.cast(tf.shape(source_features)[1], tf.float32)
        loss = loss / (4.0 * d * d)
        
        return self.lambda_coral * loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lambda_coral': self.lambda_coral
        })
        return config


class ConfidencePenaltyLoss(keras.losses.Loss):
    """
    Confidence penalty loss to prevent overconfident predictions.
    
    This loss adds a penalty term that encourages the model to be less confident,
    which can improve calibration and generalization.
    
    Args:
        beta: Weight for confidence penalty (default: 0.1)
        base_loss: Base loss function to combine with penalty
        
    Example:
        >>> import tensorflow as tf
        >>> # Create confidence penalty loss
        >>> cp_loss = ConfidencePenaltyLoss(beta=0.1)
        >>> 
        >>> # Generate sample data
        >>> y_true = tf.one_hot([0, 1, 2], depth=3)
        >>> y_pred = tf.constant([[0.99, 0.005, 0.005],  # Overconfident
        ...                       [0.6, 0.3, 0.1],        # Moderate
        ...                       [0.4, 0.3, 0.3]])       # Uncertain
        >>> 
        >>> # Calculate loss
        >>> loss = cp_loss(y_true, y_pred)
        >>> print(f"Confidence penalty loss: {loss.numpy():.4f}")
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        base_loss: Optional[keras.losses.Loss] = None,
        from_logits: bool = False,
        reduction: str = 'sum_over_batch_size',
        name: str = 'confidence_penalty_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        self.base_loss = base_loss or keras.losses.CategoricalCrossentropy(from_logits=from_logits)
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        Calculate confidence penalty loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predictions
            
        Returns:
            Loss with confidence penalty
        """
        # Calculate base loss
        base_loss_value = self.base_loss(y_true, y_pred)
        
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Calculate entropy as confidence penalty
        # High entropy = low confidence, low entropy = high confidence
        epsilon = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        entropy = -tf.reduce_sum(y_pred_clipped * tf.math.log(y_pred_clipped), axis=-1)
        
        # We want to maximize entropy (minimize negative entropy)
        confidence_penalty = -tf.reduce_mean(entropy)
        
        # Combine base loss with confidence penalty
        total_loss = base_loss_value + self.beta * confidence_penalty
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'beta': self.beta,
            'from_logits': self.from_logits
        })
        return config


def get_loss_by_name(
    loss_name: str,
    **kwargs
) -> keras.losses.Loss:
    """
    Get a loss function by name with optional parameters.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
        
    Returns:
        Initialized loss function
        
    Available losses:
        - 'focal': Focal loss for hard examples
        - 'label_smoothing': Label smoothing cross-entropy
        - 'combined': Combined loss with multiple objectives
        - 'coral': CORAL loss for domain adaptation
        - 'confidence_penalty': Confidence penalty loss
        - 'categorical_crossentropy': Standard categorical cross-entropy
        - 'sparse_categorical_crossentropy': Sparse categorical cross-entropy
        
    Example:
        >>> # Get focal loss with custom parameters
        >>> loss = get_loss_by_name('focal', alpha=0.75, gamma=3.0)
        >>> 
        >>> # Get label smoothing with custom epsilon
        >>> loss = get_loss_by_name('label_smoothing', epsilon=0.2)
        >>> 
        >>> # Get combined loss
        >>> loss = get_loss_by_name('combined', 
        ...                         losses=['focal', 'label_smoothing'],
        ...                         weights=[0.6, 0.4])
    """
    loss_name = loss_name.lower()
    
    loss_map = {
        'focal': FocalLoss,
        'focal_loss': FocalLoss,
        'label_smoothing': LabelSmoothingCrossEntropy,
        'label_smoothing_ce': LabelSmoothingCrossEntropy,
        'label_smoothing_crossentropy': LabelSmoothingCrossEntropy,
        'combined': CombinedLoss,
        'combined_loss': CombinedLoss,
        'coral': CORALLoss,
        'coral_loss': CORALLoss,
        'confidence_penalty': ConfidencePenaltyLoss,
        'confidence_penalty_loss': ConfidencePenaltyLoss,
        'categorical_crossentropy': keras.losses.CategoricalCrossentropy,
        'sparse_categorical_crossentropy': keras.losses.SparseCategoricalCrossentropy,
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name](**kwargs)


def test_losses():
    """
    Test all loss functions with sample data.
    """
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    batch_size = 8
    num_classes = 6  # Plant disease classes
    
    # Generate random predictions and labels
    y_true = tf.one_hot(np.random.randint(0, num_classes, batch_size), num_classes)
    y_pred_logits = tf.random.normal((batch_size, num_classes))
    y_pred_probs = tf.nn.softmax(y_pred_logits)
    
    print(f"Sample data shape: y_true={y_true.shape}, y_pred={y_pred_probs.shape}")
    print()
    
    # Test Focal Loss
    print("1. Focal Loss")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss_value = focal_loss(y_true, y_pred_probs)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    
    # Test with different gamma values
    for gamma in [0.0, 1.0, 3.0, 5.0]:
        focal_loss_gamma = FocalLoss(gamma=gamma)
        loss = focal_loss_gamma(y_true, y_pred_probs)
        print(f"   Gamma={gamma}: {loss.numpy():.4f}")
    print()
    
    # Test Label Smoothing
    print("2. Label Smoothing Cross-Entropy")
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss_value = ls_loss(y_true, y_pred_probs)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    
    # Test with different epsilon values
    for epsilon in [0.0, 0.05, 0.2, 0.3]:
        ls_loss_eps = LabelSmoothingCrossEntropy(epsilon=epsilon)
        loss = ls_loss_eps(y_true, y_pred_probs)
        print(f"   Epsilon={epsilon}: {loss.numpy():.4f}")
    print()
    
    # Test Combined Loss
    print("3. Combined Loss")
    combined_loss = CombinedLoss(
        losses=[
            FocalLoss(gamma=2.0),
            LabelSmoothingCrossEntropy(epsilon=0.1)
        ],
        weights=[0.7, 0.3]
    )
    loss_value = combined_loss(y_true, y_pred_probs)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    print()
    
    # Test CORAL Loss
    print("4. CORAL Loss (Domain Adaptation)")
    source_features = tf.random.normal((batch_size, 128))
    target_features = tf.random.normal((batch_size, 128))
    coral_loss = CORALLoss(lambda_coral=0.5)
    loss_value = coral_loss(source_features, target_features)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    print()
    
    # Test Confidence Penalty
    print("5. Confidence Penalty Loss")
    cp_loss = ConfidencePenaltyLoss(beta=0.1)
    loss_value = cp_loss(y_true, y_pred_probs)
    print(f"   Loss value: {loss_value.numpy():.4f}")
    print()
    
    # Test get_loss_by_name
    print("6. Get Loss by Name")
    loss_names = ['focal', 'label_smoothing', 'confidence_penalty']
    for name in loss_names:
        loss_fn = get_loss_by_name(name)
        loss_value = loss_fn(y_true, y_pred_probs)
        print(f"   {name}: {loss_value.numpy():.4f}")
    print()
    
    # Test with class weights
    print("7. Label Smoothing with Class Weights")
    class_weights = {
        0: 1.0,   # Blight
        1: 1.0,   # Healthy
        2: 1.0,   # Leaf_Spot
        3: 1.0,   # Mosaic_Virus
        4: 2.0,   # Nutrient_Deficiency (underrepresented)
        5: 1.0    # Powdery_Mildew
    }
    weighted_loss = LabelSmoothingCrossEntropy(epsilon=0.1, class_weights=class_weights)
    loss_value = weighted_loss(y_true, y_pred_probs)
    print(f"   Loss with class weights: {loss_value.numpy():.4f}")
    
    print("\n[OK] All loss functions tested successfully!")


if __name__ == "__main__":
    # Run tests
    test_losses()
    
    print("\n" + "=" * 60)
    print("Loss Functions Summary")
    print("=" * 60)
    print("""
Available loss functions for robust training:

1. FocalLoss: 
   - Focuses on hard examples
   - Reduces impact of easy samples
   - Good for imbalanced datasets
   
2. LabelSmoothingCrossEntropy:
   - Prevents overconfidence
   - Improves generalization
   - Reduces overfitting to training data
   
3. CombinedLoss:
   - Combines multiple objectives
   - Flexible weighting
   - Best of multiple approaches
   
4. CORALLoss:
   - Domain adaptation
   - Aligns feature distributions
   - Useful for clean→noisy transfer
   
5. ConfidencePenaltyLoss:
   - Penalizes overconfident predictions
   - Improves calibration
   - Better uncertainty estimates

Recommended for plant disease detection:
- Training: CombinedLoss([FocalLoss(gamma=2), LabelSmoothingCE(epsilon=0.1)], weights=[0.7, 0.3])
- Fine-tuning: LabelSmoothingCrossEntropy(epsilon=0.1, class_weights=computed_weights)
- Domain adaptation: Add CORALLoss when training on clean data for noisy deployment
    """)