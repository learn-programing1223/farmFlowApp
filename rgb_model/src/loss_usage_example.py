"""
Example Usage of Advanced Loss Functions in Training
=====================================================

This script demonstrates how to integrate the custom loss functions
with TensorFlow/Keras model training for improved generalization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from losses import (
    FocalLoss, 
    LabelSmoothingCrossEntropy,
    CombinedLoss,
    ConfidencePenaltyLoss,
    get_loss_by_name
)


def create_simple_model(num_classes=6, input_shape=(224, 224, 3)):
    """Create a simple CNN model for demonstration."""
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def example_1_focal_loss():
    """Example 1: Using Focal Loss for imbalanced dataset."""
    print("Example 1: Focal Loss for Imbalanced Dataset")
    print("-" * 50)
    
    # Create model
    model = create_simple_model()
    
    # Use focal loss with high gamma for hard examples
    focal_loss = FocalLoss(
        alpha=1.0,  # No class weighting
        gamma=2.0,  # Focus on hard examples
        label_smoothing=0.05  # Small amount of label smoothing
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    print("Model compiled with Focal Loss")
    print(f"  - Alpha: 1.0 (no class weighting)")
    print(f"  - Gamma: 2.0 (focus on hard examples)")
    print(f"  - Label smoothing: 0.05")
    print()
    
    return model


def example_2_label_smoothing():
    """Example 2: Using Label Smoothing for better generalization."""
    print("Example 2: Label Smoothing Cross-Entropy")
    print("-" * 50)
    
    # Create model
    model = create_simple_model()
    
    # Define class weights for imbalanced classes
    class_weights = {
        0: 1.0,   # Blight
        1: 1.0,   # Healthy
        2: 1.0,   # Leaf_Spot
        3: 1.0,   # Mosaic_Virus
        4: 1.5,   # Nutrient_Deficiency (underrepresented)
        5: 1.0    # Powdery_Mildew
    }
    
    # Use label smoothing with class weights
    ls_loss = LabelSmoothingCrossEntropy(
        epsilon=0.1,  # 10% label smoothing
        class_weights=class_weights
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=ls_loss,
        metrics=['accuracy']
    )
    
    print("Model compiled with Label Smoothing CE")
    print(f"  - Epsilon: 0.1 (prevents overconfidence)")
    print(f"  - Class weights: Applied for imbalanced data")
    print()
    
    return model


def example_3_combined_loss():
    """Example 3: Combining multiple losses for best of both worlds."""
    print("Example 3: Combined Loss (Focal + Label Smoothing)")
    print("-" * 50)
    
    # Create model
    model = create_simple_model()
    
    # Create combined loss
    combined_loss = CombinedLoss(
        losses=[
            FocalLoss(gamma=2.0, alpha=0.75),  # Focus on hard examples
            LabelSmoothingCrossEntropy(epsilon=0.1)  # Prevent overconfidence
        ],
        weights=[0.7, 0.3]  # More weight on focal loss
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=combined_loss,
        metrics=['accuracy']
    )
    
    print("Model compiled with Combined Loss")
    print(f"  - 70% Focal Loss (gamma=2.0)")
    print(f"  - 30% Label Smoothing (epsilon=0.1)")
    print()
    
    return model


def example_4_progressive_training():
    """Example 4: Progressive training with different losses."""
    print("Example 4: Progressive Training Strategy")
    print("-" * 50)
    
    # Create model
    model = create_simple_model()
    
    print("Stage 1: Initial training with Label Smoothing")
    # Start with label smoothing for stable training
    initial_loss = LabelSmoothingCrossEntropy(epsilon=0.15)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=initial_loss,
        metrics=['accuracy']
    )
    print(f"  - High label smoothing (0.15) for initial stability")
    
    print("\nStage 2: Fine-tuning with Focal Loss")
    # Switch to focal loss for hard examples
    finetune_loss = FocalLoss(gamma=3.0, alpha=0.75)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=finetune_loss,
        metrics=['accuracy']
    )
    print(f"  - High gamma (3.0) to focus on mistakes")
    print(f"  - Lower learning rate (1e-4)")
    
    print("\nStage 3: Final refinement with Confidence Penalty")
    # Final stage with confidence penalty
    final_loss = ConfidencePenaltyLoss(beta=0.2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=final_loss,
        metrics=['accuracy']
    )
    print(f"  - Confidence penalty to improve calibration")
    print(f"  - Very low learning rate (1e-5)")
    print()
    
    return model


def example_5_custom_training_loop():
    """Example 5: Custom training loop with dynamic loss weighting."""
    print("Example 5: Custom Training Loop with Dynamic Loss")
    print("-" * 50)
    
    # Create model
    model = create_simple_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    
    # Create multiple losses
    focal_loss = FocalLoss(gamma=2.0)
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    
    @tf.function
    def train_step(x, y, epoch):
        """Custom training step with dynamic loss weighting."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(x, training=True)
            
            # Calculate both losses
            loss_focal = focal_loss(y, predictions)
            loss_ls = ls_loss(y, predictions)
            
            # Dynamic weighting based on epoch
            # Early epochs: more label smoothing
            # Later epochs: more focal loss
            focal_weight = tf.minimum(epoch / 20.0, 0.8)
            ls_weight = 1.0 - focal_weight
            
            # Combined loss
            total_loss = focal_weight * loss_focal + ls_weight * loss_ls
        
        # Backward pass
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, loss_focal, loss_ls
    
    print("Custom training loop created:")
    print("  - Dynamic loss weighting based on epoch")
    print("  - Early epochs: More label smoothing")
    print("  - Later epochs: More focal loss")
    print("  - Smooth transition over 20 epochs")
    print()
    
    # Demonstrate one training step
    dummy_x = tf.random.normal((4, 224, 224, 3))
    dummy_y = tf.one_hot([0, 1, 2, 3], depth=6)
    
    for epoch in [1, 10, 20]:
        total_loss, f_loss, ls_loss = train_step(dummy_x, dummy_y, tf.constant(epoch, dtype=tf.float32))
        focal_weight = min(epoch / 20.0, 0.8)
        ls_weight = 1.0 - focal_weight
        print(f"  Epoch {epoch}: Focal weight={focal_weight:.2f}, LS weight={ls_weight:.2f}")
    
    return model


def example_6_loss_by_name():
    """Example 6: Using get_loss_by_name for configuration."""
    print("Example 6: Configuration-based Loss Selection")
    print("-" * 50)
    
    # Configuration dictionary (could come from config file)
    training_configs = [
        {
            'name': 'baseline',
            'loss': 'categorical_crossentropy',
            'params': {}
        },
        {
            'name': 'focal',
            'loss': 'focal',
            'params': {'gamma': 2.0, 'alpha': 0.75}
        },
        {
            'name': 'label_smoothing',
            'loss': 'label_smoothing',
            'params': {'epsilon': 0.1}
        },
        {
            'name': 'combined',
            'loss': 'combined',
            'params': {
                'losses': ['focal', 'label_smoothing'],
                'weights': [0.6, 0.4]
            }
        }
    ]
    
    print("Creating models with different loss configurations:")
    for config in training_configs:
        model = create_simple_model()
        
        # Get loss by name
        loss = get_loss_by_name(config['loss'], **config['params'])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        print(f"  - {config['name']}: {config['loss']} with params {config['params']}")
    
    print()
    return model


def main():
    """Run all examples."""
    print("=" * 60)
    print("Advanced Loss Functions - Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_1_focal_loss()
    example_2_label_smoothing()
    example_3_combined_loss()
    example_4_progressive_training()
    example_5_custom_training_loop()
    example_6_loss_by_name()
    
    print("=" * 60)
    print("Recommendations for Plant Disease Detection")
    print("=" * 60)
    print("""
1. For Initial Training:
   - Use LabelSmoothingCrossEntropy(epsilon=0.1-0.15)
   - Helps prevent overfitting to training data
   
2. For Imbalanced Classes (e.g., Nutrient Deficiency):
   - Use FocalLoss(gamma=2.0) with class weights
   - Or LabelSmoothingCE with class_weights parameter
   
3. For Best Performance:
   - Use CombinedLoss with 70% Focal + 30% Label Smoothing
   - Balances hard example mining with regularization
   
4. For Fine-tuning on Real Images:
   - Start with pre-trained weights
   - Use ConfidencePenaltyLoss(beta=0.1-0.2)
   - Helps calibrate predictions for deployment
   
5. For Progressive Training:
   - Stage 1: Label smoothing (stable start)
   - Stage 2: Focal loss (focus on errors)
   - Stage 3: Confidence penalty (calibration)
    """)


if __name__ == "__main__":
    main()