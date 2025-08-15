# Advanced Loss Functions for Robust Training

## Overview
Implemented a comprehensive suite of loss functions designed to improve model generalization on real-world, noisy internet images. These losses help prevent overfitting to clean training data and improve performance on difficult samples.

## Implemented Loss Functions

### 1. **FocalLoss**
- **Purpose**: Addresses class imbalance and focuses on hard examples
- **Key Parameters**:
  - `alpha`: Balancing factor (default: 1.0)
  - `gamma`: Focusing parameter (default: 2.0)
  - `label_smoothing`: Optional smoothing factor
- **When to Use**: 
  - Imbalanced datasets
  - When model struggles with specific classes
  - To reduce impact of easy samples

### 2. **LabelSmoothingCrossEntropy**
- **Purpose**: Prevents overconfidence and improves generalization
- **Key Parameters**:
  - `epsilon`: Smoothing factor (default: 0.1)
  - `class_weights`: Optional weights for imbalanced classes
- **When to Use**:
  - Initial training for stability
  - When model shows overconfidence
  - To improve calibration

### 3. **CombinedLoss**
- **Purpose**: Combines multiple loss functions with configurable weights
- **Key Parameters**:
  - `losses`: List of loss functions
  - `weights`: Weights for each loss
- **When to Use**:
  - To get benefits of multiple approaches
  - Fine-tuning complex models
  - When single loss is insufficient

### 4. **CORALLoss**
- **Purpose**: Domain adaptation by aligning feature distributions
- **Key Parameters**:
  - `lambda_coral`: Weight for CORAL term (default: 1.0)
- **When to Use**:
  - Training on clean data for noisy deployment
  - Cross-domain transfer learning
  - Reducing distribution shift

### 5. **ConfidencePenaltyLoss**
- **Purpose**: Penalizes overconfident predictions
- **Key Parameters**:
  - `beta`: Penalty weight (default: 0.1)
  - `base_loss`: Base loss to combine with
- **When to Use**:
  - Final fine-tuning
  - Improving prediction calibration
  - Reducing false positives

## Usage Examples

### Basic Usage
```python
from src.losses import FocalLoss, LabelSmoothingCrossEntropy, get_loss_by_name

# Option 1: Direct instantiation
focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

# Option 2: Get by name
loss = get_loss_by_name('focal', gamma=2.0, alpha=0.75)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
```

### Recommended Configurations

#### For Plant Disease Detection
```python
# Initial Training - Stable start with label smoothing
initial_loss = LabelSmoothingCrossEntropy(
    epsilon=0.1,
    class_weights={
        0: 1.0,   # Blight
        1: 1.0,   # Healthy
        2: 1.0,   # Leaf_Spot
        3: 1.0,   # Mosaic_Virus
        4: 1.5,   # Nutrient_Deficiency (underrepresented)
        5: 1.0    # Powdery_Mildew
    }
)

# Advanced Training - Combined approach
combined_loss = CombinedLoss(
    losses=[
        FocalLoss(gamma=2.0, alpha=0.75),
        LabelSmoothingCrossEntropy(epsilon=0.1)
    ],
    weights=[0.7, 0.3]
)

# Fine-tuning - Focus on calibration
finetune_loss = ConfidencePenaltyLoss(
    beta=0.15,
    base_loss=LabelSmoothingCrossEntropy(epsilon=0.05)
)
```

### Progressive Training Strategy
```python
# Stage 1: Warm-up (5 epochs)
model.compile(
    optimizer=Adam(lr=1e-3),
    loss=LabelSmoothingCrossEntropy(epsilon=0.15),
    metrics=['accuracy']
)

# Stage 2: Main training (20 epochs)
model.compile(
    optimizer=Adam(lr=5e-4),
    loss=CombinedLoss(
        losses=['focal', 'label_smoothing'],
        weights=[0.7, 0.3]
    ),
    metrics=['accuracy']
)

# Stage 3: Fine-tuning (10 epochs)
model.compile(
    optimizer=Adam(lr=1e-4),
    loss=ConfidencePenaltyLoss(beta=0.2),
    metrics=['accuracy']
)
```

## Test Results

All loss functions tested successfully with:
- **Sample Size**: 8 samples, 6 classes
- **Test Coverage**: All losses computed without errors
- **Value Ranges**: Losses in expected ranges (0.03-3.14)
- **Class Weighting**: Verified to increase loss for weighted classes

### Performance Comparison
| Loss Function | Test Value | Key Benefit |
|--------------|------------|-------------|
| Focal (γ=2) | 1.74 | Focuses on hard examples |
| Label Smoothing (ε=0.1) | 2.16 | Prevents overconfidence |
| Combined (70/30) | 1.86 | Balanced approach |
| CORAL | 0.03 | Domain alignment |
| Confidence Penalty | 2.01 | Better calibration |

## Integration with Training Scripts

### Updating Existing Training Code
```python
# Old code
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# New code with advanced loss
from src.losses import get_loss_by_name

# Choose loss based on training phase
if args.training_phase == 'initial':
    loss = get_loss_by_name('label_smoothing', epsilon=0.1)
elif args.training_phase == 'main':
    loss = get_loss_by_name('combined', 
                           losses=['focal', 'label_smoothing'],
                           weights=[0.7, 0.3])
else:  # fine-tuning
    loss = get_loss_by_name('confidence_penalty', beta=0.15)

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)
```

## Benefits for Real-World Deployment

1. **Improved Generalization**
   - Label smoothing prevents memorization
   - Focal loss handles difficult cases
   - Combined approach balances both

2. **Better Calibration**
   - Confidence penalty reduces overconfidence
   - More reliable probability estimates
   - Better uncertainty quantification

3. **Robustness to Noise**
   - Trained on hard examples (focal)
   - Regularized predictions (smoothing)
   - Domain-adapted features (CORAL)

4. **Class Imbalance Handling**
   - Focal loss naturally handles imbalance
   - Class weights in label smoothing
   - Adaptive focusing on minorities

## Recommendations

### For PlantPulse RGB Model
1. **Initial Training**: 
   - Use `LabelSmoothingCrossEntropy(epsilon=0.1)` for stability

2. **Main Training**:
   - Use `CombinedLoss` with 70% Focal + 30% Label Smoothing
   - Best balance of hard example mining and regularization

3. **Fine-tuning on Failed Cases**:
   - Use `FocalLoss(gamma=3.0)` to focus on mistakes
   - Or `ConfidencePenaltyLoss` for better calibration

4. **Domain Adaptation** (future):
   - Add `CORALLoss` when training on clean PlantVillage for real-world deployment

## Files Created
- `src/losses.py` - Main loss implementations
- `src/loss_usage_example.py` - Detailed usage examples
- `LOSSES_DOCUMENTATION.md` - This documentation

## Compatibility
✅ TensorFlow 2.x compatible  
✅ Keras integration ready  
✅ Works with TF data pipelines  
✅ Supports mixed precision training  
✅ Configurable via get_loss_by_name()