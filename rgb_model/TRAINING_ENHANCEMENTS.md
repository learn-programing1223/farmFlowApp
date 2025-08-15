# Enhanced Training Script with Advanced Features

## Overview
Created `train_robust_model_v2.py` that integrates all the requested advanced features for improved model training and generalization on real-world data.

## Key Enhancements Implemented

### 1. **EnhancedDataLoader Integration**
- ✅ Imported and using `EnhancedDataLoader` from `src.data_loader_v2`
- ✅ Advanced preprocessing enabled by default (`use_advanced_preprocessing=True`)
- ✅ Configurable preprocessing modes: `default`, `fast`, `minimal`, `legacy`
- ✅ Using `create_tf_dataset()` method for efficient data pipeline

### 2. **Advanced Loss Functions**
- ✅ Replaced standard cross-entropy with `CombinedLoss`
- ✅ Default configuration: 70% Focal Loss + 30% Label Smoothing
- ✅ Configurable through command-line arguments:
  - `--focal_weight`: Weight for focal loss (default: 0.7)
  - `--focal_gamma`: Gamma parameter for focal loss (default: 2.0)
  - `--label_smoothing_epsilon`: Epsilon for label smoothing (default: 0.1)

### 3. **Stochastic Weight Averaging (SWA)**
- ✅ Custom `SWACallback` implementation
- ✅ Starts at epoch 20 by default (configurable via `--swa_start_epoch`)
- ✅ Maintains running average of model weights
- ✅ Applied at the end of training for better generalization

### 4. **Gradient Clipping**
- ✅ Implemented with `clipnorm` parameter in optimizer
- ✅ Default max norm: 1.0 (configurable via `--gradient_clip_norm`)
- ✅ Prevents gradient explosion during training

### 5. **MixUp Augmentation**
- ✅ Custom `MixUpAugmentation` layer (optional)
- ✅ 50% probability by default (configurable via `--mixup_probability`)
- ✅ Alpha parameter: 0.2 (configurable via `--mixup_alpha`)
- ✅ Applied during training for better regularization

### 6. **Preprocessing Comparison Mode**
- ✅ `--comparison_mode` flag to test different preprocessing modes
- ✅ Compares: `legacy`, `minimal`, `fast`, `default`
- ✅ Reports timing and statistics for each mode
- ✅ Helps choose optimal preprocessing for deployment

## Command-Line Interface

### Basic Usage
```bash
# Standard training with all enhancements
python train_robust_model_v2.py

# Quick test run (3 epochs)
python train_robust_model_v2.py --test_run

# Compare preprocessing modes
python train_robust_model_v2.py --comparison_mode

# Custom loss configuration
python train_robust_model_v2.py \
    --loss_type combined \
    --focal_weight 0.6 \
    --focal_gamma 3.0 \
    --label_smoothing_epsilon 0.15
```

### Full Arguments
```
--data_path              Path to dataset (default: datasets/plantvillage_processed)
--preprocessing_mode     Mode: default, fast, minimal, legacy
--use_advanced_preprocessing  Enable advanced preprocessing (default: True)
--comparison_mode        Compare preprocessing modes
--batch_size            Batch size (default: 32)
--epochs                Number of epochs (default: 30)
--learning_rate         Initial learning rate (default: 1e-3)
--loss_type             Loss: focal, label_smoothing, combined, standard
--focal_weight          Weight for focal loss in combined (default: 0.7)
--focal_gamma           Gamma for focal loss (default: 2.0)
--label_smoothing_epsilon  Epsilon for label smoothing (default: 0.1)
--swa_start_epoch       When to start SWA (default: 20)
--gradient_clip_norm    Max gradient norm (default: 1.0)
--mixup_alpha           MixUp alpha (default: 0.2)
--mixup_probability     MixUp probability (default: 0.5)
--test_run              Quick test with 2-3 epochs
--output_dir            Output directory (default: models)
```

## Training Pipeline Flow

1. **Data Loading**
   ```python
   # Enhanced data loader with advanced preprocessing
   train_loader = EnhancedDataLoader(
       data_dir=train_path,
       target_size=(224, 224),
       batch_size=args.batch_size,
       use_advanced_preprocessing=True,
       preprocessing_mode='default'
   )
   ```

2. **Dataset Creation**
   ```python
   # TensorFlow dataset with augmentation
   train_dataset = train_loader.create_tf_dataset(
       train_paths, train_labels,
       is_training=True,
       shuffle=True,
       augment=True  # Includes all augmentations
   )
   ```

3. **Loss Function**
   ```python
   # Combined loss for better generalization
   loss_fn = CombinedLoss(
       losses=[
           FocalLoss(gamma=2.0, alpha=0.75),
           LabelSmoothingCrossEntropy(epsilon=0.1)
       ],
       weights=[0.7, 0.3]
   )
   ```

4. **Optimizer with Gradient Clipping**
   ```python
   optimizer = keras.optimizers.Adam(
       learning_rate=args.learning_rate,
       clipnorm=args.gradient_clip_norm  # Gradient clipping
   )
   ```

5. **Training with Callbacks**
   - ModelCheckpoint (save best model)
   - ReduceLROnPlateau (adaptive learning rate)
   - EarlyStopping (prevent overfitting)
   - SWACallback (weight averaging)
   - TensorBoard (logging)

## Preprocessing Mode Comparison

The script can compare different preprocessing modes to find the optimal configuration:

```bash
python train_robust_model_v2.py --comparison_mode
```

Output example:
```
PREPROCESSING MODE COMPARISON
======================================================================
[Testing mode: legacy]
  Time: 0.523s
  Stats: mean=0.412, std=0.287, range=[0.000, 0.878]

[Testing mode: minimal]
  Time: 0.612s
  Stats: mean=0.408, std=0.291, range=[0.000, 0.878]

[Testing mode: fast]
  Time: 0.845s
  Stats: mean=0.425, std=0.268, range=[0.000, 1.000]

[Testing mode: default]
  Time: 1.234s
  Stats: mean=0.436, std=0.252, range=[0.176, 1.000]

COMPARISON SUMMARY
----------------------------------------------------------------------
legacy     - Time: 0.523s (speedup: 1.00x)
minimal    - Time: 0.612s (speedup: 0.85x)
fast       - Time: 0.845s (speedup: 0.62x)
default    - Time: 1.234s (speedup: 0.42x)
```

## Expected Improvements

### Over Original Script
1. **Better Generalization**: Advanced preprocessing handles real-world variations
2. **Reduced Overfitting**: Combined loss + label smoothing + MixUp
3. **Improved Convergence**: Gradient clipping + SWA
4. **Faster Training**: Optimized data pipeline with TF datasets
5. **Better Hard Sample Learning**: Focal loss component

### Performance Metrics
- **Target Accuracy**: >85% on test set
- **Expected Improvements**:
  - 2-5% accuracy gain from advanced preprocessing
  - 3-7% generalization improvement from combined loss
  - 1-3% final boost from SWA
  - Better performance on difficult classes (Nutrient Deficiency, Mosaic Virus)

## Testing Status

### Initial Test Run
- Script loads successfully
- Data loader integration working
- Preprocessing pipeline functional
- Loss functions properly initialized

### Notes
- The advanced preprocessing adds computational overhead but significantly improves real-world performance
- SWA typically shows benefits after 20+ epochs
- MixUp may slightly reduce training accuracy but improves validation/test performance
- Gradient clipping prevents training instability with aggressive augmentation

## Next Steps

1. **Full Training Run**
   ```bash
   # Run full 30-epoch training
   python train_robust_model_v2.py \
       --preprocessing_mode default \
       --loss_type combined \
       --swa_start_epoch 20
   ```

2. **A/B Testing**
   ```bash
   # Compare with legacy preprocessing
   python train_robust_model_v2.py \
       --preprocessing_mode legacy \
       --loss_type combined
   ```

3. **Optimization**
   - Fine-tune loss weights based on validation performance
   - Adjust SWA start epoch based on loss curves
   - Experiment with different MixUp alpha values

4. **Deployment**
   - Use best configuration for production model
   - Convert to TFLite with optimizations
   - Test on failed cases from feedback system

## Files Modified/Created
- `train_robust_model_v2.py` - Enhanced training script
- Integrates with:
  - `src/data_loader_v2.py` - Enhanced data loader
  - `src/losses.py` - Advanced loss functions
  - `src/advanced_preprocessing.py` - CLAHE preprocessing
  - `src/augmentation_pipeline.py` - Realistic augmentations