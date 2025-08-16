# PlantPulse: Advanced Plant Disease Detection System - Development Specification

## Project Overview

PlantPulse is an advanced plant health monitoring system combining RGB and thermal imaging to detect plant diseases with high accuracy. The system addresses the critical challenge of domain gap between laboratory-trained models and real-world deployment, achieving robust performance on internet-sourced images through advanced preprocessing and domain adaptation techniques.

## Current Implementation Status

### RGB Model Enhancement Pipeline (Active Development)

#### Baseline Performance Issue
- **Training Accuracy**: 95% on PlantVillage/PlantDoc datasets
- **Real-World Performance**: 50-60% on internet-sourced images
- **Domain Gap**: 35-45% accuracy drop due to distribution shift
- **Root Cause**: Differences in lighting, camera quality, backgrounds, and environmental conditions

#### Enhanced Implementation (In Progress)
**Status:** OPTIMIZATION PHASE - Targeting 85-90% real-world accuracy  
**Approach:** Multi-phase enhancement pipeline addressing domain shift  

**Progress:**
- ✅ Phase 1: Advanced CLAHE preprocessing + realistic augmentation (COMPLETE)
- 🔄 Phase 2: Focal Loss + Label Smoothing + SWA (IN PROGRESS)
- 📋 Phase 3: Test-Time Augmentation optimization (NEXT)
- 📋 Phase 4: Domain Adaptation if needed (FUTURE)

**Expected Outcome:** 80-85% accuracy on internet-sourced plant images

## Technical Architecture

### Core Components

#### 1. Advanced Preprocessing Pipeline (Phase 1 Complete)
**File:** `rgb_model/src/advanced_preprocessing.py`

**Features:**
- **CLAHE Enhancement**: Adaptive histogram equalization in LAB space (clip_limit=3.0, tile_grid_size=8x8)
- **Illumination Correction**: Gaussian blur background subtraction for uniform lighting
- **Bilateral Filtering**: Noise reduction while preserving disease-relevant edges (d=9)
- **Color Constancy**: Gray world normalization for camera sensor variations
- **Multiple Processing Modes**:
  - Default: Full pipeline with all enhancements
  - Fast: Optimized for real-time processing
  - Minimal: Basic operations only
  - Legacy: Backward compatibility mode

#### 2. Augmentation Pipeline
**File:** `rgb_model/src/augmentation_pipeline.py`

**Realistic Training Augmentations:**
- **Camera Quality Simulation**: Gaussian noise, ISO noise (10-50 var_limit)
- **Motion/Focus Issues**: Motion blur, median blur, defocus effects
- **Environmental Conditions**: Rain, sun flare, shadows, fog (20% probability)
- **JPEG Compression**: Quality range 60-100 to simulate internet images
- **Geometric Transforms**: Rotation, scaling, cropping for viewpoint invariance
- **Built-in TTA Support**: Test-time augmentation for robust inference

#### 3. Enhanced Data Loading
**File:** `rgb_model/src/data_loader_v2.py`

**Capabilities:**
- Toggle between advanced and legacy preprocessing
- A/B testing functionality for optimization
- TensorFlow dataset integration with prefetching
- Preprocessing caching for performance
- Automatic augmentation for training data only
- Support for multiple data formats and sources

#### 4. Advanced Loss Functions (Phase 2)
**File:** `rgb_model/src/losses.py`

**Implemented Losses:**
- **Focal Loss**: Addresses class imbalance (alpha=1, gamma=2)
- **Label Smoothing**: Prevents overconfident predictions (epsilon=0.1)
- **Combined Loss**: 70% Focal + 30% Label Smoothing
- **CORAL Loss**: Feature distribution alignment for domain adaptation

### Model Architecture

#### RGB Disease Detection Model
**Base Architecture:** MobileNetV3-Small / EfficientNetB0  
**Input Size:** 224×224×3 RGB images  
**Output:** 7-class disease classification  

**Universal Disease Categories:**
1. **Healthy** - Normal, disease-free plants
2. **Blight** - Early/Late blight diseases
3. **Leaf Spot** - Bacterial and fungal spot diseases
4. **Powdery Mildew** - White fungal growth
5. **Rust** - Orange/brown fungal infections
6. **Mosaic Virus** - Viral infections with mosaic patterns
7. **Nutrient Deficiency** - Nitrogen, phosphorus, potassium deficiencies

**Architecture Enhancements:**
- SE blocks for channel attention
- Spatial attention mechanisms (planned)
- Multi-scale feature extraction
- DropBlock regularization
- Progressive training strategy (3 stages)

### Training Strategy

#### Current Implementation (Phase 2)
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: Cosine annealing with warm restarts
- **Stochastic Weight Averaging**: Starting epoch 20
- **MixUp Augmentation**: 50% probability during training
- **Gradient Clipping**: max_norm=1.0
- **Class Weighting**: Computed inverse frequency weights
- **Early Stopping**: Patience=15 with best model checkpointing

#### Progressive Training Stages
1. **Stage 1**: Feature extraction (frozen backbone) - 15 epochs
2. **Stage 2**: Partial fine-tuning (top 30 layers) - 20 epochs
3. **Stage 3**: Full fine-tuning (all layers) - 10 epochs

## Performance Metrics

### Current Performance
- **Baseline Validation**: 95% on PlantVillage/PlantDoc
- **Baseline Real-World**: 50-60% on internet images
- **After Phase 1**: 65-70% on internet images (estimated)
- **Target After Phase 2**: 75-80% on internet images
- **Final Target**: 80-85% with full pipeline

### Model Specifications
- **Parameters**: ~4.4M (EfficientNetB0) / ~1.6M (Custom CNN)
- **Model Size**:
  - Keras (.h5): ~18MB
  - TFLite Float16: ~9MB
  - TFLite INT8: ~4.5MB (target)
- **Inference Speed**: <100ms on mobile CPU
- **Memory Usage**: <500MB during inference

### Performance Benchmarks
| Metric | Training Set | Validation Set | Internet Images | With TTA |
|--------|-------------|----------------|-----------------|----------|
| Baseline | 95% | 92% | 50-60% | N/A |
| + Preprocessing | 94% | 91% | 65-70% | +2-3% |
| + New Losses | 93% | 90% | 70-75% | +2-3% |
| + Domain Adapt | 92% | 89% | 75-80% | +2-3% |
| **Final Target** | 92% | 89% | 80-85% | 82-87% |

## Project Structure

```
farmFlowApp/
├── rgb_model/
│   ├── src/
│   │   ├── advanced_preprocessing.py    # CLAHE preprocessing pipeline
│   │   ├── augmentation_pipeline.py     # Realistic augmentation
│   │   ├── data_loader_v2.py           # Enhanced data loader
│   │   ├── losses.py                   # Focal + Label Smoothing
│   │   ├── model_robust.py             # Model architecture
│   │   ├── test_time_augmentation.py   # TTA implementation
│   │   └── domain_adaptation.py        # DANN implementation (future)
│   ├── train_robust_model.py           # Main training script
│   ├── inference_real_world.py         # Real-world inference
│   ├── evaluate_real_world.py          # Evaluation suite
│   └── requirements.txt                # Dependencies
├── PlantPulse/                          # React Native app
├── deepresearch.md                      # Detailed research findings
└── CLAUDE.md                           # This document
```

## Data Pipeline

### Dataset Composition
- **PlantVillage**: 50,000+ controlled condition images
- **PlantDoc**: 2,500+ field condition images  
- **Internet Augmentation**: Unlabeled real-world images (planned)
- **Total Training Size**: ~55,000 images after augmentation

### Data Processing Pipeline
1. **Raw Image Input** → Various resolutions, formats, quality levels
2. **Advanced Preprocessing** → Standardized lighting, contrast, noise reduction
3. **Augmentation** (Training only) → Realistic variations simulation
4. **Normalization** → ImageNet statistics [0.485, 0.456, 0.406]
5. **Model Input** → 224×224×3 tensors

### Class Distribution Handling
```python
class_weights = {
    'Healthy': 1.0,
    'Blight': 1.2,
    'Leaf_Spot': 0.9,
    'Powdery_Mildew': 1.1,
    'Rust': 1.8,         # Underrepresented
    'Mosaic_Virus': 1.5,
    'Nutrient_Deficiency': 2.3  # Most underrepresented
}
```

## Implementation Timeline

### Phase 1: Advanced Preprocessing ✅ COMPLETE
**Duration:** 2 days  
**Deliverables:**
- ✅ CLAHE-based preprocessing module
- ✅ Realistic augmentation pipeline  
- ✅ Enhanced data loader with A/B testing
- ✅ Multiple preprocessing modes
- ✅ TTA support integration

**Impact:** +10-15% improvement on real-world images

### Phase 2: Loss Functions & Training 🔄 IN PROGRESS
**Duration:** 1-2 days  
**Deliverables:**
- ✅ Focal Loss implementation
- ✅ Label Smoothing implementation
- 🔄 Combined loss integration
- 🔄 Stochastic Weight Averaging
- 🔄 Training with EnhancedDataLoader
- ⏳ Validation on diverse test sets

**Expected Impact:** +5-8% additional improvement

### Phase 3: Inference Optimization 📋 NEXT
**Duration:** 1 day  
**Deliverables:**
- 📋 Production inference script with TTA
- 📋 Real-world evaluation suite
- 📋 Confidence calibration
- 📋 Performance benchmarking
- 📋 Mobile deployment optimization

**Expected Impact:** +2-5% from TTA

### Phase 4: Domain Adaptation 📋 FUTURE (If Needed)
**Duration:** 3-4 days  
**Deliverables:**
- 📋 DANN implementation with gradient reversal
- 📋 Unlabeled internet image collection
- 📋 CORAL loss integration
- 📋 Cross-domain validation
- 📋 Final model optimization

**Expected Impact:** +5-10% if initial phases insufficient

## Deployment Strategy

### Mobile Deployment (React Native)
```typescript
interface ModelDeployment {
  format: 'tflite';
  quantization: 'float16' | 'int8';
  size: '<10MB';
  inference_time: '<100ms';
  preprocessing: 'on-device';
  tta_enabled: boolean;
}
```

### Web Deployment (PWA)
- **Framework**: React with TypeScript
- **ML Runtime**: TensorFlow.js
- **Features**: Offline support, camera access, real-time inference
- **Compatibility**: All modern browsers with WebRTC

### API Deployment
- **Framework**: FastAPI
- **Serving**: TensorFlow Serving / TorchServe
- **Features**: Batch inference, async processing, result caching
- **Scaling**: Kubernetes with auto-scaling

## Key Algorithms

### CWSI Calculation (Water Stress)
```python
def calculate_cwsi(leaf_temp: float, air_temp: float, vpd: float) -> float:
    """Calculate Crop Water Stress Index"""
    wet_leaf_temp = air_temp - (2.0 * vpd)
    dry_leaf_temp = air_temp + 5.0
    cwsi = (leaf_temp - wet_leaf_temp) / (dry_leaf_temp - wet_leaf_temp)
    return np.clip(cwsi, 0, 1)
```

### Test-Time Augmentation
```python
def predict_with_tta(image, model, n_augments=5):
    """Average predictions across augmented versions"""
    predictions = []
    for aug in augmentations[:n_augments]:
        augmented = aug(image)
        preprocessed = preprocess(augmented)
        pred = model.predict(preprocessed)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
```

### Domain Adaptation Loss
```python
def coral_loss(source_features, target_features):
    """Correlation Alignment loss"""
    d = source_features.shape[1]
    source_cov = cov(source_features)
    target_cov = cov(target_features)
    loss = torch.norm(source_cov - target_cov, 'fro') ** 2
    return loss / (4 * d * d)
```

## Testing & Validation

### Test Categories
1. **Unit Tests**: Preprocessing functions, augmentation pipeline, loss calculations
2. **Integration Tests**: Data loading, model training, inference pipeline
3. **Performance Tests**: Inference speed, memory usage, battery consumption
4. **Robustness Tests**: Various lighting, angles, image qualities
5. **Field Tests**: Real plants with known conditions

### Evaluation Metrics
- **Accuracy**: Overall and per-class
- **Precision/Recall**: For each disease category
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Error analysis
- **Confidence Calibration**: Expected Calibration Error (ECE)
- **Inference Time**: Average and percentiles
- **Model Size**: Compressed and quantized

### Real-World Test Protocol
1. Collect 100+ internet images per disease category
2. Apply preprocessing pipeline
3. Run inference with and without TTA
4. Compare against expert annotations
5. Analyze failure cases
6. Iterate on preprocessing/augmentation

## Success Criteria

### Primary Goals
- ✅ Identify and solve domain gap issue
- 🔄 Achieve 80%+ accuracy on internet images
- 🔄 Maintain <100ms inference time
- 📋 Deploy to mobile and web platforms
- 📋 Support offline operation

### Performance Requirements
- **Accuracy**: >80% on real-world images
- **Speed**: <100ms per image on mobile CPU
- **Size**: <10MB TFLite model
- **Battery**: <5% drain per 100 inferences
- **Memory**: <500MB peak usage

### User Experience Goals
- Simple one-tap disease detection
- Clear confidence indicators
- Actionable care recommendations
- Historical tracking capability
- Works with any smartphone camera

## Future Enhancements

### Thermal Integration
- USB-C thermal camera support (InfiRay P2 Pro)
- Early stress detection (3-7 days before visible)
- CWSI calculation for water stress
- Temperature pattern analysis

### Advanced Features
- Multi-plant batch analysis
- Time-series disease progression
- Pest detection addition
- Fertilizer recommendation system
- Weather integration for predictions

### Model Improvements
- Knowledge distillation for smaller models
- Federated learning for privacy
- Active learning for continuous improvement
- Multi-modal fusion (RGB + thermal + spectral)
- Explainable AI visualizations

## Technical Dependencies

### Core Libraries
```python
tensorflow==2.12.0
albumentations==1.3.0
opencv-python==4.8.0
scikit-image==0.21.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.1
```

### Mobile Dependencies
```json
{
  "react-native": "0.72.0",
  "react-native-fast-tflite": "^1.0.0",
  "react-native-vision-camera": "^3.0.0"
}
```

## Notes & Considerations

### Current Challenges
- Domain gap between lab and field conditions (being addressed)
- Class imbalance in training data (handled with weights)
- Variable image quality from different devices (preprocessing helps)
- Limited thermal camera availability (RGB-first approach)

### Best Practices
- Always validate on real-world images, not just validation set
- Use progressive training to prevent catastrophic forgetting
- Implement confidence thresholds for production deployment
- Monitor model drift in production
- Collect user feedback for continuous improvement

### Privacy & Ethics
- All processing happens on-device
- No image data sent to servers
- User consent for any data collection
- Transparent about model limitations
- Regular bias auditing

## Contact & Resources

### Documentation
- GitHub Repository: [farmFlowApp](https://github.com/learn-programing1223/farmFlowApp)
- Research Paper: deepresearch.md
- API Documentation: (To be created)
- User Guide: (To be created)

### Related Resources
- PlantVillage Dataset: Penn State University
- PlantDoc Dataset: Singh et al.
- Domain Adaptation Papers: See deepresearch.md
- TensorFlow Lite: Official documentation

---

*Last Updated: Current Phase 2 Implementation*  
*Version: 2.0 - Enhanced with Domain Adaptation Pipeline*