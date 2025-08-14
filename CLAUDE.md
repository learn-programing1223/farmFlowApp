# PlantPulse: Thermal Camera Plant Health Monitoring App - Development Prompt

## Project Context

I need you to build PlantPulse, a React Native application that uses USB-C thermal cameras (InfiRay P2 Pro or TOPDON TC002C) to detect plant health issues before they become visible to the naked eye. The app should work primarily on iOS devices with USB-C ports (iPhone 15+) and provide real-time analysis of plant water stress, diseases, and nutrient deficiencies using thermal imaging and machine learning.

## Key Technical Requirements

### 1. Thermal Camera Integration
- Implement USB thermal camera support using react-native-uvc-camera for the InfiRay P2 Pro ($299) and TOPDON TC002C ($270)
- These cameras appear as USB Video Class devices with 256x384 resolution
- Temperature data is embedded in the bottom portion of each frame and needs to be extracted
- Parse the Realtek bridge chip dual-stream format (visual + temperature data)
- Handle camera connection/disconnection gracefully

### 2. Machine Learning Pipeline
- Use TensorFlow Lite via react-native-fast-tflite for on-device inference
- Implement a modified MobileNetV3-Small architecture for thermal image analysis
- Model should detect:
  - Water stress (CWSI > 0.36 indicates stress)
  - Disease presence (bacterial/fungal/viral)
  - Nutrient deficiencies
- Target inference speed: <100ms per frame with GPU acceleration
- Model size should be <50MB after INT8 quantization

### 3. Temperature Pattern Detection
Implement detection algorithms for these scientifically-validated patterns:
- **Water Stress**: Leaf temperature 1.5-6°C above baseline
- **Healthy Plants**: Maintain 6-15°C below ambient temperature
- **Disease Signatures**:
  - Biotrophic fungi: -2.5 to +2K temperature change
  - Necrotrophic fungi: -5 to +9K temperature change
  - Bacterial infections: Initial cooling followed by heating

### 4. User Interface
Create an intuitive interface with:
- Live thermal camera viewfinder with temperature overlay
- Real-time health status indicators (green/yellow/red)
- Spot temperature measurement tool
- Historical tracking for monitored plants
- Care recommendations based on detected issues
- Plant profile system for species-specific thresholds

### 5. Performance Optimization
- Implement adaptive frame processing (5 FPS stable, 25 FPS when detecting changes)
- Use pre-allocated memory buffers for thermal data
- GPU acceleration for all image processing
- Battery optimization with smart processing schedules

## Project Structure

```
plantpulse/
├── src/
│   ├── camera/
│   │   ├── ThermalCameraManager.ts
│   │   ├── TemperatureParser.ts
│   │   └── USBDeviceHandler.ts
│   ├── ml/
│   │   ├── PlantHealthModel.ts
│   │   ├── ThermalPreprocessor.ts
│   │   └── models/
│   │       └── plant_health_v1.tflite
│   ├── analysis/
│   │   ├── WaterStressDetector.ts
│   │   ├── DiseaseClassifier.ts
│   │   └── NutrientAnalyzer.ts
│   ├── ui/
│   │   ├── screens/
│   │   │   ├── CameraScreen.tsx
│   │   │   ├── AnalysisScreen.tsx
│   │   │   └── PlantLibraryScreen.tsx
│   │   └── components/
│   │       ├── ThermalOverlay.tsx
│   │       ├── HealthIndicator.tsx
│   │       └── TemperatureSpotMeter.tsx
│   └── utils/
│       ├── ThermalMath.ts
│       └── PlantDatabase.ts
├── android/
├── ios/
└── package.json
```

## Implementation Steps

### Step 1: Initialize React Native Project
```bash
npx react-native init PlantPulse --template react-native-template-typescript
cd PlantPulse
npm install react-native-vision-camera react-native-fast-tflite react-native-uvc-camera
npm install @react-navigation/native @react-navigation/stack react-native-paper
```

### Step 2: Thermal Camera Integration
Create a robust thermal camera manager that:
1. Detects USB thermal cameras using UVC protocol
2. Initializes video stream at 256x384 resolution
3. Extracts temperature data from the frame buffer
4. Converts raw thermal values to Celsius temperatures
5. Handles device rotation and disconnection

### Step 3: ML Model Integration
1. Convert the TensorFlow model to TFLite format with INT8 quantization
2. Implement preprocessing to normalize thermal images (0-1 range)
3. Set up multi-task inference for water stress, disease, and nutrients
4. Create post-processing to convert model outputs to user-friendly results

### Step 4: Real-time Analysis Pipeline
Build a frame processor that:
1. Receives thermal frames at 25 FPS
2. Extracts temperature matrix (256x192)
3. Runs ML inference on GPU
4. Updates UI with results in real-time
5. Triggers alerts for critical plant health issues

### Step 5: User Interface Development
Design screens following Material Design 3 guidelines:
1. **Camera Screen**: Live thermal view with overlay graphics
2. **Analysis Screen**: Detailed health metrics and recommendations
3. **Plant Library**: Species-specific care information
4. **History Screen**: Track plant health over time
5. **Settings Screen**: Calibration and preferences

## Data Models

```typescript
interface ThermalFrame {
  temperatureData: Float32Array; // 256x192 temperature values in Celsius
  timestamp: number;
  deviceId: string;
  calibrationOffset: number;
}

interface PlantHealthAnalysis {
  waterStressIndex: number; // 0-1 scale, >0.36 indicates stress
  stressLevel: 'none' | 'mild' | 'moderate' | 'severe';
  diseaseDetection: {
    type: 'healthy' | 'bacterial' | 'fungal' | 'viral';
    confidence: number;
    affectedArea: number; // percentage
  };
  nutrientStatus: {
    nitrogen: 'deficient' | 'optimal' | 'excess';
    phosphorus: 'deficient' | 'optimal' | 'excess';
    potassium: 'deficient' | 'optimal' | 'excess';
  };
  recommendations: string[];
  timestamp: number;
}

interface PlantProfile {
  id: string;
  species: string;
  nickname: string;
  optimalTempRange: { min: number; max: number };
  waterStressThreshold: number;
  lastAnalysis: PlantHealthAnalysis;
  history: PlantHealthAnalysis[];
}
```

## Key Algorithms

### Water Stress Detection (CWSI Calculation)
```typescript
function calculateCWSI(
  leafTemp: number,
  airTemp: number,
  vpd: number // Vapor Pressure Deficit
): number {
  const wetLeafTemp = airTemp - (2.0 * vpd); // Theoretical min temp
  const dryLeafTemp = airTemp + 5.0; // Theoretical max temp
  
  const cwsi = (leafTemp - wetLeafTemp) / (dryLeafTemp - wetLeafTemp);
  return Math.max(0, Math.min(1, cwsi));
}
```

### Disease Pattern Recognition
Implement pattern matching for temperature anomalies:
- Circular patterns often indicate fungal infections
- Linear patterns suggest bacterial spread
- Mosaic patterns may indicate viral infections

## Testing Requirements

1. **Unit Tests**: Temperature calculations, CWSI formula, pattern detection
2. **Integration Tests**: Camera connection, ML inference pipeline
3. **E2E Tests**: Complete user flows from camera to recommendations
4. **Performance Tests**: Ensure <100ms inference, 25 FPS capability
5. **Field Tests**: Validate with real plants showing known conditions

## Deployment Configuration

### iOS Specific Requirements
- Add USB camera permissions to Info.plist
- Configure External Accessory protocols
- Enable Camera Usage Description
- Set up App Groups for data sharing

### Build Optimization
- Enable Hermes for improved performance
- Configure ProGuard rules for Android
- Implement code splitting for features
- Optimize bundle size (<100MB target)

## Alternative Web Implementation

If native development proves challenging, implement as a Progressive Web App:
1. Use WebUSB API for thermal camera access
2. Implement WebAssembly module for thermal processing
3. Deploy TensorFlow.js model for browser inference
4. Use WebRTC for real-time video processing
5. Enable offline functionality with Service Workers

## Training Data Sources

Use these datasets for model training:
- IEEE DataPort Hydroponic Dataset (30 plants, 66 days)
- Generate synthetic thermal data using StyleGAN
- Augment with CycleGAN RGB-to-thermal conversion
- Temperature signature patterns from published research

## Success Criteria

The app should:
1. Detect water stress 3-7 days before visible symptoms
2. Achieve >90% accuracy on water stress detection
3. Process thermal frames in real-time (<100ms latency)
4. Run for >2 hours on battery power
5. Work reliably with both InfiRay P2 Pro and TOPDON TC002C cameras

## Important Implementation Notes

1. The thermal cameras output both visual and temperature data - ensure you parse both streams correctly
2. Temperature accuracy is critical - implement proper calibration routines
3. Different plant species have different optimal temperature ranges - make the system adaptable
4. Consider implementing a "learning mode" where users can train the app on their specific plants
5. Privacy is important - all processing should happen on-device without cloud uploads

Build this app with a focus on scientific accuracy, user-friendly design, and reliable performance. The goal is to make professional-grade plant health monitoring accessible to everyday gardeners and plant enthusiasts.

## Hybrid Model Approach (Thermal + RGB)

### Overview
Due to the requirement of specialized thermal cameras, we're implementing a hybrid approach that combines:
1. **Thermal Model**: For users with USB-C thermal cameras (80% validation accuracy achieved)
2. **RGB Model**: For universal accessibility using standard iPhone cameras

### RGB Model Implementation Status

#### Current Implementation (PlantVillage + PlantDisease + CycleGAN)
**Status:** PRODUCTION READY - 81.9% validation accuracy achieved!  
**Dataset:** 18,586 images (PlantVillage + PlantDisease) with 30% CycleGAN augmentation  
**Architecture:** Custom CNN with 1.59M parameters optimized for field conditions  
**Performance:** 81.9% validation accuracy with excellent generalization  

#### Universal Disease Categories (7 Classes with CycleGAN Enhancement)
The RGB model detects these generalized disease types:
- **Blight** (covers early/late blight across all crops)
- **Leaf Spot** (bacterial/fungal spots)
- **Powdery Mildew**
- **Mosaic Virus**
- **Nutrient Deficiency** (general yellowing/discoloration)
- **Rust** (added from PlantDisease dataset)
- **Healthy**

**CycleGAN Enhancement:** 30% of training images transformed from lab to field-like conditions

#### Current Model Architecture (CycleGAN-Enhanced)
**Custom CNN Architecture:**
- **Convolutional Blocks**: 4 blocks with increasing filters (32→64→128→256)
- **Normalization**: BatchNormalization after each conv layer
- **Pooling**: MaxPooling2D and GlobalAveragePooling2D
- **Dense Layers**: 512 → 256 with BatchNorm and Dropout
- **Regularization**: Dropout (0.25→0.5) progressively increased
- **Total Parameters**: 1,591,975 (optimized for accuracy)
- **Model Size**: <10MB after TFLite INT8 quantization
- **Input Shape**: (224, 224, 3)
- **Training**: Cosine annealing LR, class weighting for imbalanced data

#### Data Quality & Processing
**Combined Dataset Statistics:**
- **Total Images**: 18,586 samples (PlantVillage + PlantDisease)
- **CycleGAN Augmented**: 5,575 images (30% of total)
- **Train/Val/Test Split**: 70/15/15
- **Data Format**: JPEG images at 224x224 resolution
- **Normalization**: [0, 1] range via ImageDataGenerator
- **Class Balance**: Handled via computed class weights (Rust: 5.288x weight)
- **Augmentation**: CycleGAN field transformation + runtime augmentations

### Web & Mobile Implementation Architecture

#### Dual-Model Deployment Strategy
```typescript
interface ModelConfig {
  thermal: {
    enabled: boolean;
    model: 'plant_health_thermal_v1.tflite';
    inputSize: [256, 192];
    requiresUSBCamera: true;
    accuracy: 0.80; // Current validation accuracy
  };
  rgb: {
    enabled: boolean;
    model: 'plant_disease_final.tflite';
    inputSize: [224, 224];
    requiresUSBCamera: false;
    accuracy: 0.85; // Target accuracy
    size: '9MB'; // TFLite Float16
    classes: ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 'Nutrient_Deficiency', 'Powdery_Mildew'];
  };
}

interface PlantAnalysisResult {
  diseaseType: string;
  confidence: number;
  uncertainty?: number; // From TTA
  recommendations: string[];
  affectedArea?: number; // Percentage
  timestamp: string;
  modelUsed: 'thermal' | 'rgb' | 'hybrid';
}
```

#### React Native Integration (PlantPulse App)
**File Locations:**
- `PlantPulse/src/ml/RGBDiseaseModel.ts` - TensorFlow Lite integration
- `PlantPulse/src/ui/screens/RGBCameraScreen.tsx` - RGB camera interface
- `PlantPulse/src/ui/screens/UnifiedCameraScreen.tsx` - Hybrid thermal+RGB

**Implementation Features:**
- **TensorFlow Lite**: Use `react-native-fast-tflite` for mobile inference
- **Camera Integration**: Standard device cameras via `react-native-vision-camera`
- **Real-time Analysis**: Frame processing at 5-10 FPS
- **Offline Capability**: All processing on-device
- **Model Switching**: Dynamic thermal/RGB model selection

```typescript
// Example integration
class RGBDiseaseModel {
  private model: any;
  private classes = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 'Nutrient_Deficiency', 'Powdery_Mildew'];
  
  async loadModel() {
    this.model = await TensorFlowLite.loadModel('plant_disease_final.tflite');
  }
  
  async predict(imageUri: string): Promise<PlantAnalysisResult> {
    // Preprocess image to 224x224
    // Run inference
    // Apply confidence thresholding
    // Return structured result
  }
}
```

#### Progressive Web App (PWA) Deployment
**For Universal Access:**
- **Framework**: React with TypeScript
- **ML Backend**: TensorFlow.js conversion of TFLite model
- **Camera Access**: WebRTC getUserMedia API
- **Offline Support**: Service Workers with model caching
- **Responsive Design**: Works on phones, tablets, desktops

**Deployment Configuration:**
```json
{
  "pwa": {
    "name": "PlantPulse Disease Detector",
    "short_name": "PlantPulse",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#2E7D32",
    "theme_color": "#4CAF50",
    "icons": [...],
    "permissions": ["camera", "storage"]
  }
}
```

#### Hybrid Analysis Implementation
**When Both Models Available:**
1. **Thermal First**: Check for early stress indicators
2. **RGB Confirmation**: Validate with visible symptoms
3. **Cross-Validation**: Compare predictions for confidence
4. **Combined Confidence**: Weighted average based on model certainty

```typescript
interface HybridAnalysis {
  thermal: {
    waterStress: number; // CWSI index
    temperature: number;
    confidence: number;
  };
  rgb: {
    diseaseType: string;
    confidence: number;
    uncertainty: number;
  };
  combined: {
    recommendation: string;
    urgency: 'low' | 'medium' | 'high';
    confidence: number; // Weighted combination
  };
}
```

### Production Deployment Timeline
**Current Status: RGB Model DEPLOYED - 81.9% Accuracy**

**Phase 1 (Completed)**: RGB Model Development
- ✅ PlantVillage + PlantDisease integration (18,586 images)
- ✅ CycleGAN augmentation implemented (5,575 field-like images)
- ✅ Custom CNN architecture (1.59M parameters)
- ✅ 81.9% validation accuracy achieved
- ✅ TFLite conversion working
- ✅ Production model ready for deployment

**Phase 2 (In Progress)**: Mobile Integration
- 🔄 React Native TFLite integration
- 🔄 Camera preprocessing pipeline
- 🔄 Real-time inference optimization
- 🔄 User interface development

**Phase 3 (Next)**: Web Application
- 📋 TensorFlow.js model conversion
- 📋 PWA development with React
- 📋 Camera capture and preprocessing
- 📋 Responsive UI for mobile/desktop

**Phase 4 (Future)**: Hybrid Implementation
- 📋 Thermal model integration
- 📋 Dual-model analysis pipeline
- 📋 Advanced confidence scoring
- 📋 Field testing and validation

### Hybrid Analysis Benefits
- **Thermal**: Detects stress 3-7 days early, precise water/nutrient analysis
- **RGB**: Universal access, visible disease detection, pest identification
- **Combined**: Cross-validation, higher confidence, complete plant health picture

## RGB Model Production Pipeline

### Development Workflow
The RGB model implementation follows a streamlined 5-step production pipeline:

#### 1. Data Preparation (`prepare_plantvillage_data.py`)
**Purpose:** Download and preprocess PlantVillage dataset
**Features:**
- Automated PlantVillage dataset download (87GB → 8,452 processed images)
- Universal disease category mapping (38 specific → 6 universal classes)
- Intelligent train/validation/test splitting (70/15/15)
- Quality validation and corrupt image filtering
- NumPy array conversion for efficient loading

```python
# Usage
python prepare_plantvillage_data.py
# Output: datasets/plantvillage_processed/{train,val,test}/
```

#### 2. Model Training (`train_robust_plantvillage.py`)
**Purpose:** Train production-ready EfficientNetB0 model
**Key Features:**
- **Architecture**: EfficientNetB0 with spatial attention
- **Augmentation**: Agricultural-specific transformations (rotation, brightness, MixUp)
- **Class Weighting**: Automatic handling of imbalanced classes
- **Mixed Precision**: GPU acceleration with reduced memory usage
- **Advanced Callbacks**: Early stopping, cosine annealing, TensorBoard logging
- **Target Performance**: >85% accuracy, <10MB model size

```python
# Configuration
config = {
    'model_architecture': 'efficientnetb0',
    'batch_size': 32,
    'epochs': 50,
    'initial_lr': 1e-3,
    'use_class_weights': True,
    'use_mixed_precision': True
}
```

#### 3. Comprehensive Evaluation (`comprehensive_real_world_test.py`)
**Purpose:** Thorough model validation with real-world conditions
**Testing Features:**
- **Test-Time Augmentation**: 10 augmented predictions averaged
- **Uncertainty Quantification**: Prediction variance analysis
- **Robustness Testing**: Noise, blur, lighting variations
- **Confusion Matrix**: Detailed per-class performance
- **Classification Report**: Precision, recall, F1-score per class

```python
# Key capabilities
- Simulated real-world conditions
- Internet image testing
- Performance degradation analysis
- Confidence thresholding recommendations
```

#### 4. Model Deployment (`convert_and_deploy.py`)
**Purpose:** Convert to mobile-optimized formats
**Conversion Pipeline:**
- **Float16 Quantization**: 50% size reduction, minimal accuracy loss
- **INT8 Quantization**: 75% size reduction (experimental)
- **TensorFlow Lite**: Mobile-ready format
- **Model Validation**: Accuracy preservation verification

```python
# Output formats
- plant_disease_final.h5      # Full Keras model
- plant_disease_final.tflite  # Mobile deployment
- plant_disease_quantized.tflite  # Ultra-compact (INT8)
```

#### 5. Deployment Verification (`verify_deployed_model.py`)
**Purpose:** End-to-end deployment testing
**Verification Steps:**
- TFLite model loading and inference
- Accuracy comparison (Keras vs TFLite)
- Performance benchmarking (inference time)
- Mobile compatibility testing

### Directory Structure After Cleanup

```
rgb_model/
├── Core Pipeline (Production Ready)
│   ├── prepare_plantvillage_data.py    # Step 1: Data preparation
│   ├── train_robust_plantvillage.py    # Step 2: Model training  
│   ├── comprehensive_real_world_test.py # Step 3: Evaluation
│   ├── convert_and_deploy.py           # Step 4: Model conversion
│   └── verify_deployed_model.py        # Step 5: Deployment verification
│
├── Supporting Scripts
│   ├── START_HERE.py                   # User entry point and menu
│   ├── download_plantvillage.py        # Dataset downloader
│   ├── analyze_plantvillage_data.py    # Data analysis utilities
│   └── bestSolution.py                 # Proven working backup
│
├── Data Structure
│   ├── datasets/plantvillage_processed/
│   │   ├── train/ (6 class folders)
│   │   ├── val/ (6 class folders)
│   │   └── test/ (6 class folders)
│   └── data/splits/ (NumPy arrays)
│       ├── X_train.npy, y_train.npy
│       ├── X_val.npy, y_val.npy
│       └── X_test.npy, y_test.npy
│
├── Model Outputs
│   ├── models/robust_plantvillage_best.h5
│   ├── models/robust_plantvillage.tflite
│   ├── models/confusion_matrix.png
│   └── models/evaluation_results.json
│
└── Legacy Archive
    └── legacy_archive/ (22 archived files)
```

### Performance Metrics & Targets

#### Model Specifications
- **Input Size**: 224×224×3 RGB images
- **Output**: 6-class disease classification
- **Architecture**: EfficientNetB0 + custom head
- **Parameters**: 4.4M (mobile-optimized)
- **Model Size**: 
  - Keras (.h5): ~18MB
  - TFLite Float16: ~9MB
  - TFLite INT8: ~4.5MB (target)

#### Performance Targets
- **Accuracy**: >85% (validated on test set)
- **Inference Speed**: <100ms on mobile CPU
- **Memory Usage**: <500MB during inference
- **Battery Impact**: Minimal with efficient preprocessing

#### Class Distribution Handling
**Challenge**: Nutrient Deficiency class underrepresented  
**Solution**: Computed class weights with inverse frequency
```python
class_weights = {
    'Blight': 1.0,
    'Healthy': 1.2, 
    'Leaf_Spot': 0.8,
    'Mosaic_Virus': 1.4,
    'Nutrient_Deficiency': 2.1,  # Weighted higher
    'Powdery_Mildew': 1.0
}
```

### Advanced Features

#### Agricultural-Specific Augmentation
```python
augmentation_pipeline = [
    RandomFlip("horizontal"),           # Natural leaf orientations
    RandomRotation(0.15),              # Various camera angles  
    RandomZoom(0.1),                   # Different distances
    RandomBrightness(0.2),             # Outdoor lighting conditions
    RandomContrast(0.15),              # Shadow variations
    MixUp(alpha=0.2, probability=0.3)  # Advanced regularization
]
```

#### Test-Time Augmentation (TTA)
- **Method**: 10 augmented predictions averaged
- **Benefit**: 2-3% accuracy improvement
- **Uncertainty**: Prediction variance as confidence measure
- **Real-world**: Robust to lighting/angle variations

#### Transfer Learning Strategy
- **Phase 1**: Freeze 80% of EfficientNetB0 layers
- **Phase 2**: Gradual unfreezing with lower learning rates
- **Phase 3**: End-to-end fine-tuning with cosine annealing
- **Optimization**: AdamW with weight decay for better generalization

## Troubleshooting & Common Issues

### Setup Issues
**Problem**: TensorFlow installation conflicts  
**Solution**: Use Python 3.11 with TensorFlow 2.12.0
```bash
pip install tensorflow==2.12.0 tensorflow-datasets==4.9.2
```

**Problem**: CUDA/GPU not detected  
**Solution**: CPU training supported, expect 2-3x longer training time

**Problem**: Memory allocation warnings  
**Solution**: Reduce batch_size from 32 to 16 or 8

### Data Issues
**Problem**: PlantVillage download timeout  
**Solution**: Use `download_plantvillage.py` with retry mechanism

**Problem**: Corrupted images in dataset  
**Solution**: Automatic filtering in `prepare_plantvillage_data.py`

**Problem**: Class imbalance warnings  
**Solution**: Class weights automatically calculated and applied

### Training Issues  
**Problem**: Low accuracy (<50%)  
**Solutions**:
- Check data normalization ([0,1] range required)
- Verify class mapping correctness
- Reduce learning rate to 1e-4
- Increase epochs to 100

**Problem**: Overfitting (train > val accuracy)  
**Solutions**:
- Increase dropout rates (0.5 → 0.7)
- Add more augmentation
- Reduce model complexity
- Early stopping with patience=15

**Problem**: Training crashes  
**Solutions**:
- Disable mixed precision: `mixed_precision.set_global_policy('float32')`
- Reduce batch size to 8
- Check available disk space (>10GB required)

### Deployment Issues
**Problem**: TFLite conversion fails  
**Solution**: Use Float16 quantization instead of INT8

**Problem**: Model accuracy drops after conversion  
**Solution**: Test with representative dataset during quantization

**Problem**: Large model size (>10MB)  
**Solutions**:
- Use MobileNetV3Small instead of EfficientNetB0
- Apply more aggressive quantization
- Prune less important connections

### Performance Optimization
**Inference Speed**: 
- Use TFLite instead of full TensorFlow
- Enable GPU acceleration on mobile
- Batch multiple predictions when possible

**Memory Usage**:
- Preload model once, reuse for multiple predictions
- Use image preprocessing pipelines
- Clear intermediate tensors

**Battery Life**:
- Implement smart prediction scheduling
- Use lower resolution inputs (192×192) if acceptable
- Cache frequently used models in memory