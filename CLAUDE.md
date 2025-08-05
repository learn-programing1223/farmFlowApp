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

### RGB Model Specifications

#### Universal Disease Categories
Instead of crop-specific diseases, the RGB model will detect generalized disease types:
- **Blight** (covers early/late blight across all crops)
- **Leaf Spot** (bacterial/fungal spots)
- **Powdery Mildew**
- **Rust**
- **Mosaic Virus**
- **Nutrient Deficiency** (general yellowing/discoloration)
- **Pest Damage**
- **Healthy**

#### Training Strategy
1. **Multi-Dataset Approach**:
   - PlantVillage Dataset (primary)
   - PlantDoc Dataset (20+ crops, real-world conditions)
   - PlantNet Dataset (field images)
   - Kaggle Plant Pathology datasets
   - Custom augmented dataset using CycleGAN

2. **Data Preprocessing**:
   - Map specific diseases to universal categories
   - Balance classes across different crop types
   - Apply aggressive augmentation for generalization
   - Target: 80% validation accuracy

3. **Model Architecture**:
   - EfficientNet-B0 backbone (lightweight, mobile-friendly)
   - Global Average Pooling + Dense layers
   - Multi-scale feature extraction
   - Model size < 20MB after quantization

### Web Implementation Architecture

#### Dual-Model Deployment
```typescript
interface ModelConfig {
  thermal: {
    enabled: boolean;
    model: 'plant_health_thermal_v1.tflite';
    inputSize: [256, 192];
    requiresUSBCamera: true;
  };
  rgb: {
    enabled: boolean;
    model: 'plant_health_rgb_universal_v1.tflite';
    inputSize: [224, 224];
    requiresUSBCamera: false;
  };
}

interface AnalysisMode {
  type: 'thermal' | 'rgb' | 'hybrid';
  primaryModel: 'thermal' | 'rgb';
  fallbackModel?: 'thermal' | 'rgb';
}
```

#### Web Interface Features
1. **Model Toggle**: Users can switch between thermal/RGB analysis
2. **Camera Detection**: Auto-detect available cameras and suggest best mode
3. **Hybrid Analysis**: When thermal camera is available, combine both models for higher accuracy
4. **Progressive Enhancement**: RGB-only on standard devices, full features with thermal

#### Deployment Strategy
1. **Frontend**: React web app with responsive design
2. **Model Serving**: TensorFlow.js for in-browser inference
3. **PWA Features**: Offline capability, camera access
4. **Hosting**: Vercel/Netlify with CDN for model files

### Implementation Timeline
1. **Phase 1**: RGB model training with universal categories (Week 1-2)
2. **Phase 2**: Web interface with model toggle (Week 3)
3. **Phase 3**: Integration of both models (Week 4)
4. **Phase 4**: Testing and optimization (Week 5)

### Hybrid Analysis Benefits
- **Thermal**: Detects stress 3-7 days early, precise water/nutrient analysis
- **RGB**: Universal access, visible disease detection, pest identification
- **Combined**: Cross-validation, higher confidence, complete plant health picture