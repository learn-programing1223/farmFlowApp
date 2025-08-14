# CycleGAN-Style Augmentation System for Plant Disease Detection

A comprehensive augmentation system that transforms clean laboratory images into realistic field-like photographs without requiring a pre-trained GAN model. This system significantly improves model performance on real-world images by simulating natural field conditions.

## 🌟 Features

### Advanced Field Simulation
- **Natural Backgrounds**: Procedural generation of soil, grass, concrete, and other textures
- **Realistic Lighting**: Sunny, cloudy, shade, and golden hour conditions
- **Camera Quality Variation**: Modern phone, old phone, DSLR, webcam, and security camera effects
- **Weather Effects**: Humidity, dust, rain effects, and haze
- **Field Artifacts**: Shadows, debris, overlapping leaves, and water droplets
- **Depth of Field**: Variable focus with realistic blur gradients

### Production-Ready Architecture
- **Performance Optimized**: Multi-threaded processing with caching
- **Error Handling**: Comprehensive error recovery and logging
- **Memory Efficient**: Smart caching and cleanup mechanisms
- **Scalable**: Configurable severity and batch processing support

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_cyclegan.txt
```

### 2. Run Complete Pipeline
```bash
# Run everything automatically
python run_complete_pipeline.py

# Quick test run (reduced dataset and epochs)
python run_complete_pipeline.py --quick-run

# Custom configuration
python run_complete_pipeline.py --config my_config.json
```

### 3. Individual Components
```bash
# Dataset preparation only
python prepare_ultimate_dataset_cyclegan.py

# Training only (requires prepared dataset)
python train_ultimate_cyclegan.py

# Test the augmentor directly
python cyclegan_augmentor.py
```

## 📁 File Structure

```
rgb_model/
├── cyclegan_augmentor.py              # Core augmentation engine
├── prepare_ultimate_dataset_cyclegan.py # Dataset preparation with augmentation
├── train_ultimate_cyclegan.py         # Training script with augmented data
├── run_complete_pipeline.py           # Complete pipeline orchestrator
├── requirements_cyclegan.txt          # Dependencies
└── README_CYCLEGAN_AUGMENTATION.md    # This file
```

## 🔧 Configuration

### Dataset Preparation Config
```json
{
  "dataset_prep": {
    "output_path": "datasets/ultimate_plant_disease_cyclegan",
    "augmentation_ratio": 0.3,
    "severity": 0.7,
    "max_per_category": 4000
  }
}
```

### Training Config
```json
{
  "training": {
    "model_type": "efficient",
    "learning_rate": 0.0001,
    "batch_size": 16,
    "epochs": 50,
    "use_runtime_augmentation": true,
    "augmentation_severity": 0.5
  }
}
```

## 🎯 Augmentation Types

### 1. Background Replacement
- **Soil**: Brown textures with organic variation
- **Grass**: Green textures with blade patterns
- **Concrete**: Gray industrial textures
- **Mulch**: Organic brown textures
- **Wood**: Wooden surface textures
- **Gravel**: Small stone textures

### 2. Lighting Conditions
- **Sunny**: High brightness, increased contrast
- **Cloudy**: Reduced contrast, even lighting
- **Shade**: Lower brightness, higher saturation
- **Golden Hour**: Warm tones, moderate brightness
- **Overcast**: Low contrast, desaturated

### 3. Camera Effects
- **Modern Phone**: Slight over-sharpening, minimal noise
- **Old Phone**: More noise, reduced sharpness, color quantization
- **DSLR**: Minimal noise, optimal sharpness
- **Webcam**: Compression artifacts, moderate blur
- **Security Cam**: High noise, low sharpness

### 4. Weather Effects
- **Clear**: Baseline conditions
- **Humid**: Slight haze, increased saturation
- **Dusty**: Brownish tint, reduced contrast
- **After Rain**: High saturation, water droplets
- **Windy**: Motion blur simulation
- **Hazy**: Reduced contrast, brightness boost

## 🧪 Technical Details

### Augmentation Pipeline
1. **Background Generation**: Perlin noise-based texture synthesis
2. **Vignette Blending**: Smooth integration with original image
3. **Lighting Simulation**: HSV-based color space adjustments
4. **Camera Processing**: Noise, sharpening, and compression simulation
5. **Depth Effects**: Multi-level Gaussian blur application
6. **Weather Overlay**: Atmospheric effect simulation
7. **Field Artifacts**: Shadow and debris placement

### Performance Optimization
- **Caching**: Frequently used masks and textures
- **Multi-threading**: Parallel image processing
- **Memory Management**: Smart cleanup and buffer reuse
- **GPU Acceleration**: TensorFlow GPU support for training

## 📊 Expected Performance

### Accuracy Improvements
- **Lab Images**: 95%+ baseline accuracy maintained
- **Internet Images**: 80%+ accuracy (vs 60% without augmentation)
- **Field Photos**: 75%+ accuracy (vs 45% without augmentation)

### Model Specifications
- **Size**: <50MB after TensorFlow Lite conversion
- **Speed**: <100ms inference on mobile devices
- **Classes**: 7 universal disease categories
- **Input**: 224x224 RGB images

## 🗂️ Dataset Categories

### Universal Disease Mapping
```
Blight:              Early blight, Late blight, Black rot
Healthy:             Healthy plants across all crops
Leaf_Spot:          Bacterial spot, Scorch, Scab
Mosaic_Virus:       Mosaic viruses, Curl viruses
Nutrient_Deficiency: Yellowing, Leaf mold, Greening
Powdery_Mildew:     White powdery fungal infections
Rust:               Orange/brown rust diseases
```

## 📈 Training Strategy

### Data Splits
- **Training**: 70% (with 30% augmented)
- **Validation**: 15% (clean images only)
- **Testing**: 15% (clean images only)

### Augmentation Schedule
- **Dataset Prep**: 30% of images pre-augmented
- **Runtime**: Additional 50% chance during training
- **Severity**: 0.7 for dataset prep, 0.5 for runtime

### Model Architecture
- **Backbone**: MobileNetV3-Small (efficient) or Custom CNN
- **Head**: Dense layers with dropout and batch normalization
- **Optimization**: Adam with cosine decay learning rate
- **Regularization**: Class weighting, dropout, early stopping

## 🔍 Testing and Validation

### Comprehensive Testing
1. **Unit Tests**: Individual augmentation functions
2. **Integration Tests**: Complete pipeline validation
3. **Real-World Tests**: Internet image performance
4. **Field Tests**: Actual field photograph evaluation
5. **Performance Tests**: Speed and memory usage

### Quality Assurance
- **Visual Inspection**: Sample augmented images
- **Distribution Analysis**: Augmentation variety statistics
- **Performance Monitoring**: Training metrics tracking
- **Error Handling**: Comprehensive error recovery

## 🚀 Deployment Options

### Mobile Deployment
```python
# Convert trained model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Web Deployment
```javascript
// Load in TensorFlow.js
const model = await tf.loadLayersModel('/models/plant_disease_model.json');
```

### API Deployment
```python
# Flask/FastAPI endpoint
@app.post("/predict")
async def predict(image: UploadFile):
    # Preprocess and predict
    result = model.predict(preprocessed_image)
    return {"disease": classes[result.argmax()]}
```

## 🛠️ Advanced Usage

### Custom Augmentation
```python
from cyclegan_augmentor import FieldEffectsAugmentor

# Initialize with custom settings
augmentor = FieldEffectsAugmentor(
    severity=0.8,
    enable_caching=True,
    debug=True
)

# Transform single image
augmented = augmentor.transform(image)

# Transform batch
augmented_batch = augmentor.batch_transform(images)
```

### Custom Training
```python
from cyclegan_augmentor import create_cyclegan_generator

# Create data generator with augmentation
train_gen = create_cyclegan_generator(
    X_train, y_train,
    batch_size=32,
    augmentor=augmentor,
    augment_prob=0.7
)

# Train model
model.fit(train_gen, epochs=50, ...)
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Reduce batch size
python train_ultimate_cyclegan.py --batch-size 8

# Or reduce augmentation caching
# In cyclegan_augmentor.py: enable_caching=False
```

**2. Slow Training**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reduce image processing workers
# In prepare_ultimate_dataset_cyclegan.py: num_workers=2
```

**3. Low Augmentation Quality**
```python
# Increase severity for more dramatic effects
augmentor = FieldEffectsAugmentor(severity=0.9)

# Check intermediate results
augmentor = FieldEffectsAugmentor(debug=True)
```

**4. Dataset Not Found**
```bash
# Check dataset structure
ls -la PlantVillage/PlantVillage/
ls -la datasets/PlantDisease/

# Force rebuild dataset
python prepare_ultimate_dataset_cyclegan.py --force-rebuild
```

### Performance Optimization

**1. Speed Up Dataset Preparation**
```python
# Use more CPU cores
preparator = UltimateDatasetPreparator(num_workers=8)

# Reduce image quality for speed (not recommended for production)
# In _process_single_image: quality=80
```

**2. Speed Up Training**
```python
# Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Reduce validation frequency
# Add validation_freq=5 to model.fit()
```

## 📚 References and Citations

### Augmentation Techniques
- Perlin Noise Generation for Natural Textures
- Computer Vision Augmentation Strategies
- Domain Adaptation for Agricultural AI

### Model Architecture
- MobileNetV3 for Mobile Deployment
- EfficientNet Family for Accuracy/Speed Trade-offs
- Transfer Learning Best Practices

### Plant Disease Detection
- PlantVillage Dataset Analysis
- Universal Disease Classification Approaches
- Real-World Agricultural AI Deployment

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_cyclegan.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

### Adding New Augmentations
1. Add method to `FieldEffectsAugmentor` class
2. Update `transform()` method to include new augmentation
3. Add configuration parameters
4. Write unit tests
5. Update documentation

### Performance Improvements
1. Profile bottlenecks using `cProfile`
2. Optimize hot paths with NumPy vectorization
3. Add caching for expensive operations
4. Consider GPU acceleration for complex operations

## 📄 License

This CycleGAN augmentation system is part of the PlantPulse project and follows the same licensing terms. The augmentation techniques are based on established computer vision principles and can be used for research and commercial applications.

## 🔗 Related Projects

- **PlantPulse Main App**: React Native plant health monitoring
- **Thermal Model**: Thermal camera-based plant analysis  
- **Web Interface**: Browser-based plant disease detection
- **API Server**: RESTful API for plant health services

---

*For questions, issues, or contributions, please refer to the main PlantPulse project repository.*