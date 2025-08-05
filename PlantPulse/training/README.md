# PlantPulse Model Training Guide

This directory contains the training pipeline for the PlantPulse thermal plant health detection model.

## Overview

The model is a multi-task neural network based on MobileNetV3-Small that analyzes thermal images to detect:

1. **Water Stress** - CWSI (Crop Water Stress Index) prediction
2. **Disease Classification** - Bacterial, Fungal, or Viral infections
3. **Nutrient Analysis** - N, P, K deficiency detection
4. **Affected Area Segmentation** - Pixel-wise problem area detection

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset Options

### Option 1: IEEE DataPort Hydroponic Dataset (Recommended)

1. Visit [IEEE DataPort Lettuce Dataset](https://ieee-dataport.org/open-access/lettuce-dataset)
2. Create a free account and download the dataset
3. Extract to `./hydroponic_dataset/`

This dataset contains:
- 30 hydroponic lettuce plants
- 66 days of monitoring
- Thermal + RGB images
- Environmental sensor data

### Option 2: Synthetic Data (For Testing)

The training script can generate synthetic thermal patterns that simulate:
- Healthy plants (cooler than ambient)
- Water stress (temperature increase)
- Disease patterns (circular, linear, mosaic)
- Nutrient deficiencies

## Training the Model

### Quick Start (Synthetic Data)

```bash
# Train with synthetic data
python train_plant_health_model.py
```

### With Real Dataset

```bash
# Prepare dataset
python dataset_loader.py --download

# Train with real data
python train_with_real_data.py --dataset ./hydroponic_dataset
```

### Training Configuration

Edit parameters in `train_plant_health_model.py`:

```python
IMG_SIZE = 224          # Input image size
BATCH_SIZE = 32         # Batch size (reduce if OOM)
EPOCHS = 100           # Training epochs
LEARNING_RATE = 0.001  # Initial learning rate
```

## Model Architecture

```
Input (224x224x1 thermal image)
    ↓
Conv2D (1→3 channels)
    ↓
MobileNetV3-Small (backbone)
    ↓
    ├── Water Stress Head → Sigmoid (0-1)
    ├── Disease Head → Softmax (4 classes)
    ├── Nutrient Head → Sigmoid (3 values)
    └── Segmentation Head → U-Net decoder
```

## Output Files

After training:

1. **plant_health_model.h5** - Full Keras model
2. **plant_health_v1.tflite** - Quantized TFLite model for mobile
3. **training_history.json** - Training metrics
4. **plant_health_best.h5** - Best checkpoint

## Model Performance Targets

- Water stress detection: >90% accuracy (±0.1 CWSI)
- Disease classification: >85% accuracy
- Nutrient analysis: >80% accuracy
- Model size: <50MB (INT8 quantized)
- Inference time: <100ms on mobile GPU

## Data Augmentation

The pipeline includes thermal-specific augmentations:

1. **Thermal noise** - Sensor noise simulation
2. **Atmospheric effects** - Humidity absorption
3. **Dead pixels** - Sensor defects
4. **Vignetting** - Lens effects
5. **Emissivity variation** - Surface property changes

## Custom Dataset Format

To use your own thermal dataset:

### Directory Structure
```
custom_dataset/
├── thermal/           # Thermal images
│   ├── plant1_day1_001.npy  # Temperature arrays
│   └── plant1_day1_001.png  # Or 16-bit images
├── labels.json       # Ground truth labels
└── metadata.csv      # Optional metadata
```

### Labels Format (labels.json)
```json
{
  "plant1_day1": {
    "water_stress": 0.0,
    "disease": "healthy",
    "nutrients": {
      "nitrogen": 0.5,
      "phosphorus": 0.5,
      "potassium": 0.5
    },
    "last_watered_hours": 12
  }
}
```

## Deployment to PlantPulse App

1. Copy `plant_health_v1.tflite` to:
   ```
   PlantPulse/src/ml/models/plant_health_v1.tflite
   ```

2. The app will automatically load the model on startup

## Monitoring Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir ./logs
```

## Tips for Better Results

1. **More Data**: The model improves significantly with more thermal images
2. **Balanced Classes**: Ensure equal representation of all conditions
3. **Temperature Calibration**: Accurate temperature values are crucial
4. **Multiple Plants**: Train on diverse plant species for generalization
5. **Time Series**: Include temporal progression of conditions

## Troubleshooting

**Out of Memory**: Reduce BATCH_SIZE or IMG_SIZE

**Poor Accuracy**: 
- Check temperature calibration
- Increase training data
- Adjust augmentation parameters

**Slow Training**: Enable GPU with `tf.config.list_physical_devices('GPU')`

## Citation

If using the IEEE DataPort dataset:
```
@dataset{hydroponic2023,
  title={Hydroponic Lettuce Growth Dataset with Thermal Imaging},
  author={IEEE DataPort Contributors},
  year={2023},
  publisher={IEEE DataPort},
  doi={10.21227/xxxxx}
}
```