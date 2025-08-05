# PlantPulse Thermal Model Training Guide

## Overview
We've created a complete pipeline for training plant health models with REAL thermal datasets instead of synthetic data. This will significantly improve model accuracy and real-world performance.

## Key Components Created

### 1. Dataset Download Script (`download_eth_dataset.py`)
- Downloads ETH Zurich thermal dataset (best for stress detection)
- ~1000+ thermal images with labeled conditions:
  - Optimal
  - Drought stress
  - Nitrogen deficiency (low/surplus)
  - Weed pressure
- Alternative dataset information included

### 2. Thermal Data Loader (`thermal_data_loader.py`)
- Loads real thermal images (PNG, JPG, TIFF formats)
- Converts pixel values to temperature estimates
- Creates multi-task labels from dataset structure
- Generates segmentation masks from thermal patterns
- Supports multiple dataset formats:
  - ETH Zurich
  - Date Palm (pest damage)
  - Generic thermal datasets

### 3. Real Thermal Training Script (`train_with_real_thermal.py`)
- Optimized model architecture for thermal images
- Stronger regularization to prevent overfitting
- Multi-task learning (water stress, disease, nutrients, segmentation)
- TensorFlow Lite conversion with INT8 quantization
- Comprehensive training visualization

## Quick Start

### Step 1: Setup Environment
```bash
cd /Users/aayan/Documents/GitHub/farmFlowApp/PlantPulse/training
./setup_thermal_training.sh
```

### Step 2: Get Thermal Data

#### Option A: Download ETH Zurich Dataset (Recommended)
```bash
source venv/bin/activate
python download_eth_dataset.py
```
This downloads ~2.5GB of real thermal images with stress labels.

#### Option B: Create Test Dataset
```bash
source venv/bin/activate
python thermal_data_loader.py
```
This creates a small test dataset for validation.

#### Option C: Use Your Own Thermal Images
Place thermal images in folders named by condition:
```
data/custom_thermal/
├── healthy/
├── drought/
├── diseased/
└── nutrient_deficient/
```

### Step 3: Train Model
```bash
source venv/bin/activate
python train_with_real_thermal.py
```

The script will:
1. Show available datasets
2. Let you select which to use
3. Train with early stopping
4. Save best model checkpoints
5. Convert to TensorFlow Lite

### Step 4: Monitor Training
```bash
tensorboard --logdir logs
```
Open http://localhost:6006 to view training progress.

## Expected Results

With real thermal data, expect:
- **Disease classification**: 75-85% accuracy (vs 70% synthetic)
- **Water stress detection**: <0.15 MAE (vs 0.28 synthetic)
- **Segmentation**: >95% accuracy
- **Model size**: <1MB with INT8 quantization

## Dataset Recommendations

1. **ETH Zurich** (Best overall)
   - Professional thermal imaging
   - Multiple stress conditions
   - Ground truth labels
   - Sugar beet crops

2. **Date Palm** (Pest detection)
   - 800+ images
   - 4 damage levels
   - Good for disease/pest focus

3. **NASA ECOSTRESS** (Large scale)
   - Satellite thermal data
   - 38m resolution
   - Global coverage
   - Requires preprocessing

4. **Your Own Data**
   - Use InfiRay P2 Pro or TOPDON TC002C
   - Capture at consistent times
   - Label by condition
   - Include metadata (temperature, humidity)

## Model Deployment

After training, you'll have:
- `thermal_model_[dataset]_[timestamp].tflite` - Optimized for mobile
- `thermal_model_[dataset]_[timestamp]_best.h5` - Best validation checkpoint
- `thermal_model_[dataset]_[timestamp]_history.json` - Training metrics

Copy the `.tflite` file to your React Native app:
```bash
cp thermal_model_*.tflite ../src/ml/models/plant_health_v2.tflite
```

## Troubleshooting

### Download Issues
- ETH dataset is large (2.5GB)
- Use wget/curl for resume support:
  ```bash
  wget -c http://robotics.ethz.ch/~asl-datasets/2018_plant_stress_phenotyping_dataset/images.zip
  ```

### Memory Issues
- Reduce batch size in training script
- Use data generator instead of loading all at once
- Consider Google Colab for free GPU

### Poor Results
- Ensure thermal images are properly normalized
- Check label distribution (balanced classes?)
- Increase training data
- Adjust temperature range for your climate

## Next Steps

1. **Collect More Data**: The more diverse thermal data, the better
2. **Fine-tune for Your Crops**: Each plant species has different thermal signatures
3. **Integrate with App**: Update the React Native app to use new model
4. **Field Testing**: Validate with real plants in your environment

## Resources

- ETH Dataset Paper: [Link to publication]
- Thermal Imaging for Plants: [Research papers]
- TensorFlow Lite Guide: https://www.tensorflow.org/lite
- Our GitHub: [Your repository]

Happy training! 🌱🔥📷