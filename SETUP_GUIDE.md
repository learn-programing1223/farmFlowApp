# 🚀 Complete Setup Guide for PlantPulse

This guide will help you set up the PlantPulse project from scratch on a new computer.

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: 20GB free space for datasets and models
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA for faster training)
- **Python**: 3.8 - 3.11 (3.12+ may have compatibility issues with TensorFlow)

### Software Requirements
- Git
- Python 3.8-3.11
- pip (Python package manager)
- (Optional) CUDA 11.2+ and cuDNN 8.1+ for GPU support

## 🔧 Step-by-Step Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/farmFlowApp.git
cd farmFlowApp
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow (CPU version)
pip install tensorflow==2.13.0

# OR for GPU support (if you have CUDA installed):
pip install tensorflow-gpu==2.13.0

# Install other requirements
pip install -r rgb_model/requirements.txt
```

If `requirements.txt` is missing or incomplete, install manually:
```bash
pip install numpy pandas scikit-learn opencv-python pillow matplotlib seaborn tqdm
```

### Step 4: Create Required Directories
```bash
# Create data directories
mkdir -p data/PlantVillage
mkdir -p data/PlantDoc
mkdir -p data/splits
mkdir -p data/cache
mkdir -p models/rgb_model
```

### Step 5: Download Datasets

#### Option A: Automated Download (if implemented)
```bash
python rgb_model/download_datasets.py
```

#### Option B: Manual Download
1. **PlantVillage Dataset**:
   - Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
   - Download and extract to: `data/PlantVillage/`
   - Structure should be: `data/PlantVillage/raw/color/[disease_folders]`

2. **PlantDoc Dataset** (Optional):
   - Go to: https://github.com/pratikkayal/PlantDoc-Dataset
   - Download and extract to: `data/PlantDoc/`
   - Structure should be: `data/PlantDoc/train/` and `data/PlantDoc/test/`

### Step 6: Verify Installation
```bash
# Test if everything is working
python rgb_model/quick_start.py
```

You should see:
```
✓ TensorFlow 2.x.x
✓ Model classes imported
✓ Model created successfully
✓ Quick start test completed successfully!
```

## 🏃‍♂️ Quick Start Options

### Option 1: Train from Scratch (Full Training)
```bash
# This will take 2-6 hours depending on your hardware
python rgb_model/train_robust_model.py \
    --data-dir ./data \
    --batch-size 32 \
    --stage1-epochs 15 \
    --stage2-epochs 20 \
    --stage3-epochs 10
```

### Option 2: Quick Test Training (Small Dataset)
```bash
# For testing setup (10 minutes)
python rgb_model/train_rgb_model.py \
    --plantvillage-subset 0.1 \
    --samples-per-class 100 \
    --batch-size 16 \
    --stage1-epochs 2 \
    --stage2-epochs 2 \
    --stage3-epochs 1
```

### Option 3: Use Pre-trained Weights (If Available)
```bash
# Download pre-trained weights (if provided)
# Place in models/rgb_model/final/

# Test inference
python rgb_model/test_model.py --weights models/rgb_model/final/model.weights.h5
```

## 💻 Platform-Specific Instructions

### macOS (Apple Silicon M1/M2)
```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal  # For GPU acceleration

# You may need to install additional dependencies
brew install libomp
```

### Windows
```bash
# If you encounter errors with numpy/scipy
pip install numpy==1.23.5
pip install scipy==1.10.1

# For long path support (if needed)
git config --system core.longpaths true
```

### Linux/Ubuntu
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libhdf5-dev libatlas-base-dev

# For GPU support
sudo apt-get install nvidia-cuda-toolkit
```

## 🔍 Troubleshooting

### Issue 1: "No module named 'tensorflow'"
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Issue 2: "Out of Memory" during training
```bash
# Reduce batch size
python rgb_model/train_robust_model.py --batch-size 16
# or even
python rgb_model/train_robust_model.py --batch-size 8
```

### Issue 3: "No such file or directory: data/PlantVillage"
```bash
# Make sure you downloaded the datasets
# Create directories and download datasets as shown in Step 4-5
```

### Issue 4: GPU not detected
```bash
# Check CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Should show your GPU. If empty, reinstall tensorflow-gpu and CUDA
```

### Issue 5: Disk space issues
```bash
# Use compressed data format
python rgb_model/train_robust_model_no_cache.py --batch-size 16

# Or clean up cache
rm -rf data/cache
rm -rf data/splits/*.npy
```

## 📊 Expected Training Results

After successful training, you should see:
- **Stage 1**: ~60-70% validation accuracy
- **Stage 2**: ~75-80% validation accuracy  
- **Stage 3**: ~80-85% validation accuracy
- **Final model size**: ~20-50MB
- **TFLite model**: ~5-15MB (INT8 quantized)

## 🎯 Next Steps

1. **Convert to TFLite for mobile**:
```bash
python rgb_model/convert_to_tflite.py \
    --model-path models/rgb_model/final/model.keras \
    --quantization int8
```

2. **Analyze results**:
```bash
python rgb_model/analyze_results.py
```

3. **Test on your own images**:
```python
from rgb_model.src.model_robust import create_robust_model
import cv2
import numpy as np

# Load model
model = create_robust_model(num_classes=7)
model.load_weights('models/rgb_model/final/model.weights.h5')

# Load and preprocess your image
img = cv2.imread('your_plant_image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.get_model()(img)
classes = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
           'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']
predicted_class = classes[np.argmax(prediction)]
print(f"Predicted: {predicted_class} ({np.max(prediction)*100:.1f}% confidence)")
```

## 💡 Tips for Success

1. **Start small**: Test with subset of data first
2. **Monitor resources**: Use `htop` (Linux/Mac) or Task Manager (Windows)
3. **Save progress**: Models checkpoint automatically
4. **Use GPU**: Training is 10-50x faster with GPU
5. **Be patient**: Full training takes 2-6 hours

## 📧 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Look at existing GitHub issues
3. Create a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

Good luck with your plant disease detection journey! 🌱🤖