#!/bin/bash

# PlantPulse Quick Setup Script
# This script automates the setup process for new installations

echo "======================================"
echo "   PlantPulse Setup Script v1.0      "
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "✓ Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."

# Check if running on Mac with Apple Silicon
if [[ $(uname -m) == 'arm64' ]] && [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected Apple Silicon Mac"
    pip install tensorflow-macos==2.13.0
    pip install tensorflow-metal
else
    pip install tensorflow==2.13.0
fi

# Install other requirements
pip install numpy==1.23.5  # Use compatible version
pip install pandas scikit-learn opencv-python pillow matplotlib seaborn tqdm

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/PlantVillage/raw/color
mkdir -p data/PlantDoc/train
mkdir -p data/PlantDoc/test
mkdir -p data/splits
mkdir -p data/cache
mkdir -p data/augmented
mkdir -p models/rgb_model/final

# Test installation
echo ""
echo "Testing installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"

echo ""
echo "======================================"
echo "   Setup Complete!                   "
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Download datasets:"
echo "   - PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
echo "   - Extract to: data/PlantVillage/"
echo ""
echo "2. Test the setup:"
echo "   python rgb_model/quick_start.py"
echo ""
echo "3. Train a model:"
echo "   python rgb_model/train_robust_model.py --batch-size 32"
echo ""
echo "Remember to activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""