#!/bin/bash

# Setup script for thermal dataset training
echo "=========================================="
echo "PLANTPULSE THERMAL TRAINING SETUP"
echo "=========================================="

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/eth_thermal
mkdir -p data/date_palm_thermal
mkdir -p data/test_thermal
mkdir -p logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install tensorflow numpy opencv-python matplotlib pillow tqdm

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. DOWNLOAD ETH ZURICH DATASET (Recommended):"
echo "   python download_eth_dataset.py"
echo ""
echo "   Or manually download from:"
echo "   http://robotics.ethz.ch/~asl-datasets/2018_plant_stress_phenotyping_dataset/images.zip"
echo ""
echo "2. CREATE TEST DATASET (Quick start):"
echo "   python thermal_data_loader.py"
echo ""
echo "3. TRAIN WITH REAL THERMAL DATA:"
echo "   python train_with_real_thermal.py"
echo ""
echo "4. MONITOR TRAINING:"
echo "   tensorboard --logdir logs"
echo ""
echo "=========================================="