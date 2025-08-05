#!/bin/bash

# Activate virtual environment and install RGB model requirements

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing RGB model requirements..."
pip install -r ../rgb_model/requirements.txt

echo "Installation complete!"