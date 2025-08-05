@echo off
REM PlantPulse Quick Setup Script for Windows
REM This script automates the setup process for new installations

echo ======================================
echo    PlantPulse Setup Script v1.0      
echo ======================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8-3.11 from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take 5-10 minutes...

REM Install TensorFlow
pip install tensorflow==2.13.0

REM Install other requirements
pip install numpy==1.23.5
pip install pandas scikit-learn opencv-python pillow matplotlib seaborn tqdm

REM Create necessary directories
echo.
echo Creating data directories...
if not exist "data\PlantVillage\raw\color" mkdir "data\PlantVillage\raw\color"
if not exist "data\PlantDoc\train" mkdir "data\PlantDoc\train"
if not exist "data\PlantDoc\test" mkdir "data\PlantDoc\test"
if not exist "data\splits" mkdir "data\splits"
if not exist "data\cache" mkdir "data\cache"
if not exist "data\augmented" mkdir "data\augmented"
if not exist "models\rgb_model\final" mkdir "models\rgb_model\final"

REM Test installation
echo.
echo Testing installation...
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"

echo.
echo ======================================
echo    Setup Complete!                   
echo ======================================
echo.
echo Next steps:
echo 1. Download datasets:
echo    - PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
echo    - Extract to: data\PlantVillage\
echo.
echo 2. Test the setup:
echo    python rgb_model\quick_start.py
echo.
echo 3. Train a model:
echo    python rgb_model\train_robust_model.py --batch-size 32
echo.
echo Remember to activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
pause