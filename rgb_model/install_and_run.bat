@echo off
echo ========================================
echo RGB Model Setup and Training
echo ========================================
echo.

echo Step 1: Installing dependencies...
pip install numpy tensorflow pillow scikit-learn matplotlib tqdm opencv-python pandas --quiet

echo.
echo Step 2: Starting training with fixed model...
echo ----------------------------------------
python train_with_fixed_model.py --batch-size 8 --epochs 10 --subset-size 1000

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause