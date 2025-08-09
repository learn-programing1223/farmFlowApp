@echo off
echo ========================================
echo Installing RGB Model Dependencies
echo ========================================
echo.

echo Installing core packages...
pip install numpy tensorflow pillow scikit-learn matplotlib tqdm

echo.
echo Installing additional packages...
pip install opencv-python pandas

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo You can now run:
echo   python train_with_fixed_model.py --batch-size 8 --epochs 10
echo.
pause