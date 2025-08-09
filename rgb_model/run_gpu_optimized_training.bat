@echo off
echo ===============================================
echo GPU-OPTIMIZED TRAINING SETUP
echo ===============================================
echo.

echo Installing GPU monitoring packages...
pip install GPUtil psutil nvidia-ml-py3 --quiet

echo.
echo ===============================================
echo STARTING GPU-OPTIMIZED TRAINING
echo ===============================================
echo.
echo This will use:
echo   - RTX 3060 Ti GPU with mixed precision (FP16)
echo   - All Ryzen 7 CPU cores for data loading
echo   - Optimized batch size (64) for 8GB VRAM
echo   - XLA JIT compilation for faster execution
echo   - Memory-efficient data pipeline
echo.
echo Press any key to start training...
pause > nul

python train_gpu_optimized.py

echo.
echo Training completed!
pause