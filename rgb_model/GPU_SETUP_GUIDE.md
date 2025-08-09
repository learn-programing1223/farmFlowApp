# GPU Setup Guide for RTX 3060 Ti on Windows

## Current Status
Your training is running on CPU with all cores optimized. While functional, GPU would provide 2-3x speedup.

## Why GPU isn't Working
TensorFlow 2.20 (your version) requires specific CUDA/cuDNN versions that aren't automatically installed on Windows.

## Option 1: Use WSL2 (Recommended - Easiest)
WSL2 (Windows Subsystem for Linux) provides better GPU support:

```bash
# In Windows PowerShell (as Admin):
wsl --install

# Restart PC, then in WSL2:
sudo apt update
sudo apt install python3-pip python3-dev
pip3 install tensorflow[and-cuda]
```

## Option 2: Native Windows Setup (Complex)
1. **Install CUDA Toolkit 11.8** (specific version required)
   - Download: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Choose: Windows > x86_64 > 11 > exe (local)

2. **Install cuDNN 8.6** (requires NVIDIA account)
   - Download: https://developer.nvidia.com/cudnn
   - Extract and copy files to CUDA installation folder

3. **Set Environment Variables**:
   ```
   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   PATH += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   PATH += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
   ```

4. **Reinstall TensorFlow**:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.15.0  # Last version with Windows GPU support
   ```

## Option 3: Use PyTorch Instead (Alternative)
PyTorch has better Windows GPU support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Current Training Performance (CPU)
- **Using**: All 16 threads of Ryzen 7
- **Speed**: ~5-6 seconds per step
- **Total time**: ~5-10 hours with early stopping
- **Model will auto-save best version**

## Quick Test for GPU
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

## Notes
- Your current CPU training IS optimized and will complete successfully
- GPU would make it 2-3x faster but isn't essential
- The model quality will be the same either way
- Training is already saving the best model automatically