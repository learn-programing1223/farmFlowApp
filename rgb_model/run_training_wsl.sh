#!/bin/bash
# OVERNIGHT TRAINING SCRIPT FOR WSL WITH GPU
# Run this in WSL with your tf_env activated

echo "======================================"
echo "OVERNIGHT GPU TRAINING SCRIPT"
echo "Start Time: $(date)"
echo "======================================"

# Activate the virtual environment if not already active
if [[ "$VIRTUAL_ENV" != *"tf_env"* ]]; then
    echo "Activating tf_env..."
    source ~/tf_env/bin/activate
fi

# Navigate to project directory
cd ~/farmFlowApp/rgb_model

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Found: {len(gpus)} GPU(s) - {gpus}')"
echo ""

# Run the training with GPU
echo "Starting 50-epoch training with GPU..."
python train_robust_model_v2.py \
    --preprocessing_mode legacy \
    --loss_type combined \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --swa_start_epoch 30 \
    --gradient_clip_norm 1.0 \
    --mixup_alpha 0.2 \
    --mixup_probability 0.3

echo ""
echo "======================================"
echo "TRAINING COMPLETE!"
echo "End Time: $(date)"
echo "======================================"