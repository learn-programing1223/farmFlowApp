# OVERNIGHT TRAINING SCRIPT FOR POWERSHELL
# Run at 2 AM with Task Scheduler or manually

Write-Host "======================================" -ForegroundColor Green
Write-Host "OVERNIGHT TRAINING SCRIPT" -ForegroundColor Green
Write-Host "Start Time: $(Get-Date)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Green

# Navigate to project directory
cd "C:\Users\aayan\OneDrive\Documents\GitHub\farmFlowApp\rgb_model"

# Option 1: RECOMMENDED - Use train_robust_model_v2.py (with all fixes)
Write-Host "`nStarting 50-epoch training with enhanced model..." -ForegroundColor Cyan
python train_robust_model_v2.py `
    --preprocessing_mode legacy `
    --loss_type combined `
    --epochs 50 `
    --batch_size 32 `
    --learning_rate 0.0001 `
    --swa_start_epoch 30 `
    --gradient_clip_norm 1.0 `
    --mixup_alpha 0.2 `
    --mixup_probability 0.3

# Option 2: FALLBACK - If you encounter issues, use the simpler script
# Uncomment the lines below if Option 1 fails
# Write-Host "`nRunning simpler training script..." -ForegroundColor Yellow
# python train_overnight_no_tflite.py --epochs 50 --batch_size 32 --learning_rate 0.0001

Write-Host "`n======================================" -ForegroundColor Green
Write-Host "TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "End Time: $(Get-Date)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Green

# Play a sound when done (optional)
[console]::beep(1000,500)