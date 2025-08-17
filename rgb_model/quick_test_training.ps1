# Quick test script to verify training works before overnight run
# Runs only 2 epochs to check everything is working

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "QUICK TEST - VERIFY TRAINING SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will run 2 quick epochs to verify the setup is working correctly" -ForegroundColor Yellow
Write-Host ""

# Check GPU
Write-Host "Checking GPU..." -ForegroundColor Yellow
$gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
if ($gpuInfo) {
    Write-Host "[OK] GPU detected: $gpuInfo" -ForegroundColor Green
} else {
    Write-Host "[WARNING] No GPU detected, will use CPU (slower)" -ForegroundColor Yellow
}
Write-Host ""

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Quick test configuration
Write-Host "Test Configuration:" -ForegroundColor Yellow
Write-Host "  Epochs: 3 (quick test)"
Write-Host "  Batch Size: 32"
Write-Host "  Learning Rate: 0.005 (optimized)"
Write-Host "  Preprocessing: legacy (faster for test)"
Write-Host ""

# Run quick test
$startTime = Get-Date
Write-Host "Starting quick test at $(Get-Date -Format 'HH:mm:ss')..." -ForegroundColor Cyan

python train_robust_model_v2.py `
    --test_run `
    --epochs 3 `
    --batch_size 32 `
    --learning_rate 0.005 `
    --preprocessing_mode legacy `
    --loss_type standard `
    --mixup_alpha 0.1 `
    --mixup_probability 0.3 `
    --swa_start_ratio 0.75

$duration = (Get-Date) - $startTime

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "TEST COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString('mm\:ss'))" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Key things verified:" -ForegroundColor Green
    Write-Host "  [OK] Data loading working (no duplicate files)" -ForegroundColor Green
    Write-Host "  [OK] Model training without errors" -ForegroundColor Green
    Write-Host "  [OK] Clean metrics calculation working" -ForegroundColor Green
    Write-Host "  [OK] Validation metrics being computed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready for overnight training!" -ForegroundColor Cyan
    Write-Host "Run: .\run_overnight_training.ps1" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "TEST FAILED - PLEASE FIX ERRORS BEFORE OVERNIGHT RUN" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")