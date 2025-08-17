# PowerShell script for overnight GPU training with fixed metrics calculation
# This script runs the enhanced training with proper metrics calculation on clean data

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "OVERNIGHT GPU TRAINING - PLANT DISEASE DETECTION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU availability
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
Write-Host ""

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Write-Host "Working directory: $scriptPath" -ForegroundColor Green
Write-Host ""

# Training configuration
$epochs = 30  # Reduced from 50 - faster convergence with optimized LR
$batchSize = 32
$preprocessingMode = "default"  # Use advanced preprocessing for best results
$lossType = "combined"  # Focal + Label Smoothing
$learningRate = 0.005  # Optimized learning rate (5x faster!)
$swaStartEpoch = 20  # Start SWA after 20 epochs

Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  Epochs: $epochs"
Write-Host "  Batch Size: $batchSize"
Write-Host "  Preprocessing: $preprocessingMode (advanced)"
Write-Host "  Loss Function: $lossType"
Write-Host "  Learning Rate: $learningRate"
Write-Host "  SWA Start: Epoch $swaStartEpoch"
Write-Host ""

# Create output directory for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "models/run_$timestamp"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Write-Host "Output directory: $outputDir" -ForegroundColor Green
Write-Host ""

# Build the command
$pythonCmd = @(
    "python",
    "train_robust_model_v2.py",
    "--epochs", $epochs,
    "--batch_size", $batchSize,
    "--preprocessing_mode", $preprocessingMode,
    "--use_advanced_preprocessing",
    "--loss_type", $lossType,
    "--learning_rate", $learningRate,
    "--swa_start_epoch", $swaStartEpoch,
    "--mixup_alpha", "0.2",
    "--mixup_probability", "0.5",
    "--gradient_clip_norm", "1.0",
    "--focal_gamma", "2.0",
    "--label_smoothing_epsilon", "0.1",
    "--focal_weight", "0.7",
    "--output_dir", $outputDir
)

# Log file path
$logFile = "$outputDir/training_log.txt"

Write-Host "Starting training at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "Command: $($pythonCmd -join ' ')" -ForegroundColor Gray
Write-Host "Log file: $logFile" -ForegroundColor Gray
Write-Host ""
Write-Host "Training in progress... (This will take several hours)" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

# Start training with output to both console and log file
& $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length-1)] 2>&1 | Tee-Object -FilePath $logFile

# Check if training completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    
    # Parse the log for final metrics
    $logContent = Get-Content $logFile -Tail 100
    $finalAccuracy = $logContent | Select-String -Pattern "Final Test Accuracy: ([\d.]+%)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
    
    if ($finalAccuracy) {
        Write-Host "Final Test Accuracy: $finalAccuracy" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "Model saved in: $outputDir" -ForegroundColor Green
    Write-Host "Training log: $logFile" -ForegroundColor Green
    
    # List generated model files
    Write-Host ""
    Write-Host "Generated files:" -ForegroundColor Yellow
    Get-ChildItem $outputDir -Filter "*.h5" | ForEach-Object { Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" }
    Get-ChildItem $outputDir -Filter "*.keras" | ForEach-Object { Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" }
    Get-ChildItem $outputDir -Filter "*.tflite" | ForEach-Object { Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" }
    Get-ChildItem $outputDir -Filter "*.json" | ForEach-Object { Write-Host "  - $($_.Name)" }
    
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "TRAINING FAILED OR INTERRUPTED" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Check the log file for details: $logFile" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan

# Keep window open
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")