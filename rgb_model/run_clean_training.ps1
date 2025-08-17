# Clean training launcher - No warnings, clean output
# This script runs the optimized training with all warnings suppressed

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "CLEAN GPU TRAINING - PLANT DISEASE DETECTION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists and is activated
$venvPath = ".\venv_plantdisease"
if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Host "Using virtual environment: $venvPath" -ForegroundColor Green
    $pythonExe = "$venvPath\Scripts\python.exe"
} else {
    Write-Host "No virtual environment found, using system Python" -ForegroundColor Yellow
    Write-Host "Run .\setup_environment.ps1 first for best results" -ForegroundColor Yellow
    $pythonExe = "python"
}

# Check GPU availability
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
$gpuCheck = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($gpuCheck) {
    Write-Host "[OK] GPU detected: $gpuCheck" -ForegroundColor Green
} else {
    Write-Host "[WARNING] No GPU detected - training will be slower" -ForegroundColor Yellow
}

# Training configuration
$epochs = 30
$batchSize = 32
$preprocessingMode = "default"
$lossType = "combined"
$learningRate = 0.005
$swaStartRatio = 0.75
$mixupAlpha = 0.1
$mixupProbability = 0.3

Write-Host ""
Write-Host "Training Configuration:" -ForegroundColor Yellow
Write-Host "  Epochs: $epochs"
Write-Host "  Batch Size: $batchSize"
Write-Host "  Learning Rate: $learningRate (optimized)"
Write-Host "  Preprocessing: $preprocessingMode"
Write-Host "  Loss: $lossType"
Write-Host "  SWA: Starts at 75% (epoch $([int]($epochs * $swaStartRatio)))"
Write-Host "  MixUp: alpha=$mixupAlpha, prob=$mixupProbability"
Write-Host ""

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "models/clean_run_$timestamp"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Write-Host "Output directory: $outputDir" -ForegroundColor Green

# Build command using the clean wrapper
$pythonCmd = @(
    $pythonExe,
    "train_clean.py",  # Use the clean wrapper
    "--epochs", $epochs,
    "--batch_size", $batchSize,
    "--preprocessing_mode", $preprocessingMode,
    "--use_advanced_preprocessing",
    "--loss_type", $lossType,
    "--learning_rate", $learningRate,
    "--swa_start_ratio", $swaStartRatio,
    "--mixup_alpha", $mixupAlpha,
    "--mixup_probability", $mixupProbability,
    "--gradient_clip_norm", "1.0",
    "--focal_gamma", "2.0",
    "--label_smoothing_epsilon", "0.1",
    "--focal_weight", "0.7",
    "--output_dir", $outputDir
)

# Log file
$logFile = "$outputDir\training_log.txt"

Write-Host ""
Write-Host "Starting clean training at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "Log file: $logFile" -ForegroundColor Gray
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Start training with clean output
$startTime = Get-Date

# Run training and capture output
& $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length-1)] 2>&1 | Tee-Object -FilePath $logFile

$duration = (Get-Date) - $startTime

# Check if training completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    Write-Host "Model saved in: $outputDir" -ForegroundColor Green
    
    # List generated files
    Write-Host ""
    Write-Host "Generated files:" -ForegroundColor Yellow
    Get-ChildItem $outputDir -Filter "*.h5" | ForEach-Object { 
        Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" -ForegroundColor White
    }
    Get-ChildItem $outputDir -Filter "*.keras" | ForEach-Object { 
        Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" -ForegroundColor White
    }
    Get-ChildItem $outputDir -Filter "*.tflite" | ForEach-Object { 
        Write-Host "  - $($_.Name) ($('{0:N2}' -f ($_.Length / 1MB)) MB)" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "TRAINING FAILED OR INTERRUPTED" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Check log file: $logFile" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")