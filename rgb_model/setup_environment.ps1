# PowerShell script to set up a clean Python environment for training
# This fixes compatibility issues with Python 3.13 and TensorFlow

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "SETTING UP CLEAN PYTHON ENVIRONMENT FOR PLANT DISEASE DETECTION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.(\d+)") {
    $minorVersion = [int]$matches[1]
    Write-Host "Current Python version: $pythonVersion" -ForegroundColor Yellow
    
    if ($minorVersion -eq 13) {
        Write-Host "[WARNING] Python 3.13 detected - TensorFlow may not be fully compatible!" -ForegroundColor Red
        Write-Host "Recommended: Use Python 3.9, 3.10, or 3.11" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Options:" -ForegroundColor Cyan
        Write-Host "1. Install Python 3.11 from: https://www.python.org/downloads/release/python-3118/" -ForegroundColor White
        Write-Host "2. Use Anaconda/Miniconda to manage Python versions" -ForegroundColor White
        Write-Host "3. Continue with current version (may have issues)" -ForegroundColor White
        Write-Host ""
        
        $continue = Read-Host "Continue with Python 3.13? (y/n)"
        if ($continue -ne 'y') {
            exit
        }
    } elseif ($minorVersion -ge 9 -and $minorVersion -le 11) {
        Write-Host "[OK] Python version is compatible" -ForegroundColor Green
    }
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
$venvPath = ".\venv_plantdisease"

# Remove old venv if exists
if (Test-Path $venvPath) {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

# Create new venv
python -m venv $venvPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Virtual environment created" -ForegroundColor Green

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install compatible packages
Write-Host ""
Write-Host "Installing compatible packages..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray

# First, install critical dependencies with specific versions
pip install numpy==1.24.3
pip install protobuf==4.25.1

# Then install TensorFlow
pip install tensorflow==2.15.0

# Install remaining packages
pip install -r requirements_fixed.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "ENVIRONMENT SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To activate this environment in the future, run:" -ForegroundColor Yellow
    Write-Host "  .\venv_plantdisease\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "To start training, run:" -ForegroundColor Yellow
    Write-Host "  .\run_clean_training.ps1" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "[ERROR] Package installation failed" -ForegroundColor Red
    Write-Host "Try manual installation:" -ForegroundColor Yellow
    Write-Host "  1. Use Python 3.10 or 3.11" -ForegroundColor White
    Write-Host "  2. Create fresh virtual environment" -ForegroundColor White
    Write-Host "  3. Install packages one by one from requirements_fixed.txt" -ForegroundColor White
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")