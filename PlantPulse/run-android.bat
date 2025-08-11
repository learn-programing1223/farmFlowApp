@echo off
echo ========================================
echo PlantPulse - Running on Android
echo ========================================
echo.

echo Starting Metro bundler...
start cmd /k "npx react-native start"

echo.
echo Waiting for Metro to start...
timeout /t 5 /nobreak > nul

echo.
echo Building and installing on Android device/emulator...
echo Make sure your Android emulator is running or device is connected!
echo.

npx react-native run-android

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Build failed!
    echo ========================================
    echo.
    echo Troubleshooting:
    echo 1. Is Android emulator running?
    echo 2. Is USB debugging enabled on your device?
    echo 3. Try: cd android ^&^& gradlew clean
    echo.
) else (
    echo.
    echo ========================================
    echo App launched successfully!
    echo ========================================
    echo.
    echo The PlantPulse app should now be running on your device.
    echo Take a photo of any plant to detect diseases!
    echo.
)

pause