@echo off
echo ========================================
echo PlantPulse Setup Script
echo ========================================
echo.

echo Installing dependencies...
call npm install

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the app:
echo   - Android: npm run android
echo   - iOS (Mac only): npm run ios
echo   - Start Metro: npm start
echo.
echo For Android testing on Windows:
echo   1. Open Android Studio
echo   2. Start an Android emulator
echo   3. Run: npm run android
echo.
pause