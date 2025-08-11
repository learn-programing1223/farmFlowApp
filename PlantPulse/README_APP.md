# PlantPulse - AI Plant Disease Detection App

## Overview
PlantPulse is a React Native mobile application that uses machine learning to detect plant diseases from photos. The app features a custom-trained TensorFlow Lite model that achieves 95% accuracy in identifying 7 common plant disease categories.

## Features
- **Real-time Disease Detection**: Take a photo and get instant disease diagnosis
- **95% Accuracy**: Custom CNN model trained on 14,000+ plant images
- **Offline Capability**: All processing happens on-device, no internet required
- **Detailed Recommendations**: Get specific treatment plans for each disease
- **Plant History Tracking**: Monitor your plants' health over time
- **Dual Camera Modes**: RGB camera for disease detection, thermal camera support (when available)

## Disease Categories
1. **Blight** - Early and late blight detection
2. **Leaf Spot** - Bacterial and fungal spots
3. **Mosaic Virus** - Viral infections
4. **Nutrient Deficiency** - Nitrogen, phosphorus, potassium deficiencies
5. **Powdery Mildew** - White fungal growth
6. **Rust** - Orange/brown pustules
7. **Healthy** - No disease detected

## Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- React Native development environment
- For iOS: Mac with Xcode 14+
- For Android: Android Studio with SDK 31+

### Installation

1. **Install Dependencies**
```bash
cd PlantPulse
npm install
```

2. **iOS Setup** (Mac only)
```bash
cd ios
pod install
cd ..
```

3. **Android Setup**
No additional setup required - the TFLite model is already in `android/app/src/main/assets/`

## Running the App

### Android Development

#### Option 1: Android Emulator (Recommended for Windows)
```bash
# Start Android emulator from Android Studio first
npx react-native run-android
```

#### Option 2: Physical Android Device
1. Enable Developer Mode on your phone
2. Enable USB Debugging
3. Connect phone via USB
4. Run:
```bash
npx react-native run-android
```

### iOS Development (Mac only)

#### Option 1: iOS Simulator
```bash
npx react-native run-ios
```

#### Option 2: Physical iPhone
1. Open `ios/PlantPulse.xcworkspace` in Xcode
2. Select your device from the device list
3. Click Run button or press Cmd+R

### Using Expo (Easiest for Testing)

If you want to test on iPhone without a Mac:

1. **Install Expo CLI**
```bash
npm install -g expo-cli
```

2. **Convert to Expo** (optional)
```bash
npx install-expo-modules
```

3. **Run with Expo**
```bash
npx expo start
```

4. **Test on Phone**
- Download Expo Go app from App Store/Play Store
- Scan the QR code shown in terminal

## Testing the Disease Detection

### Quick Test
1. Launch the app
2. Tap the camera icon
3. Point at any plant (healthy or diseased)
4. Take a photo
5. View results and recommendations

### Test Images
You can test with these common plant issues:
- Yellow leaves → Nutrient Deficiency
- Brown spots → Leaf Spot or Blight
- White powder → Powdery Mildew
- Orange pustules → Rust

## Model Information
- **Architecture**: Custom CNN with 1.44M parameters
- **Input Size**: 224x224 RGB images
- **Model Size**: 1.5MB (TFLite optimized)
- **Accuracy**: 95.10% on test set
- **Inference Speed**: <100ms on modern phones

## Project Structure
```
PlantPulse/
├── src/
│   ├── ml/                    # Machine learning models
│   │   └── RGBDiseaseModel.ts # TFLite model wrapper
│   ├── ui/
│   │   └── screens/
│   │       ├── RGBCameraScreen.tsx    # Disease detection camera
│   │       └── UnifiedCameraScreen.tsx # Camera mode selector
│   ├── data/
│   │   └── plantCareDatabase.ts # Disease treatment info
│   ├── services/
│   │   └── plantHistoryService.ts # Scan history tracking
│   └── utils/
│       └── imageProcessing.ts # Image preprocessing
├── android/
│   └── app/src/main/assets/
│       └── plant_disease_model.tflite # Android model
├── ios/
│   └── PlantPulse/Resources/
│       └── plant_disease_model.tflite # iOS model
└── package.json
```

## Troubleshooting

### Android Issues
```bash
# Clean and rebuild
cd android
./gradlew clean
cd ..
npx react-native run-android
```

### iOS Issues
```bash
# Clean build folder
cd ios
xcodebuild clean
pod install --repo-update
cd ..
npx react-native run-ios
```

### Metro Issues
```bash
# Reset Metro cache
npx react-native start --reset-cache
```

### Model Not Loading
- Verify TFLite file exists in assets folders
- Check console for loading errors
- Ensure react-native-fast-tflite is properly linked

## Building for Production

### Android APK
```bash
cd android
./gradlew assembleRelease
# APK will be in android/app/build/outputs/apk/release/
```

### iOS IPA
1. Open Xcode
2. Select "Any iOS Device" as target
3. Product → Archive
4. Distribute App → App Store Connect

## Performance Tips
- The app works best with good lighting
- Center the affected plant part in frame
- Hold camera steady for best results
- Avoid blurry or dark photos

## Future Features
- [ ] Thermal camera integration for stress detection
- [ ] Cloud backup for scan history
- [ ] Plant identification (species detection)
- [ ] Growth tracking with time-lapse
- [ ] Community features for sharing diagnoses

## License
MIT

## Support
For issues or questions, please open an issue on GitHub.

---

**Note**: This app is for educational purposes. For critical plant health decisions, consult with agricultural experts.