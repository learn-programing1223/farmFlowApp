# PlantPulse Setup Instructions

## Prerequisites

1. **macOS** with Xcode 14+ installed
2. **Node.js** 18+ and npm
3. **CocoaPods** (install with `sudo gem install cocoapods`)
4. **React Native CLI** (`npm install -g react-native-cli`)
5. **iPhone 15+** with USB-C port (for thermal camera support)

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to project directory
cd PlantPulse

# Install npm packages
npm install

# Install iOS pods
cd ios
pod install
cd ..
```

### 2. Configure Xcode Project

1. Open `ios/PlantPulse.xcworkspace` in Xcode
2. Select your development team in Signing & Capabilities
3. Ensure the Swift bridging header is configured:
   - Build Settings → Swift Compiler → Objective-C Bridging Header
   - Set to: `PlantPulse/PlantPulse-Bridging-Header.h`

### 3. Add TensorFlow Lite Model

The app needs the trained ML model file. For now, it will use placeholder data for testing.

To add a real model later:
1. Train the model using the provided datasets
2. Convert to TFLite format with INT8 quantization
3. Place `plant_health_v1.tflite` in `src/ml/models/`

### 4. Run the App

```bash
# Start Metro bundler
npm start

# In another terminal, run on iOS
npm run ios
```

Or run from Xcode:
1. Select your iPhone device
2. Click the Run button

## Testing Without Thermal Camera

The app includes mock data generation for testing without a physical thermal camera:
- Camera connection simulation works automatically
- Thermal data is generated with realistic temperature ranges
- ML inference uses placeholder results

## Connecting Thermal Cameras

### Supported Cameras
- **InfiRay P2 Pro** ($299) - Recommended
- **TOPDON TC002C** ($270) - Budget option

### Connection Steps
1. Connect the thermal camera to your iPhone's USB-C port
2. Launch PlantPulse
3. Tap "Connect Camera" on the Camera screen
4. The app will detect and connect automatically

### Troubleshooting

**Camera not detected:**
- Ensure the camera is properly connected
- Check that the camera is powered on
- Try disconnecting and reconnecting

**No thermal data:**
- Verify the camera model is supported
- Check iOS Settings → PlantPulse → Camera permission is enabled
- Restart the app

**Performance issues:**
- Reduce frame processing rate in Settings
- Disable GPU acceleration if experiencing crashes
- Close other apps to free memory

## Development Mode Features

- **Mock thermal data**: Realistic temperature patterns for testing
- **Placeholder ML inference**: Simulated plant health analysis
- **Debug overlay**: Shows FPS and processing metrics
- **Temperature simulation**: Generates water stress and disease patterns

## Next Steps

1. **Get thermal camera SDK**: Contact InfiRay/TOPDON for iOS SDK access
2. **Train ML model**: Use provided datasets and training scripts
3. **Calibrate for your plants**: Build species-specific profiles
4. **Field testing**: Validate with real plants showing known conditions

## Architecture Overview

```
PlantPulse/
├── src/
│   ├── camera/          # Thermal camera integration
│   ├── ml/              # TensorFlow Lite models
│   ├── analysis/        # Plant health algorithms
│   ├── ui/              # React Native components
│   └── types/           # TypeScript definitions
├── ios/
│   └── PlantPulse/      # Native iOS modules
└── android/             # Android support (future)
```

## Contributing

See CONTRIBUTING.md for development guidelines and code style.

## License

MIT License - See LICENSE file for details