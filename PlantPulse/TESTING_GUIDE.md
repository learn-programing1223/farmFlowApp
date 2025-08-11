# PlantPulse Testing Guide

## Quick Start Testing (Windows)

### 1. Setup Android Emulator
1. Open Android Studio
2. Click "AVD Manager" (or Tools → AVD Manager)
3. Create a new virtual device:
   - Choose "Pixel 6" or similar
   - Select system image (API 31+ recommended)
   - Click "Finish"
4. Start the emulator by clicking the play button

### 2. Install and Run App
```bash
# One-time setup
setup.bat

# Run the app
run-android.bat
```

### 3. Test Disease Detection
1. App will open with camera screen
2. Click any plant image from Google and save to your computer
3. In emulator, open Photos/Gallery app
4. Upload the saved image to emulator (drag and drop)
5. Return to PlantPulse app
6. Take a "photo" (it will use gallery)
7. See disease detection results!

## Alternative: Test with Expo (No Emulator Needed)

### Setup Expo
```bash
npm install -g expo-cli
npx expo start
```

### Test on Your Phone
1. Install "Expo Go" from App Store/Play Store
2. Scan the QR code shown in terminal
3. App opens on your phone!
4. Take real photos of plants

## Test Scenarios

### Scenario 1: Healthy Plant
- Point at any green, healthy plant
- Expected: "Healthy" result with care tips

### Scenario 2: Yellow Leaves
- Find a plant with yellowing leaves
- Expected: "Nutrient Deficiency" with fertilizer recommendations

### Scenario 3: Brown Spots
- Look for leaves with brown/black spots
- Expected: "Leaf Spot" or "Blight" with treatment options

### Scenario 4: White Powder
- Find plants with white powdery coating
- Expected: "Powdery Mildew" with fungicide suggestions

## Using Test Images

Download these test images to try the app:

1. **Tomato Blight**: Search "tomato early blight leaves"
2. **Powdery Mildew**: Search "powdery mildew cucumber"
3. **Rust Disease**: Search "rust disease bean leaves"
4. **Healthy Plant**: Search "healthy tomato plant leaves"
5. **Nutrient Deficiency**: Search "nitrogen deficiency plants yellow"

## Performance Metrics

Expected performance on different devices:

| Device | Inference Time | RAM Usage |
|--------|---------------|-----------|
| High-end (Pixel 7) | <50ms | ~150MB |
| Mid-range (Pixel 4a) | <100ms | ~200MB |
| Low-end | <200ms | ~250MB |

## Debugging Tips

### Check Model Loading
```javascript
// In RGBCameraScreen.tsx, add console logs:
console.log('Model loaded:', modelLoaded);
```

### Verify Image Processing
```javascript
// Check image dimensions
console.log('Image size:', width, 'x', height);
```

### Monitor Predictions
```javascript
// Log confidence scores
console.log('Predictions:', results.confidence);
```

## Common Issues

### "Model not found"
- Check `android/app/src/main/assets/plant_disease_model.tflite` exists
- Verify file is exactly 1.5MB

### "Camera permission denied"
- Go to Settings → Apps → PlantPulse → Permissions
- Enable Camera permission

### "App crashes on photo"
- Likely memory issue
- Try closing other apps
- Restart emulator with more RAM

### "Wrong predictions"
- Ensure good lighting
- Center the plant in frame
- Avoid blurry photos

## Test Checklist

- [ ] App launches without crashes
- [ ] Camera permission request appears
- [ ] Camera preview shows correctly
- [ ] Can capture photo
- [ ] Disease detection completes in <2 seconds
- [ ] Results show disease name and confidence
- [ ] Recommendations display properly
- [ ] Can scan another plant
- [ ] History saves correctly
- [ ] App works offline

## Automated Testing

Run unit tests:
```bash
npm test
```

Run E2E tests (if configured):
```bash
npm run test:e2e
```

## Beta Testing

1. **Android**: 
   - Build APK: `cd android && ./gradlew assembleRelease`
   - Share APK file from `android/app/build/outputs/apk/release/`

2. **iOS** (requires Apple Developer account):
   - Upload to TestFlight
   - Share invite link

## Feedback Collection

After testing, consider these questions:
1. How accurate were the disease predictions?
2. Was the app responsive and fast?
3. Were the treatment recommendations helpful?
4. Any crashes or errors?
5. Suggestions for improvement?

---

**Ready to test!** Start with the Android emulator for quickest results.