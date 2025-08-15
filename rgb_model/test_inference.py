#!/usr/bin/env python3
"""
Test script for real-world inference
"""

import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image

# Create a test image if needed
def create_test_image():
    """Create a test image for inference testing."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    test_image_path = test_dir / "test_plant.jpg"
    if not test_image_path.exists():
        # Create a green-ish image (simulating a leaf)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Green channel dominant (healthy leaf simulation)
        img_array[:, :, 1] = 180  # Green
        img_array[:, :, 0] = 100  # Blue
        img_array[:, :, 2] = 120  # Red
        
        # Add some variation
        noise = np.random.randint(-30, 30, (224, 224, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    return test_image_path

def test_inference():
    """Test the inference system."""
    print("=" * 70)
    print("TESTING REAL-WORLD INFERENCE")
    print("=" * 70)
    
    # Create test image
    test_image = create_test_image()
    
    # Test 1: Import test
    print("\n1. Testing imports...")
    try:
        from inference_real_world import RealWorldInference
        print("   [OK] Imports successful")
    except Exception as e:
        print(f"   [ERROR] Import failed: {e}")
        return
    
    # Test 2: Find a model
    print("\n2. Looking for trained model...")
    model_paths = [
        Path('models/enhanced_best.h5'),
        Path('models/plantvillage_robust_best.h5'),
        Path('models/plantvillage_robust_final.h5'),
        Path('models/robust_plantvillage_best.h5')
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            print(f"   [OK] Found model: {model_path}")
            break
    
    if not model_path:
        print("   [WARNING] No trained model found. Creating dummy model...")
        # Create a dummy model for testing
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(6, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        Path('models').mkdir(exist_ok=True)
        model_path = Path('models/test_model.h5')
        model.save(model_path)
        print(f"   [OK] Created test model: {model_path}")
    
    # Test 3: Initialize inference without TTA
    print("\n3. Testing inference without TTA...")
    try:
        inference_no_tta = RealWorldInference(
            model_path=str(model_path),
            preprocessing_mode='fast',
            use_tta=False,
            verbose=False
        )
        
        result = inference_no_tta.predict_single_image(test_image)
        print(f"   [OK] Prediction: {result['prediction']}")
        print(f"   [OK] Confidence: {result['confidence']:.2%}")
        print(f"   [OK] Time: {result['timing']['total_ms']:.2f} ms")
        
    except Exception as e:
        print(f"   [ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Initialize inference with TTA
    print("\n4. Testing inference with TTA...")
    try:
        inference_tta = RealWorldInference(
            model_path=str(model_path),
            preprocessing_mode='fast',
            use_tta=True,
            tta_count=3,  # Fewer for testing
            verbose=False
        )
        
        result = inference_tta.predict_single_image(test_image)
        print(f"   [OK] Prediction: {result['prediction']}")
        print(f"   [OK] Confidence: {result['confidence']:.2%}")
        print(f"   [OK] Uncertainty: {result['uncertainty']:.4f}")
        print(f"   [OK] Time: {result['timing']['total_ms']:.2f} ms")
        
    except Exception as e:
        print(f"   [ERROR] TTA inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test different preprocessing modes
    print("\n5. Testing preprocessing modes...")
    modes = ['legacy', 'fast', 'default']
    
    for mode in modes:
        try:
            inference = RealWorldInference(
                model_path=str(model_path),
                preprocessing_mode=mode,
                use_tta=False,
                verbose=False
            )
            
            result = inference.predict_single_image(test_image)
            print(f"   [OK] {mode:10} - Time: {result['timing']['total_ms']:.2f} ms")
            
        except Exception as e:
            print(f"   [ERROR] {mode} mode failed: {e}")
    
    # Test 6: Batch prediction
    print("\n6. Testing batch prediction...")
    try:
        # Create multiple test images
        test_dir = Path("test_images")
        for i in range(3):
            img_path = test_dir / f"test_{i}.jpg"
            if not img_path.exists():
                # Create variations
                img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(img_path)
        
        # Run batch prediction
        inference = RealWorldInference(
            model_path=str(model_path),
            preprocessing_mode='fast',
            use_tta=False,
            verbose=False
        )
        
        results = inference.predict_directory(test_dir, output_format='json')
        print(f"   [OK] Processed {len(results)} images")
        
    except Exception as e:
        print(f"   [ERROR] Batch prediction failed: {e}")
    
    # Test 7: Supported formats
    print("\n7. Testing image format support...")
    from inference_real_world import SUPPORTED_FORMATS
    print(f"   [OK] Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("[OK] All basic tests completed")
    print("\nRecommendations:")
    print("- Use 'fast' mode for real-time applications")
    print("- Use 'default' mode with TTA for maximum accuracy")
    print("- TTA adds ~3-5x overhead but improves robustness")
    print("\nNext steps:")
    print("1. Test with real plant images from the internet")
    print("2. Run benchmark to compare TTA performance")
    print("3. Process a directory of diverse images")

if __name__ == "__main__":
    test_inference()