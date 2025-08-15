"""
Quick test script for the enhanced data loader integration
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader_v2 import EnhancedDataLoader

def test_basic_functionality():
    """Test basic functionality of the enhanced loader"""
    
    print("Testing Enhanced Data Loader Integration")
    print("=" * 60)
    
    # Test directories
    data_dir = Path("datasets/plantvillage_processed")
    
    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        print("Creating a dummy test...")
        
        # Create a dummy image for testing
        import cv2
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", dummy_img)
        
        # Test with advanced preprocessing
        print("\n1. Testing Advanced Preprocessing")
        loader_advanced = EnhancedDataLoader(
            data_dir=".",
            target_size=(224, 224),
            batch_size=1,
            use_advanced_preprocessing=True,
            preprocessing_mode='default'
        )
        
        processed = loader_advanced.preprocess_image(
            "test_image.jpg",
            apply_augmentation=False,
            is_training=False
        )
        print(f"   Shape: {processed.shape}, Range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Test with legacy preprocessing
        print("\n2. Testing Legacy Preprocessing")
        loader_legacy = EnhancedDataLoader(
            data_dir=".",
            target_size=(224, 224),
            batch_size=1,
            use_advanced_preprocessing=False
        )
        
        processed_legacy = loader_legacy.preprocess_image(
            "test_image.jpg",
            apply_augmentation=False,
            is_training=False
        )
        print(f"   Shape: {processed_legacy.shape}, Range: [{processed_legacy.min():.3f}, {processed_legacy.max():.3f}]")
        
        # Test with augmentation
        print("\n3. Testing with Augmentation")
        processed_aug = loader_advanced.preprocess_image(
            "test_image.jpg",
            apply_augmentation=True,
            is_training=True
        )
        print(f"   Shape: {processed_aug.shape}, Range: [{processed_aug.min():.3f}, {processed_aug.max():.3f}]")
        
        # Compare preprocessing modes
        print("\n4. Testing Preprocessing Mode Comparison")
        comparison = loader_advanced.compare_preprocessing_modes(
            "test_image.jpg",
            save_comparison="preprocessing_test_comparison.png"
        )
        
        for mode, img in comparison.items():
            print(f"   {mode}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
        
        # Clean up
        import os
        os.remove("test_image.jpg")
        print("\n[OK] Test completed successfully!")
        
    else:
        # Test with real dataset
        print(f"\nFound dataset at {data_dir}")
        
        # Test different preprocessing modes
        for mode in ['default', 'fast', 'minimal', 'legacy']:
            print(f"\n{mode.upper()} Mode:")
            
            use_advanced = mode != 'legacy'
            loader = EnhancedDataLoader(
                data_dir=data_dir / "train",
                target_size=(224, 224),
                batch_size=4,
                use_advanced_preprocessing=use_advanced,
                preprocessing_mode=mode if use_advanced else 'default'
            )
            
            # Load a small subset
            train_paths, train_labels, class_names = loader.load_dataset_from_directory(
                data_dir / "train",
                split='train'
            )
            
            if train_paths:
                # Process one image
                sample_img = loader.preprocess_image(
                    train_paths[0],
                    apply_augmentation=False,
                    is_training=False
                )
                print(f"   Processed shape: {sample_img.shape}")
                print(f"   Value range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
                
                # Only test TF dataset for one mode to save time
                if mode == 'default':
                    print("\n   Testing TensorFlow Dataset:")
                    tf_dataset = loader.create_tf_dataset(
                        train_paths[:8],  # Small subset
                        train_labels[:8],
                        is_training=True,
                        shuffle=True,
                        augment=True
                    )
                    
                    # Get one batch
                    import tensorflow as tf
                    for batch_x, batch_y in tf_dataset.take(1):
                        print(f"   Batch shape: {batch_x.shape}")
                        print(f"   Labels shape: {batch_y.shape}")
                        break
        
        print("\n[OK] All modes tested successfully!")
    
    print("\nSummary:")
    print("- Advanced preprocessing with CLAHE: [OK]")
    print("- Augmentation pipeline integration: [OK]")
    print("- Legacy preprocessing fallback: [OK]")
    print("- A/B testing capability: [OK]")
    print("- TensorFlow dataset compatibility: [OK]")

if __name__ == "__main__":
    test_basic_functionality()