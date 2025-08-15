"""
Enhanced Data Loader with Advanced Preprocessing and Augmentation
==================================================================

This module provides an improved data loading system that integrates the advanced
preprocessing and augmentation pipelines for better real-world performance.

Key Features:
- Integrates advanced_preprocessing for consistent image standardization
- Uses augmentation_pipeline for realistic training data variation
- Supports both legacy and new preprocessing modes for A/B testing
- Efficient batch processing with TensorFlow data pipeline
- Configurable preprocessing and augmentation parameters

Author: PlantPulse Team
Date: 2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import cv2
from tqdm import tqdm
import albumentations as A

# Import the new preprocessing and augmentation modules
import sys
sys.path.append(str(Path(__file__).parent))
from advanced_preprocessing import AdvancedPreprocessor, create_default_preprocessor, create_fast_preprocessor
from augmentation_pipeline import AugmentationPipeline


class EnhancedDataLoader:
    """
    Enhanced data loader with integrated advanced preprocessing and augmentation.
    
    This class provides flexible data loading with support for both legacy and
    new preprocessing pipelines, allowing for A/B testing and gradual migration.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        use_advanced_preprocessing: bool = True,
        preprocessing_mode: str = 'default',  # 'default', 'fast', 'minimal', 'legacy'
        augmentation_config: Optional[Dict] = None,
        cache_preprocessed: bool = False
    ):
        """
        Initialize the enhanced data loader.
        
        Args:
            data_dir: Base directory containing dataset
            target_size: Target image size (width, height)
            batch_size: Batch size for data loading
            use_advanced_preprocessing: Enable/disable advanced preprocessing
            preprocessing_mode: Preprocessing mode selection
            augmentation_config: Custom augmentation configuration
            cache_preprocessed: Cache preprocessed images for faster loading
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.batch_size = batch_size
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.preprocessing_mode = preprocessing_mode
        self.augmentation_config = augmentation_config
        self.cache_preprocessed = cache_preprocessed
        
        # Initialize preprocessors
        self._setup_preprocessors()
        
        # Initialize augmentation pipelines
        self._setup_augmentation()
        
        # Cache for preprocessed images
        self.preprocessed_cache = {} if cache_preprocessed else None
        
        # Dataset info
        self.class_names = []
        self.num_classes = 0
        
    def _setup_preprocessors(self):
        """Setup preprocessing pipelines based on mode."""
        if self.use_advanced_preprocessing:
            if self.preprocessing_mode == 'default':
                self.preprocessor = create_default_preprocessor()
            elif self.preprocessing_mode == 'fast':
                self.preprocessor = create_fast_preprocessor()
            elif self.preprocessing_mode == 'minimal':
                from advanced_preprocessing import create_minimal_preprocessor
                self.preprocessor = create_minimal_preprocessor()
            else:
                # Legacy mode - simple resize and normalize
                self.preprocessor = None
        else:
            self.preprocessor = None
        
        # Update preprocessor target size
        if self.preprocessor:
            self.preprocessor.target_size = self.target_size
    
    def _setup_augmentation(self):
        """Setup augmentation pipelines for training and validation."""
        if self.augmentation_config:
            # Use custom configuration
            self.train_augmentation = AugmentationPipeline.create_training_pipeline(
                config=self.augmentation_config,
                image_size=self.target_size[0],
                normalize=False  # We'll normalize after preprocessing
            )
        else:
            # Use default configuration
            self.train_augmentation = AugmentationPipeline.create_training_pipeline(
                image_size=self.target_size[0],
                normalize=False
            )
        
        # Validation augmentation (minimal)
        self.val_augmentation = AugmentationPipeline.create_validation_pipeline(
            image_size=self.target_size[0],
            normalize=False
        )
        
        # Test-time augmentation pipelines
        self.tta_pipelines = AugmentationPipeline.create_test_time_augmentation_pipeline(
            image_size=self.target_size[0],
            normalize=False,
            num_augmentations=5
        )
    
    def load_dataset_from_directory(
        self,
        dataset_path: Union[str, Path],
        split: str = 'train'
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Load dataset from directory structure.
        
        Args:
            dataset_path: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            Tuple of (image_paths, labels, class_names)
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        image_paths = []
        labels = []
        class_names = []
        
        # Get class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        
        print(f"\nLoading {split} data from {dataset_path}")
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Load images from each class
        for class_idx, class_name in enumerate(class_names):
            class_dir = dataset_path / class_name
            
            # Get all image files
            img_files = list(class_dir.glob('*.jpg'))
            img_files += list(class_dir.glob('*.jpeg'))
            img_files += list(class_dir.glob('*.png'))
            
            for img_path in img_files:
                image_paths.append(str(img_path))
                labels.append(class_idx)
        
        # Store class info
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        print(f"Loaded {len(image_paths)} images")
        
        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for idx, count in zip(unique, counts):
            print(f"  {class_names[idx]}: {count} images")
        
        return image_paths, labels, class_names
    
    def preprocess_image(
        self,
        image_path: Union[str, Path],
        apply_augmentation: bool = False,
        is_training: bool = False
    ) -> np.ndarray:
        """
        Preprocess a single image with optional augmentation.
        
        Args:
            image_path: Path to image file
            apply_augmentation: Whether to apply augmentation
            is_training: Whether this is for training (affects augmentation)
            
        Returns:
            Preprocessed image array
        """
        # Check cache if enabled
        if self.cache_preprocessed and str(image_path) in self.preprocessed_cache:
            return self.preprocessed_cache[str(image_path)]
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if requested
        if apply_augmentation:
            if is_training and self.train_augmentation:
                augmented = self.train_augmentation(image=image)
                image = augmented['image']
            elif not is_training and self.val_augmentation:
                augmented = self.val_augmentation(image=image)
                image = augmented['image']
        
        # Apply preprocessing
        if self.use_advanced_preprocessing and self.preprocessor:
            # Advanced preprocessing
            processed = self.preprocessor.preprocess_single(image)
        else:
            # Legacy preprocessing - simple resize and normalize
            processed = cv2.resize(image, self.target_size)
            processed = processed.astype(np.float32) / 255.0
        
        # Cache if enabled
        if self.cache_preprocessed:
            self.preprocessed_cache[str(image_path)] = processed
        
        return processed
    
    def create_tf_dataset(
        self,
        image_paths: List[str],
        labels: List[int],
        is_training: bool = True,
        shuffle: bool = True,
        augment: bool = True
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with preprocessing and augmentation.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            is_training: Whether this is training data
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow Dataset object
        """
        def load_and_preprocess(image_path, label):
            """Load and preprocess image."""
            # Read image file
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            
            # Ensure shape is defined
            image.set_shape([None, None, 3])
            
            # Convert to numpy for preprocessing
            def preprocess_fn(img_tensor):
                img_numpy = img_tensor.numpy()
                
                # Apply augmentation if training
                if is_training and augment and self.train_augmentation:
                    augmented = self.train_augmentation(image=img_numpy)
                    img_numpy = augmented['image']
                elif not is_training and self.val_augmentation:
                    augmented = self.val_augmentation(image=img_numpy)
                    img_numpy = augmented['image']
                
                # Apply preprocessing
                if self.use_advanced_preprocessing and self.preprocessor:
                    # Note: preprocessor expects BGR, but we have RGB from tf.image.decode
                    img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
                    processed = self.preprocessor.preprocess_single(img_bgr)
                else:
                    # Legacy preprocessing
                    processed = cv2.resize(img_numpy, self.target_size)
                    processed = processed.astype(np.float32) / 255.0
                
                return processed.astype(np.float32)
            
            # Use tf.py_function to apply numpy preprocessing
            image = tf.py_function(
                func=preprocess_fn,
                inp=[image],
                Tout=tf.float32
            )
            
            # Set shape for downstream operations
            image.set_shape([self.target_size[1], self.target_size[0], 3])
            
            # One-hot encode label
            label = tf.one_hot(label, depth=self.num_classes)
            
            return image, label
        
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        # Apply preprocessing
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_data_generators(
        self,
        train_dir: Union[str, Path],
        val_dir: Union[str, Path],
        test_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, tf.data.Dataset]:
        """
        Create data generators for train, validation, and optionally test sets.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            test_dir: Optional test data directory
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' datasets
        """
        datasets = {}
        
        # Load training data
        train_paths, train_labels, _ = self.load_dataset_from_directory(train_dir, 'train')
        datasets['train'] = self.create_tf_dataset(
            train_paths, train_labels,
            is_training=True,
            shuffle=True,
            augment=True
        )
        
        # Load validation data
        val_paths, val_labels, _ = self.load_dataset_from_directory(val_dir, 'val')
        datasets['val'] = self.create_tf_dataset(
            val_paths, val_labels,
            is_training=False,
            shuffle=False,
            augment=False
        )
        
        # Load test data if provided
        if test_dir:
            test_paths, test_labels, _ = self.load_dataset_from_directory(test_dir, 'test')
            datasets['test'] = self.create_tf_dataset(
                test_paths, test_labels,
                is_training=False,
                shuffle=False,
                augment=False
            )
        
        return datasets
    
    def apply_test_time_augmentation(
        self,
        image_path: Union[str, Path],
        return_average: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Apply test-time augmentation for improved predictions.
        
        Args:
            image_path: Path to image file
            return_average: Whether to return averaged predictions
            
        Returns:
            Augmented images or averaged result
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented_images = []
        
        # Apply each TTA pipeline
        for tta_pipeline in self.tta_pipelines:
            augmented = tta_pipeline(image=image)['image']
            
            # Apply preprocessing
            if self.use_advanced_preprocessing and self.preprocessor:
                # Convert back to BGR for preprocessor
                aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                processed = self.preprocessor.preprocess_single(aug_bgr)
            else:
                # Legacy preprocessing
                processed = cv2.resize(augmented, self.target_size)
                processed = processed.astype(np.float32) / 255.0
            
            augmented_images.append(processed)
        
        if return_average:
            return np.mean(augmented_images, axis=0)
        else:
            return augmented_images
    
    def compare_preprocessing_modes(
        self,
        sample_image_path: Union[str, Path],
        save_comparison: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare different preprocessing modes for A/B testing.
        
        Args:
            sample_image_path: Path to sample image
            save_comparison: Optional path to save comparison visualization
            
        Returns:
            Dictionary with preprocessed images for each mode
        """
        results = {}
        
        # Load original image
        image = cv2.imread(str(sample_image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {sample_image_path}")
        
        # Legacy preprocessing
        legacy = cv2.resize(image, self.target_size)
        legacy = cv2.cvtColor(legacy, cv2.COLOR_BGR2RGB)
        legacy = legacy.astype(np.float32) / 255.0
        results['legacy'] = legacy
        
        # Advanced preprocessing - default mode
        preprocessor_default = create_default_preprocessor()
        preprocessor_default.target_size = self.target_size
        results['advanced_default'] = preprocessor_default.preprocess_single(image)
        
        # Advanced preprocessing - fast mode
        preprocessor_fast = create_fast_preprocessor()
        preprocessor_fast.target_size = self.target_size
        results['advanced_fast'] = preprocessor_fast.preprocess_single(image)
        
        # Advanced preprocessing - minimal mode
        from advanced_preprocessing import create_minimal_preprocessor
        preprocessor_minimal = create_minimal_preprocessor()
        preprocessor_minimal.target_size = self.target_size
        results['advanced_minimal'] = preprocessor_minimal.preprocess_single(image)
        
        if save_comparison:
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            
            for idx, (mode_name, processed_img) in enumerate(results.items()):
                axes[idx].imshow(processed_img)
                axes[idx].set_title(mode_name)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_comparison, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison saved to {save_comparison}")
        
        return results


def test_enhanced_loader():
    """Test the enhanced data loader with a small batch."""
    print("Testing Enhanced Data Loader")
    print("=" * 60)
    
    # Set up test parameters
    data_dir = Path("C:/Users/aayan/OneDrive/Documents/GitHub/farmFlowApp/rgb_model/datasets/plantvillage_processed")
    
    # Test with advanced preprocessing enabled
    print("\n1. Testing with Advanced Preprocessing (Default Mode)")
    loader_advanced = EnhancedDataLoader(
        data_dir=data_dir / "train",
        target_size=(224, 224),
        batch_size=4,
        use_advanced_preprocessing=True,
        preprocessing_mode='default'
    )
    
    # Load dataset
    train_paths, train_labels, class_names = loader_advanced.load_dataset_from_directory(
        data_dir / "train",
        split='train'
    )
    
    # Test single image preprocessing
    if train_paths:
        print(f"\n2. Testing single image preprocessing")
        sample_image = loader_advanced.preprocess_image(
            train_paths[0],
            apply_augmentation=True,
            is_training=True
        )
        print(f"   Preprocessed image shape: {sample_image.shape}")
        print(f"   Value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    
    # Test with legacy preprocessing
    print("\n3. Testing with Legacy Preprocessing")
    loader_legacy = EnhancedDataLoader(
        data_dir=data_dir / "train",
        target_size=(224, 224),
        batch_size=4,
        use_advanced_preprocessing=False
    )
    
    if train_paths:
        legacy_image = loader_legacy.preprocess_image(
            train_paths[0],
            apply_augmentation=False,
            is_training=False
        )
        print(f"   Legacy preprocessed shape: {legacy_image.shape}")
        print(f"   Value range: [{legacy_image.min():.3f}, {legacy_image.max():.3f}]")
    
    # Test TensorFlow dataset creation
    print("\n4. Testing TensorFlow Dataset Creation")
    if len(train_paths) > 0:
        # Take only first 20 images for quick test
        test_paths = train_paths[:min(20, len(train_paths))]
        test_labels = train_labels[:min(20, len(train_labels))]
        
        tf_dataset = loader_advanced.create_tf_dataset(
            test_paths,
            test_labels,
            is_training=True,
            shuffle=True,
            augment=True
        )
        
        # Get one batch
        for batch_images, batch_labels in tf_dataset.take(1):
            print(f"   Batch images shape: {batch_images.shape}")
            print(f"   Batch labels shape: {batch_labels.shape}")
            print(f"   Image dtype: {batch_images.dtype}")
            print(f"   Value range: [{tf.reduce_min(batch_images):.3f}, {tf.reduce_max(batch_images):.3f}]")
    
    # Test preprocessing mode comparison
    if train_paths:
        print("\n5. Testing Preprocessing Mode Comparison")
        comparison = loader_advanced.compare_preprocessing_modes(
            train_paths[0],
            save_comparison="preprocessing_comparison.png"
        )
        for mode, img in comparison.items():
            print(f"   {mode}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    print("\n✓ Enhanced data loader test completed successfully!")
    print("\nKey Features Tested:")
    print("  - Advanced preprocessing with CLAHE and illumination correction")
    print("  - Augmentation pipeline with realistic internet photo conditions")
    print("  - Legacy preprocessing fallback option")
    print("  - TensorFlow dataset integration")
    print("  - A/B testing comparison modes")


if __name__ == "__main__":
    test_enhanced_loader()