"""
Streaming data loader for handling large datasets like PlantNet
Prevents memory overflow by loading data in batches
"""

import os
import numpy as np
import zipfile
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import json
from tqdm import tqdm
import tensorflow as tf

from dataset_harmonizer import PlantDiseaseHarmonizer
from preprocessing_simple import CrossCropPreprocessor


class PlantNetStreamingLoader:
    """
    Memory-efficient streaming loader for PlantNet-300K dataset.
    Processes images on-the-fly without loading entire dataset into memory.
    """
    
    def __init__(self, zip_path: str, batch_size: int = 32, 
                 max_images: Optional[int] = None,
                 shuffle: bool = True):
        """
        Args:
            zip_path: Path to plantnet_300K.zip
            batch_size: Batch size for streaming
            max_images: Maximum images to use (None for all)
            shuffle: Whether to shuffle file list
        """
        self.zip_path = Path(zip_path)
        self.batch_size = batch_size
        self.max_images = max_images
        self.shuffle = shuffle
        
        self.harmonizer = PlantDiseaseHarmonizer()
        self.preprocessor = CrossCropPreprocessor()
        
        # Get file list from zip
        self.image_files = self._get_image_list()
        
    def _get_image_list(self) -> List[str]:
        """Get list of all image files in zip"""
        print(f"Scanning PlantNet archive at {self.zip_path}...")
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            files = [f for f in zf.namelist() 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    and not f.startswith('__MACOSX')]
        
        print(f"Found {len(files)} images in PlantNet")
        
        if self.max_images and len(files) > self.max_images:
            np.random.seed(42)
            files = np.random.choice(files, self.max_images, replace=False).tolist()
            print(f"Using {len(files)} images (limited by max_images)")
        
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(files)
        
        return files
    
    def stream_batches(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator that yields batches of preprocessed images and labels.
        Memory-efficient: only keeps one batch in memory at a time.
        """
        batch_images = []
        batch_labels = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for img_file in tqdm(self.image_files, desc="Streaming PlantNet"):
                try:
                    # Extract and process single image
                    with zf.open(img_file) as f:
                        img_data = f.read()
                        
                        # Save to temp file for processing
                        temp_path = Path('/tmp') / 'temp_plantnet.jpg'
                        with open(temp_path, 'wb') as tf:
                            tf.write(img_data)
                        
                        # Preprocess image
                        img_array = self.preprocessor.preprocess_image(
                            str(temp_path),
                            apply_augmentation=False,
                            is_training=False
                        )
                        
                        # PlantNet images are healthy plants
                        # This provides diverse "Healthy" examples
                        label = 'Healthy'
                        
                        batch_images.append(img_array)
                        batch_labels.append(label)
                        
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                    
                    # Yield batch when full
                    if len(batch_images) >= self.batch_size:
                        # Convert to arrays
                        X_batch = np.array(batch_images[:self.batch_size])
                        y_batch = self._encode_labels(batch_labels[:self.batch_size])
                        
                        yield X_batch, y_batch
                        
                        # Keep remainder for next batch
                        batch_images = batch_images[self.batch_size:]
                        batch_labels = batch_labels[self.batch_size:]
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
            
            # Yield final partial batch if exists
            if batch_images:
                X_batch = np.array(batch_images)
                y_batch = self._encode_labels(batch_labels)
                yield X_batch, y_batch
    
    def _encode_labels(self, labels: List[str]) -> np.ndarray:
        """Convert labels to one-hot encoding"""
        # 7 universal categories
        categories = ['Healthy', 'Blight', 'Leaf_Spot', 'Powdery_Mildew', 
                     'Rust', 'Mosaic_Virus', 'Nutrient_Deficiency']
        
        label_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        # One-hot encode
        y = np.zeros((len(labels), len(categories)))
        for i, label in enumerate(labels):
            idx = label_to_idx.get(label, 0)  # Default to Healthy if unknown
            y[i, idx] = 1
        
        return y
    
    def create_tf_dataset(self) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset for efficient training.
        Uses tf.data API for optimal performance.
        """
        # Define output signature
        output_signature = (
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
        )
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            self.stream_batches,
            output_signature=output_signature
        )
        
        # Apply optimizations
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def estimate_steps(self) -> int:
        """Estimate number of steps per epoch"""
        return len(self.image_files) // self.batch_size


class CombinedStreamingLoader:
    """
    Combines multiple datasets including streaming PlantNet.
    Manages memory efficiently for large-scale training.
    """
    
    def __init__(self, data_dir: str = './data',
                 use_plantnet_full: bool = True,
                 plantnet_samples: Optional[int] = None):
        """
        Args:
            data_dir: Base data directory
            use_plantnet_full: Use entire PlantNet dataset
            plantnet_samples: Limit PlantNet samples (None for all)
        """
        self.data_dir = Path(data_dir)
        self.use_plantnet_full = use_plantnet_full
        self.plantnet_samples = plantnet_samples
        
        # Regular loader for smaller datasets
        from data_loader import MultiDatasetLoader
        self.regular_loader = MultiDatasetLoader(data_dir)
        
        # PlantNet streaming loader
        self.plantnet_loader = None
        if use_plantnet_full:
            plantnet_path = self._find_plantnet_zip()
            if plantnet_path:
                self.plantnet_loader = PlantNetStreamingLoader(
                    plantnet_path,
                    batch_size=32,
                    max_images=plantnet_samples
                )
    
    def _find_plantnet_zip(self) -> Optional[Path]:
        """Find PlantNet zip file"""
        possible_paths = [
            self.data_dir / 'plantnet_300K.zip',
            Path(__file__).parent / 'data' / 'plantnet_300K.zip',
            self.data_dir / 'PlantNet' / 'plantnet_300K.zip'
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found PlantNet at: {path}")
                return path
        
        print("PlantNet zip not found")
        return None
    
    def prepare_mixed_training_data(self) -> Tuple[tf.data.Dataset, Tuple, Tuple]:
        """
        Prepares mixed dataset with streaming PlantNet and regular datasets.
        
        Returns:
            - Training dataset (streaming)
            - Validation data (in-memory)
            - Test data (in-memory)
        """
        print("Loading regular datasets...")
        
        # Load smaller datasets into memory
        regular_datasets = self.regular_loader.load_all_datasets(
            use_cache=True,
            plantvillage_subset=0.5,  # Use 50% of PlantVillage
            include_augmented=False
        )
        
        # Create balanced dataset from regular sources
        X_regular, y_regular = self.regular_loader.create_balanced_dataset(
            regular_datasets,
            samples_per_class=1000
        )
        
        # Split regular data for validation and test
        from sklearn.model_selection import train_test_split
        
        # 70% train, 15% val, 15% test
        X_train_reg, X_temp, y_train_reg, y_temp = train_test_split(
            X_regular, y_regular, test_size=0.3, random_state=42,
            stratify=y_regular.argmax(axis=1)
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=y_temp.argmax(axis=1)
        )
        
        print(f"Regular training samples: {len(X_train_reg)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create training dataset combining regular + PlantNet streaming
        if self.plantnet_loader:
            print("Creating mixed training dataset with PlantNet streaming...")
            
            # Convert regular training data to TF dataset
            regular_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train_reg, y_train_reg)
            ).batch(32)
            
            # Get PlantNet streaming dataset
            plantnet_dataset = self.plantnet_loader.create_tf_dataset()
            
            # Interleave datasets for balanced training
            # This ensures we see both disease and healthy examples
            mixed_dataset = tf.data.Dataset.sample_from_datasets(
                [regular_dataset, plantnet_dataset],
                weights=[0.7, 0.3]  # 70% disease data, 30% healthy PlantNet
            )
            
            # Apply optimizations
            mixed_dataset = mixed_dataset.prefetch(tf.data.AUTOTUNE)
            
            train_dataset = mixed_dataset
        else:
            # Just use regular data if PlantNet not available
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train_reg, y_train_reg)
            ).batch(32).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, (X_val, y_val), (X_test, y_test)
    
    def estimate_steps_per_epoch(self) -> int:
        """Estimate total steps per epoch for mixed dataset"""
        regular_steps = 0
        plantnet_steps = 0
        
        if hasattr(self, 'X_train_reg'):
            regular_steps = len(self.X_train_reg) // 32
        
        if self.plantnet_loader:
            plantnet_steps = self.plantnet_loader.estimate_steps()
        
        # Weighted average based on sampling ratio
        return int(regular_steps * 0.7 + plantnet_steps * 0.3)


def test_streaming_loader():
    """Test the streaming loader"""
    print("Testing PlantNet Streaming Loader")
    print("=" * 60)
    
    # Find PlantNet zip
    possible_paths = [
        Path('./data/plantnet_300K.zip'),
        Path('./src/data/plantnet_300K.zip'),
        Path('../data/plantnet_300K.zip')
    ]
    
    zip_path = None
    for path in possible_paths:
        if path.exists():
            zip_path = path
            break
    
    if not zip_path:
        print("PlantNet zip not found!")
        return
    
    # Test streaming with small batch
    loader = PlantNetStreamingLoader(
        zip_path,
        batch_size=4,
        max_images=20  # Just test with 20 images
    )
    
    print(f"Testing with {len(loader.image_files)} images")
    
    # Get one batch
    for X_batch, y_batch in loader.stream_batches():
        print(f"Batch shape: {X_batch.shape}")
        print(f"Labels shape: {y_batch.shape}")
        print(f"Label sums: {y_batch.sum(axis=1)}")  # Should all be 1.0
        break
    
    print("✓ Streaming loader working!")


if __name__ == "__main__":
    test_streaming_loader()