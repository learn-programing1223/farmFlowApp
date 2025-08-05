import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import zipfile
import requests

from dataset_harmonizer import PlantDiseaseHarmonizer
# Use simple preprocessing to avoid albumentations issues
from preprocessing_simple import CrossCropPreprocessor


class MultiDatasetLoader:
    """
    Loads and harmonizes multiple plant disease datasets for universal model training.
    Supports PlantVillage, PlantDoc, PlantNet, and Kaggle datasets.
    """
    
    def __init__(self, base_data_dir: str = './data', 
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            base_data_dir: Base directory for all datasets
            target_size: Target image size
        """
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        
        # Initialize components
        self.harmonizer = PlantDiseaseHarmonizer()
        self.preprocessor = CrossCropPreprocessor(target_size)
        
        # Dataset paths
        self.dataset_paths = {
            'plantvillage': self.base_data_dir / 'PlantVillage',
            'plantdoc': self.base_data_dir / 'PlantDoc',
            'plantnet': self.base_data_dir / 'PlantNet',
            'kaggle': self.base_data_dir / 'KagglePlantPathology',
            'augmented': self.base_data_dir / 'augmented'
        }
        
        # Cache for loaded data
        self.data_cache = {}
        
    def download_dataset_info(self):
        """
        Provides information about dataset download sources.
        """
        dataset_info = {
            'PlantVillage': {
                'url': 'https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset',
                'size': '~3.2GB',
                'images': 54303,
                'classes': 38,
                'format': 'folder structure'
            },
            'PlantDoc': {
                'url': 'https://github.com/pratikkayal/PlantDoc-Dataset',
                'size': '~110MB',
                'images': 2598,
                'classes': 27,
                'format': 'images + CSV annotations'
            },
            'PlantNet': {
                'url': 'https://zenodo.org/record/5645731',
                'size': '~15GB (can use subset)',
                'images': 306146,
                'classes': 1081,
                'format': 'tar.gz archives'
            },
            'KagglePlantPathology': {
                'url': 'https://www.kaggle.com/c/plant-pathology-2021-fgvc8',
                'size': '~1GB',
                'images': 18632,
                'classes': 12,
                'format': 'images + CSV'
            }
        }
        
        print("\n=== Dataset Download Information ===")
        for dataset, info in dataset_info.items():
            print(f"\n{dataset}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        return dataset_info
    
    def load_plantvillage(self, subset_percent: float = 1.0) -> Tuple[List[str], List[str]]:
        """
        Loads PlantVillage dataset with folder structure:
        PlantVillage/raw/color/
            Pepper__bell___Bacterial_spot/
            Pepper__bell___healthy/
            Potato___Early_blight/
            ...
        """
        print("\nLoading PlantVillage dataset...")
        dataset_path = self.dataset_paths['plantvillage']
        
        # Check for different possible locations
        if (dataset_path / 'raw' / 'color').exists():
            # GitHub version with raw/color structure
            dataset_path = dataset_path / 'raw' / 'color'
        elif not dataset_path.exists():
            print(f"PlantVillage dataset not found at {dataset_path}")
            print("Please download from Kaggle and extract to the data directory")
            return [], []
        
        images = []
        labels = []
        
        # Iterate through class folders
        for class_folder in sorted(dataset_path.iterdir()):
            if class_folder.is_dir() and not class_folder.name.startswith('.'):
                class_name = class_folder.name
                
                # Process images in folder
                image_files = list(class_folder.glob('*.jpg')) + \
                            list(class_folder.glob('*.JPG')) + \
                            list(class_folder.glob('*.png'))
                
                # Apply subset if needed
                if subset_percent < 1.0:
                    num_samples = int(len(image_files) * subset_percent)
                    image_files = np.random.choice(image_files, num_samples, replace=False)
                
                for img_path in image_files:
                    images.append(str(img_path))
                    labels.append(class_name)
        
        print(f"Loaded {len(images)} images from {len(set(labels))} classes")
        
        # Harmonize labels
        images, labels = self.harmonizer.harmonize_dataset('PlantVillage', images, labels)
        
        return images, labels
    
    def load_plantdoc(self) -> Tuple[List[str], List[str]]:
        """
        Loads PlantDoc dataset with bounding box annotations.
        PlantDoc is a detection dataset, but we'll use full images for classification.
        Expected structure:
        PlantDoc/
            TRAIN/
            TEST/
            train_labels.csv
            test_labels.csv
        """
        print("\nLoading PlantDoc dataset...")
        dataset_path = self.dataset_paths['plantdoc']
        
        if not dataset_path.exists():
            print(f"PlantDoc dataset not found at {dataset_path}")
            return [], []
        
        images = []
        labels = []
        
        # Load train and test sets
        for split, csv_name, folder_name in [('train', 'train_labels.csv', 'TRAIN'), 
                                             ('test', 'test_labels.csv', 'TEST')]:
            csv_path = dataset_path / csv_name
            images_dir = dataset_path / folder_name
            
            if csv_path.exists() and images_dir.exists():
                df = pd.read_csv(csv_path)
                
                # PlantDoc has columns: filename, width, height, class, xmin, ymin, xmax, ymax
                # Group by filename to get unique images (since one image can have multiple bboxes)
                unique_images = df.groupby('filename').first().reset_index()
                
                for _, row in unique_images.iterrows():
                    img_name = row['filename']
                    label = row['class']  # This is the disease/plant label
                    
                    img_path = images_dir / img_name
                    if img_path.exists():
                        images.append(str(img_path))
                        labels.append(str(label))  # Ensure label is string
                    else:
                        # Some filenames might not have the exact path
                        # Try to find the image
                        possible_names = [img_name, img_name.replace(' ', '_')]
                        found = False
                        for name in possible_names:
                            if (images_dir / name).exists():
                                images.append(str(images_dir / name))
                                labels.append(str(label))
                                found = True
                                break
                        
                        if not found:
                            # Skip if image not found
                            continue
        
        print(f"Loaded {len(images)} unique images from PlantDoc")
        print(f"Unique PlantDoc labels before harmonization: {len(set(labels))}")
        
        # Show some example labels
        unique_labels = list(set(labels))[:10]
        print(f"Example PlantDoc labels: {unique_labels}")
        
        # Harmonize labels
        images, labels = self.harmonizer.harmonize_dataset('PlantDoc', images, labels)
        
        return images, labels
    
    def load_kaggle_plant_pathology(self) -> Tuple[List[str], List[str]]:
        """
        Loads Kaggle Plant Pathology dataset.
        Expected structure:
        KagglePlantPathology/
            train_images/
            train.csv
        """
        print("\nLoading Kaggle Plant Pathology dataset...")
        dataset_path = self.dataset_paths['kaggle']
        
        if not dataset_path.exists():
            print(f"Kaggle dataset not found at {dataset_path}")
            return [], []
        
        images = []
        labels = []
        
        # Load CSV
        csv_path = dataset_path / 'train.csv'
        images_dir = dataset_path / 'train_images'
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Handle multi-label format
            disease_columns = ['healthy', 'scab', 'frog_eye_leaf_spot', 'rust', 
                             'complex', 'powdery_mildew']
            
            for _, row in df.iterrows():
                img_path = images_dir / f"{row['image']}.jpg"
                if img_path.exists():
                    # Get primary disease label
                    disease_values = [row[col] for col in disease_columns if col in df.columns]
                    if any(disease_values):
                        primary_disease = disease_columns[np.argmax(disease_values)]
                        images.append(str(img_path))
                        labels.append(f"apple_{primary_disease}")
        
        print(f"Loaded {len(images)} images from Kaggle Plant Pathology")
        
        # Harmonize labels
        images, labels = self.harmonizer.harmonize_dataset('Kaggle', images, labels)
        
        return images, labels
    
    def load_augmented_images(self) -> Tuple[List[str], List[str]]:
        """
        Loads synthetic augmented images
        """
        print("\nLoading augmented synthetic images...")
        augmented_path = self.dataset_paths['augmented']
        
        if not augmented_path.exists():
            print(f"No augmented images found at {augmented_path}")
            return [], []
        
        images = []
        labels = []
        
        # Load images from each class directory
        for class_dir in augmented_path.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                
                # Get all images in this class
                for img_path in class_dir.glob('*.png'):
                    images.append(str(img_path))
                    labels.append(class_name)
        
        print(f"Loaded {len(images)} augmented images")
        
        # These are already in universal categories, no need to harmonize
        return images, labels
    
    def load_all_datasets(self, use_cache: bool = True,
                         plantvillage_subset: float = 1.0,
                         include_augmented: bool = False) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Loads all available datasets.
        
        Args:
            use_cache: Whether to use cached data if available
            plantvillage_subset: Percentage of PlantVillage to use (for faster testing)
            include_augmented: Whether to include synthetic augmented images
        
        Returns:
            Dictionary mapping dataset names to (images, labels) tuples
        """
        cache_path = self.base_data_dir / 'dataset_cache.pkl'
        
        if use_cache and cache_path.exists():
            print("Loading datasets from cache...")
            import pickle
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        all_datasets = {}
        
        # Load each dataset
        pv_images, pv_labels = self.load_plantvillage(plantvillage_subset)
        if pv_images:
            all_datasets['PlantVillage'] = (pv_images, pv_labels)
        
        pd_images, pd_labels = self.load_plantdoc()
        if pd_images:
            all_datasets['PlantDoc'] = (pd_images, pd_labels)
        
        kp_images, kp_labels = self.load_kaggle_plant_pathology()
        if kp_images:
            all_datasets['Kaggle'] = (kp_images, kp_labels)
        
        # Load augmented images if requested
        if include_augmented:
            aug_images, aug_labels = self.load_augmented_images()
            if aug_images:
                all_datasets['Augmented'] = (aug_images, aug_labels)
        
        # Cache the loaded data
        if use_cache:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(all_datasets, f)
            print(f"Cached dataset information to {cache_path}")
        
        return all_datasets
    
    def create_balanced_dataset(self, datasets: Dict[str, Tuple[List[str], List[str]]],
                               samples_per_class: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a balanced dataset from multiple sources.
        
        Returns:
            Tuple of (images, one_hot_labels) as numpy arrays
        """
        print("\nCreating balanced dataset...")
        
        # Get balanced samples
        balanced_images, balanced_labels = self.harmonizer.create_balanced_dataset(
            datasets, samples_per_class
        )
        
        # Create label encoder
        unique_labels = sorted(set(balanced_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"\nProcessing {len(balanced_images)} images...")
        
        # Process images in parallel
        processed_images = []
        valid_labels = []
        
        def process_image(img_path, label):
            try:
                img = self.preprocessor.preprocess_image(
                    img_path, 
                    apply_augmentation=False,
                    is_training=False
                )
                return img, label
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                return None, None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_image, img_path, label): (img_path, label)
                for img_path, label in zip(balanced_images, balanced_labels)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                img, label = future.result()
                if img is not None:
                    processed_images.append(img)
                    valid_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(processed_images, dtype=np.float32)
        
        # Convert labels to one-hot encoding
        y_indices = [label_to_idx[label] for label in valid_labels]
        y = to_categorical(y_indices, num_classes=len(unique_labels))
        
        print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
        print(f"Classes: {unique_labels}")
        
        # Save label mapping
        self._save_label_mapping(label_to_idx)
        
        return X, y
    
    def _save_label_mapping(self, label_to_idx: Dict[str, int]):
        """Saves label mapping for inference."""
        mapping_path = self.base_data_dir / 'label_mapping.json'
        
        # Create reverse mapping
        idx_to_label = {str(idx): label for label, idx in label_to_idx.items()}
        
        mapping_info = {
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': len(label_to_idx)
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        print(f"Label mapping saved to {mapping_path}")
    
    def prepare_train_val_test_split(self, X: np.ndarray, y: np.ndarray,
                                    val_split: float = 0.15,
                                    test_split: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Splits data into train, validation, and test sets.
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Second split: train vs val
        val_size = val_split / (1 - test_split)  # Adjust for the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, 
            stratify=y_temp.argmax(axis=1)
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Print split information
        print("\nDataset splits:")
        for split_name, (X_split, y_split) in splits.items():
            print(f"  {split_name}: {X_split.shape[0]} samples")
        
        # Save splits for reproducibility
        self._save_splits(splits)
        
        return splits
    
    def _save_splits(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Saves data splits to disk with compression and error handling."""
        import shutil
        
        splits_dir = self.base_data_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        # Check available disk space
        stat = shutil.disk_usage(str(self.base_data_dir))
        available_gb = stat.free / (1024**3)
        
        # Estimate required space (rough estimate)
        total_size = sum(X.nbytes + y.nbytes for X, y in splits.values())
        required_gb = total_size / (1024**3) * 1.5  # 1.5x for safety
        
        if available_gb < required_gb:
            print(f"\nWARNING: Low disk space!")
            print(f"Available: {available_gb:.2f} GB")
            print(f"Required: {required_gb:.2f} GB")
            print("Attempting compressed save...")
            use_compression = True
        else:
            use_compression = False
            
        try:
            for split_name, (X, y) in splits.items():
                print(f"\nSaving {split_name} split...")
                
                if use_compression:
                    # Save with compression
                    np.savez_compressed(
                        splits_dir / f'{split_name}_data.npz',
                        X=X, y=y
                    )
                    print(f"  Saved compressed: {split_name}_data.npz")
                else:
                    # Try regular save with error handling
                    try:
                        np.save(splits_dir / f'X_{split_name}.npy', X)
                        np.save(splits_dir / f'y_{split_name}.npy', y)
                        print(f"  Saved: X_{split_name}.npy, y_{split_name}.npy")
                    except OSError as e:
                        print(f"  Regular save failed: {e}")
                        print("  Falling back to compressed save...")
                        np.savez_compressed(
                            splits_dir / f'{split_name}_data.npz',
                            X=X, y=y
                        )
                        print(f"  Saved compressed: {split_name}_data.npz")
                        
        except Exception as e:
            print(f"\nERROR saving splits: {e}")
            print("\nTrying alternative: saving metadata only...")
            # Save just the file paths and labels instead
            metadata = {
                'shape': {name: (X.shape, y.shape) for name, (X, y) in splits.items()},
                'dtype': {name: (str(X.dtype), str(y.dtype)) for name, (X, y) in splits.items()},
                'samples': {name: len(X) for name, (X, y) in splits.items()}
            }
            with open(splits_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {splits_dir / 'metadata.json'}")
            print("\nNOTE: Data arrays not saved due to disk space. Training will proceed without cache.")
            return
            
        print(f"\nSplits saved successfully to {splits_dir}")
    
    def load_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Loads previously saved splits (handles both compressed and regular formats)."""
        splits_dir = self.base_data_dir / 'splits'
        
        if not splits_dir.exists():
            raise ValueError("No saved splits found. Run prepare_train_val_test_split first.")
        
        splits = {}
        for split_name in ['train', 'val', 'test']:
            # Try compressed format first
            compressed_path = splits_dir / f'{split_name}_data.npz'
            if compressed_path.exists():
                data = np.load(compressed_path)
                X = data['X']
                y = data['y']
                splits[split_name] = (X, y)
            else:
                # Try regular format
                X_path = splits_dir / f'X_{split_name}.npy'
                y_path = splits_dir / f'y_{split_name}.npy'
                if X_path.exists() and y_path.exists():
                    X = np.load(X_path)
                    y = np.load(y_path)
                    splits[split_name] = (X, y)
                else:
                    print(f"WARNING: Could not load {split_name} split")
        
        return splits


def main():
    """Main function to demonstrate dataset loading and preparation."""
    # Initialize loader
    loader = MultiDatasetLoader(base_data_dir='./data')
    
    # Show dataset information
    loader.download_dataset_info()
    
    # Load all datasets (using small subset for testing)
    all_datasets = loader.load_all_datasets(
        use_cache=True,
        plantvillage_subset=0.1  # Use 10% for faster testing
    )
    
    if not all_datasets:
        print("\nNo datasets found. Please download at least one dataset.")
        return
    
    # Create balanced dataset
    X, y = loader.create_balanced_dataset(all_datasets, samples_per_class=100)
    
    # Split data
    splits = loader.prepare_train_val_test_split(X, y)
    
    print("\nDataset preparation complete!")
    print(f"Ready for training with {len(splits['train'][0])} training samples")


if __name__ == "__main__":
    main()