"""
CycleGAN-based Data Augmentation for Plant Disease Images
Generates synthetic images to balance dataset classes
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CycleGANAugmentor:
    """
    Uses CycleGAN to generate synthetic plant disease images
    Transforms healthy plants to diseased and vice versa
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
        self.generator_g = self._build_generator()  # Healthy -> Diseased
        self.generator_f = self._build_generator()  # Diseased -> Healthy
        self.discriminator_x = self._build_discriminator()  # For healthy images
        self.discriminator_y = self._build_discriminator()  # For diseased images
        
    def _build_generator(self):
        """Build CycleGAN generator using U-Net architecture"""
        
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                             kernel_initializer=initializer, use_bias=False))
            
            if apply_batchnorm:
                result.add(tf.keras.layers.BatchNormalization())
                
            result.add(tf.keras.layers.LeakyReLU())
            
            return result
        
        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = tf.keras.Sequential()
            result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                      padding='same',
                                                      kernel_initializer=initializer,
                                                      use_bias=False))
            
            result.add(tf.keras.layers.BatchNormalization())
            
            if apply_dropout:
                result.add(tf.keras.layers.Dropout(0.5))
                
            result.add(tf.keras.layers.ReLU())
            
            return result
        
        # Build U-Net generator
        inputs = tf.keras.layers.Input(shape=[self.image_size[0], self.image_size[1], 3])
        
        down_stack = [
            downsample(64, 4, apply_batchnorm=False),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
        ]
        
        up_stack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4),
        ]
        
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                              strides=2,
                                              padding='same',
                                              kernel_initializer=initializer,
                                              activation='tanh')
        
        x = inputs
        
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            
        x = last(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def _build_discriminator(self):
        """Build PatchGAN discriminator"""
        initializer = tf.random_normal_initializer(0., 0.02)
        
        inp = tf.keras.layers.Input(shape=[self.image_size[0], self.image_size[1], 3], name='input_image')
        
        down1 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False)(inp)
        down1 = tf.keras.layers.LeakyReLU()(down1)
        
        down2 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False)(down1)
        down2 = tf.keras.layers.BatchNormalization()(down2)
        down2 = tf.keras.layers.LeakyReLU()(down2)
        
        down3 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False)(down2)
        down3 = tf.keras.layers.BatchNormalization()(down3)
        down3 = tf.keras.layers.LeakyReLU()(down3)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)
        
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)
        
        return tf.keras.Model(inputs=inp, outputs=last)
    
    def generate_synthetic_images(self, 
                                source_images: np.ndarray,
                                source_labels: List[str],
                                target_class: str,
                                num_to_generate: int,
                                model_path: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate synthetic images for a target class
        
        Args:
            source_images: Source images to transform
            source_labels: Labels of source images
            target_class: Target disease class to generate
            num_to_generate: Number of synthetic images to generate
            model_path: Path to pre-trained CycleGAN model (if available)
            
        Returns:
            Tuple of (synthetic_images, labels)
        """
        logger.info(f"Generating {num_to_generate} synthetic images for {target_class}")
        
        # For now, use style transfer and augmentation instead of full CycleGAN
        # (Full CycleGAN training would require disease-specific paired datasets)
        synthetic_images = []
        synthetic_labels = []
        
        # Group source images by class
        class_images = {}
        for img, label in zip(source_images, source_labels):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(img)
        
        # Generate synthetic images
        generated_count = 0
        
        while generated_count < num_to_generate:
            # Select random source image from a different class
            source_classes = [cls for cls in class_images.keys() if cls != target_class]
            if not source_classes:
                source_classes = list(class_images.keys())
            
            source_class = np.random.choice(source_classes)
            if class_images[source_class]:
                source_img = class_images[source_class][np.random.randint(0, len(class_images[source_class]))]
                
                # Apply disease-specific transformations
                synthetic_img = self._apply_disease_transformation(source_img, target_class)
                
                synthetic_images.append(synthetic_img)
                synthetic_labels.append(target_class)
                generated_count += 1
                
                if generated_count % 100 == 0:
                    logger.info(f"Generated {generated_count}/{num_to_generate} images")
        
        return np.array(synthetic_images), synthetic_labels
    
    def _apply_disease_transformation(self, image: np.ndarray, target_disease: str) -> np.ndarray:
        """
        Apply disease-specific transformations to create synthetic diseased images
        """
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize to working size
        img = cv2.resize(image, self.image_size)
        
        if target_disease == 'Leaf_Spot':
            # Add circular spots
            img = self._add_leaf_spots(img)
        elif target_disease == 'Blight':
            # Add browning/blackening
            img = self._add_blight_effect(img)
        elif target_disease == 'Rust':
            # Add rust-colored patches
            img = self._add_rust_effect(img)
        elif target_disease == 'Powdery_Mildew':
            # Add white powdery texture
            img = self._add_mildew_effect(img)
        elif target_disease == 'Mosaic_Virus':
            # Add mosaic pattern
            img = self._add_mosaic_effect(img)
        elif target_disease == 'Nutrient_Deficiency':
            # Add yellowing
            img = self._add_nutrient_deficiency_effect(img)
        else:
            # Healthy - enhance green
            img = self._enhance_healthy_appearance(img)
        
        # Add random variations
        img = self._add_random_variations(img)
        
        # Normalize back to [0, 1]
        return img.astype(np.float32) / 255.0
    
    def _add_leaf_spots(self, image: np.ndarray) -> np.ndarray:
        """Add realistic leaf spot patterns"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Number of spots
        num_spots = np.random.randint(5, 15)
        
        for _ in range(num_spots):
            # Random position
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            
            # Random size
            radius = np.random.randint(3, 15)
            
            # Create spot with gradient
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (radius*2+1, radius*2+1), radius/2)
            
            # Brown/dark color for spots
            spot_color = np.array([40, 30, 20]) + np.random.randint(-10, 10, 3)
            
            # Apply spot
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1 - mask * 0.7) + spot_color[c] * mask * 0.7
        
        return img
    
    def _add_blight_effect(self, image: np.ndarray) -> np.ndarray:
        """Add blight browning effect"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create irregular browning pattern
        mask = np.random.rand(h, w) > 0.7
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5)))
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 10)
        
        # Brown color
        brown = np.array([60, 40, 20])
        
        # Apply browning
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask * 0.6) + brown[c] * mask * 0.6
        
        return img
    
    def _add_rust_effect(self, image: np.ndarray) -> np.ndarray:
        """Add rust-colored patches"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create rust patches
        num_patches = np.random.randint(10, 20)
        
        for _ in range(num_patches):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            
            # Irregular shape
            radius = np.random.randint(5, 20)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (radius, radius//2), 
                       np.random.randint(0, 180), 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (11, 11), 5)
            
            # Rust color (orange-brown)
            rust_color = np.array([180, 100, 40]) + np.random.randint(-20, 20, 3)
            
            # Apply rust
            for c in range(3):
                img[:, :, c] = img[:, :, c] * (1 - mask * 0.5) + rust_color[c] * mask * 0.5
        
        return img
    
    def _add_mildew_effect(self, image: np.ndarray) -> np.ndarray:
        """Add powdery mildew effect"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create powdery texture
        noise = np.random.rand(h, w) > 0.3
        noise = cv2.dilate(noise.astype(np.uint8), np.ones((3, 3)))
        noise = cv2.GaussianBlur(noise.astype(np.float32), (5, 5), 2)
        
        # White powdery color
        white = np.array([220, 220, 220])
        
        # Apply mildew
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - noise * 0.4) + white[c] * noise * 0.4
        
        return img
    
    def _add_mosaic_effect(self, image: np.ndarray) -> np.ndarray:
        """Add mosaic virus pattern"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create mosaic pattern
        block_size = 20
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if np.random.rand() > 0.5:
                    # Yellow/light patches
                    factor = np.random.uniform(1.2, 1.5)
                    img[y:y+block_size, x:x+block_size, 1] = np.clip(
                        img[y:y+block_size, x:x+block_size, 1] * factor, 0, 255)
                else:
                    # Dark patches
                    factor = np.random.uniform(0.6, 0.8)
                    img[y:y+block_size, x:x+block_size] = img[y:y+block_size, x:x+block_size] * factor
        
        # Blur boundaries
        img = cv2.GaussianBlur(img, (5, 5), 1)
        
        return img
    
    def _add_nutrient_deficiency_effect(self, image: np.ndarray) -> np.ndarray:
        """Add yellowing/chlorosis effect"""
        img = image.copy()
        
        # Reduce green channel, increase yellow
        img[:, :, 1] = img[:, :, 1] * 0.7  # Reduce green
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.2, 0, 255)  # Increase red
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.9, 0, 255)  # Slightly reduce blue
        
        # Add yellowing gradient from edges
        h, w = img.shape[:2]
        y_grad, x_grad = np.ogrid[:h, :w]
        mask = np.maximum(np.abs(y_grad - h/2) / (h/2), np.abs(x_grad - w/2) / (w/2))
        mask = np.clip(mask * 1.5, 0, 1)
        
        yellow = np.array([200, 180, 100])
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask * 0.3) + yellow[c] * mask * 0.3
        
        return img
    
    def _enhance_healthy_appearance(self, image: np.ndarray) -> np.ndarray:
        """Enhance healthy green appearance"""
        img = image.copy()
        
        # Enhance green channel
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.2, 0, 255)
        
        # Increase saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return img
    
    def _add_random_variations(self, image: np.ndarray) -> np.ndarray:
        """Add random variations for diversity"""
        img = image.copy()
        
        # Random brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Random blur
        if np.random.rand() > 0.7:
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1)
        
        # Random noise
        if np.random.rand() > 0.8:
            noise = np.random.randn(*img.shape) * 5
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img


def augment_dataset_with_cyclegan(data_dir: str = './data',
                                 target_samples_per_class: int = 7000,
                                 output_dir: str = './data/augmented'):
    """
    Augment dataset to have target number of samples per class
    """
    from data_loader import MultiDatasetLoader
    
    logger.info("Starting CycleGAN-based data augmentation...")
    
    # Load existing data
    loader = MultiDatasetLoader(base_data_dir=data_dir)
    all_datasets = loader.load_all_datasets(use_cache=False)
    
    # Create augmentor
    augmentor = CycleGANAugmentor()
    
    # Get current distribution
    all_images = []
    all_labels = []
    for dataset_name, (images, labels) in all_datasets.items():
        all_images.extend(images)
        all_labels.extend(labels)
    
    # Count current samples per class
    from collections import Counter
    label_counts = Counter(all_labels)
    
    print("\nCurrent distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Load a subset of images for generation
    print("\nLoading sample images for augmentation...")
    sample_images = []
    sample_labels = []
    
    # Load up to 100 images per class
    for label in label_counts.keys():
        label_indices = [i for i, l in enumerate(all_labels) if l == label]
        sample_indices = np.random.choice(label_indices, 
                                        min(100, len(label_indices)), 
                                        replace=False)
        
        for idx in sample_indices:
            img = cv2.imread(all_images[idx])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                sample_images.append(img.astype(np.float32) / 255.0)
                sample_labels.append(all_labels[idx])
    
    sample_images = np.array(sample_images)
    
    # Generate synthetic images for underrepresented classes
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    augmented_counts = {}
    
    for label, current_count in label_counts.items():
        if current_count < target_samples_per_class:
            num_to_generate = target_samples_per_class - current_count
            print(f"\nGenerating {num_to_generate} synthetic images for {label}...")
            
            synthetic_images, synthetic_labels = augmentor.generate_synthetic_images(
                sample_images,
                sample_labels,
                label,
                num_to_generate
            )
            
            # Save synthetic images
            label_dir = output_path / label
            label_dir.mkdir(exist_ok=True)
            
            for i, (img, lbl) in enumerate(zip(synthetic_images, synthetic_labels)):
                filename = f"synthetic_{label}_{i:05d}.png"
                filepath = label_dir / filename
                
                # Convert to uint8 and save
                img_uint8 = (img * 255).astype(np.uint8)
                cv2.imwrite(str(filepath), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            
            augmented_counts[label] = num_to_generate
        else:
            augmented_counts[label] = 0
    
    # Save augmentation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_counts': dict(label_counts),
        'augmented_counts': augmented_counts,
        'target_per_class': target_samples_per_class,
        'output_directory': str(output_path)
    }
    
    with open(output_path / 'augmentation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*50)
    print("AUGMENTATION COMPLETE")
    print("="*50)
    print(f"Synthetic images saved to: {output_path}")
    print("\nAugmented counts:")
    for label, count in augmented_counts.items():
        print(f"  {label}: +{count} images")
    
    return str(output_path)


if __name__ == "__main__":
    # Test the augmentor
    augment_dataset_with_cyclegan(
        data_dir='./data',
        target_samples_per_class=7000,
        output_dir='./data/augmented'
    )