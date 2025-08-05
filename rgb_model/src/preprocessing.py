import cv2
import numpy as np
from typing import Tuple, Optional, List
import tensorflow as tf
from PIL import Image
try:
    import albumentations as A
except ImportError:
    print("Warning: albumentations not installed. Some augmentations will be limited.")
    A = None
import os

class CrossCropPreprocessor:
    """
    Advanced preprocessing pipeline for cross-crop disease detection.
    Handles background complexity, illumination normalization, and disease-aware augmentation.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
        # Define augmentation pipeline
        self.train_transform = self._get_train_augmentation()
        self.val_transform = self._get_val_augmentation()
        
    def _get_train_augmentation(self):
        """Disease-aware augmentation pipeline for training."""
        if A is None:
            return None
        return A.Compose([
            # Spatial augmentations (conservative to preserve disease patterns)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(height=self.target_size[0], 
                              width=self.target_size[1], 
                              scale=(0.8, 1.0), 
                              ratio=(0.9, 1.1), 
                              p=1.0),
            
            # Color augmentations (preserve disease color characteristics)
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, 
                                   sat_shift_limit=20, 
                                   val_shift_limit=20, 
                                   p=0.9),
                A.RGBShift(r_shift_limit=15, 
                          g_shift_limit=15, 
                          b_shift_limit=15, 
                          p=0.9),
            ], p=0.9),
            
            # Brightness/contrast (mild to preserve symptoms)
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                      contrast_limit=0.2, 
                                      p=0.5),
            
            # Blur/noise (simulate real-world conditions)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=(3, 5), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), 
                          intensity=(0.1, 0.5), 
                          p=0.5),
            ], p=0.3),
            
            # Advanced augmentations
            A.CoarseDropout(max_holes=8, 
                           max_height=int(0.1 * self.target_size[0]), 
                           max_width=int(0.1 * self.target_size[1]), 
                           fill_value=0, 
                           p=0.3),
        ])
    
    def _get_val_augmentation(self):
        """Minimal augmentation for validation/test."""
        if A is None:
            return None
        return A.Compose([
            A.Resize(height=self.target_size[0], 
                    width=self.target_size[1], 
                    p=1.0),
        ])
    
    def preprocess_image(self, image_path: str, 
                        apply_augmentation: bool = True,
                        is_training: bool = True) -> np.ndarray:
        """
        Preprocesses a single image with background handling and illumination normalization.
        
        Args:
            image_path: Path to the image
            apply_augmentation: Whether to apply augmentation
            is_training: Whether in training mode
            
        Returns:
            Preprocessed image array
        """
        # Load and validate image
        image = self.load_and_validate_image(image_path)
        
        # Apply background-aware preprocessing if needed
        if self.has_complex_background(image):
            image = self.apply_background_subtraction(image)
        
        # Normalize illumination using CLAHE
        image = self.normalize_illumination(image)
        
        # Apply augmentation if requested
        if apply_augmentation and A is not None:
            transform = self.train_transform if is_training else self.val_transform
            if transform is not None:
                augmented = transform(image=image)
                image = augmented['image']
            else:
                # Fallback to simple resize
                image = cv2.resize(image, self.target_size)
        else:
            # Just resize if no augmentation
            image = cv2.resize(image, self.target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_and_validate_image(self, image_path: str) -> np.ndarray:
        """Loads and validates an image."""
        try:
            # Try loading with cv2 first
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL for problematic formats
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Validate image
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image as fallback
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
    
    def has_complex_background(self, image: np.ndarray) -> bool:
        """
        Detects if image has complex background using edge detection.
        Complex backgrounds have many edges outside the central region.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density in border regions
        h, w = edges.shape
        border_size = int(min(h, w) * 0.2)
        
        # Extract border regions
        top_border = edges[:border_size, :]
        bottom_border = edges[-border_size:, :]
        left_border = edges[:, :border_size]
        right_border = edges[:, -border_size:]
        
        # Calculate edge density
        border_edges = np.sum([
            np.sum(top_border > 0),
            np.sum(bottom_border > 0),
            np.sum(left_border > 0),
            np.sum(right_border > 0)
        ])
        
        total_border_pixels = 2 * border_size * (h + w - 2 * border_size)
        edge_density = border_edges / total_border_pixels
        
        # Threshold for complex background
        return edge_density > 0.15
    
    def apply_background_subtraction(self, image: np.ndarray) -> np.ndarray:
        """
        Applies GrabCut algorithm for background subtraction.
        Focuses on the central region where the leaf is likely to be.
        """
        h, w = image.shape[:2]
        
        # Initialize mask - assume plant is in center
        mask = np.zeros((h, w), np.uint8)
        
        # Define rectangle around center (likely plant location)
        rect_x = int(w * 0.1)
        rect_y = int(h * 0.1)
        rect_w = int(w * 0.8)
        rect_h = int(h * 0.8)
        rect = (rect_x, rect_y, rect_w, rect_h)
        
        # Initialize foreground and background models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to image
            result = cv2.bitwise_and(image, image, mask=mask2)
            
            # Fill background with mean color
            bg_color = np.mean(image[mask2 == 0], axis=0).astype(np.uint8)
            result[mask2 == 0] = bg_color
            
            return result
            
        except Exception as e:
            print(f"Background subtraction failed: {str(e)}")
            return image
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes illumination using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Helps with varying lighting conditions in field images.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to RGB
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return result
    
    def create_preprocessing_pipeline(self, batch_size: int = 32):
        """
        Creates a TensorFlow preprocessing pipeline for efficient batch processing.
        """
        def preprocess_tf(image_path):
            # Read and decode image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            
            # Resize
            image = tf.image.resize(image, self.target_size)
            
            # Normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            return image
        
        return preprocess_tf


class MixUpAugmentation:
    """
    Implements MixUp augmentation for better generalization.
    Preserves disease features while increasing data diversity.
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def mixup(self, x1: np.ndarray, y1: np.ndarray, 
              x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies MixUp augmentation to a pair of samples.
        
        Args:
            x1, x2: Input images
            y1, y2: One-hot encoded labels
            
        Returns:
            Mixed image and label
        """
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix inputs
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def mixup_batch(self, x_batch: np.ndarray, 
                    y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies MixUp to an entire batch."""
        batch_size = x_batch.shape[0]
        indices = np.random.permutation(batch_size)
        
        mixed_x = self.alpha * x_batch + (1 - self.alpha) * x_batch[indices]
        mixed_y = self.alpha * y_batch + (1 - self.alpha) * y_batch[indices]
        
        return mixed_x, mixed_y


class CutMixAugmentation:
    """
    Implements CutMix augmentation for disease detection.
    Cuts and pastes patches between training images.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def cutmix(self, x1: np.ndarray, y1: np.ndarray,
               x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies CutMix augmentation."""
        h, w = x1.shape[:2]
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Sample box coordinates
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_x = x1.copy()
        mixed_x[bby1:bby2, bbx1:bbx2] = x2[bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y


def test_preprocessing():
    """Test the preprocessing pipeline."""
    preprocessor = CrossCropPreprocessor()
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, test_image)
    
    # Test preprocessing
    processed = preprocessor.preprocess_image(test_path, apply_augmentation=True)
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)


if __name__ == "__main__":
    test_preprocessing()