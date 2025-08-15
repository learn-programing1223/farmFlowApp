"""
Advanced Preprocessing Module for Plant Disease Detection
==========================================================

This module implements sophisticated image preprocessing techniques to standardize
internet/real-world images to match training data quality. The preprocessing pipeline
includes illumination correction, CLAHE enhancement, noise reduction, and color
constancy normalization.

Key Features:
- Illumination correction using Gaussian blur background subtraction
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
- Bilateral filtering for edge-preserving noise reduction
- Color constancy normalization using gray world assumption
- Configurable parameters for all preprocessing steps
- Support for single image and batch processing

Author: PlantPulse Team
Date: 2025
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
import warnings
from pathlib import Path


class AdvancedPreprocessor:
    """
    Advanced image preprocessor for plant disease detection.
    
    This class implements a comprehensive preprocessing pipeline designed to
    standardize real-world images to match the quality and characteristics
    of training data. Each preprocessing step is configurable and can be
    enabled/disabled based on requirements.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        illumination_correction: bool = True,
        gaussian_kernel_size: int = 51,
        clahe_enhancement: bool = True,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        bilateral_filtering: bool = True,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        color_constancy: bool = True,
        normalize_output: bool = True,
        preserve_aspect_ratio: bool = True
    ):
        """
        Initialize the advanced preprocessor with configurable parameters.
        
        Args:
            target_size: Output image size (width, height)
            illumination_correction: Enable/disable illumination correction
            gaussian_kernel_size: Kernel size for Gaussian blur in illumination correction
            clahe_enhancement: Enable/disable CLAHE enhancement
            clahe_clip_limit: Threshold for contrast limiting in CLAHE
            clahe_grid_size: Size of grid for histogram equalization
            bilateral_filtering: Enable/disable bilateral filtering
            bilateral_d: Diameter of pixel neighborhood for bilateral filter
            bilateral_sigma_color: Filter sigma in the color space
            bilateral_sigma_space: Filter sigma in the coordinate space
            color_constancy: Enable/disable color constancy normalization
            normalize_output: Normalize output to [0, 1] range
            preserve_aspect_ratio: Maintain aspect ratio during resizing
        """
        self.target_size = target_size
        self.illumination_correction = illumination_correction
        self.gaussian_kernel_size = gaussian_kernel_size
        self.clahe_enhancement = clahe_enhancement
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.bilateral_filtering = bilateral_filtering
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.color_constancy = color_constancy
        self.normalize_output = normalize_output
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Create CLAHE object if enabled
        if self.clahe_enhancement:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination using Gaussian blur background subtraction.
        
        This method estimates the background illumination using a large Gaussian
        kernel and subtracts it from the original image to remove lighting
        variations while preserving local details.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Illumination-corrected image
        """
        # Convert to LAB color space for better illumination handling
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Estimate background illumination using Gaussian blur
        background = cv2.GaussianBlur(
            l_channel, 
            (self.gaussian_kernel_size, self.gaussian_kernel_size), 
            0
        )
        
        # Subtract background and normalize
        corrected_l = cv2.subtract(l_channel, background)
        mean_val = int(np.mean(background))
        corrected_l = cv2.add(corrected_l, mean_val)
        
        # Merge channels and convert back to BGR
        corrected_lab = cv2.merge([corrected_l, a_channel, b_channel])
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        return corrected_bgr
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.
        
        CLAHE improves local contrast and brings out details in both bright and
        dark regions of the image. Processing in LAB color space preserves
        color information while enhancing contrast.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        enhanced_l = self.clahe.apply(l_channel)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving noise reduction.
        
        Bilateral filtering reduces noise while keeping edges sharp, which is
        crucial for preserving disease symptoms and leaf boundaries.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Filtered image
        """
        filtered = cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return filtered
    
    def apply_color_constancy(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color constancy normalization using gray world assumption.
        
        This method corrects color casts by assuming the average color of the
        scene should be gray, helping to standardize images taken under
        different lighting conditions.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Color-corrected image
        """
        # Calculate mean values for each channel
        b_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        r_mean = np.mean(image[:, :, 2])
        
        # Calculate gray world correction factors
        avg_gray = (b_mean + g_mean + r_mean) / 3.0
        
        # Avoid division by zero
        b_factor = avg_gray / (b_mean + 1e-10)
        g_factor = avg_gray / (g_mean + 1e-10)
        r_factor = avg_gray / (r_mean + 1e-10)
        
        # Apply correction
        corrected = image.astype(np.float32)
        corrected[:, :, 0] *= b_factor
        corrected[:, :, 1] *= g_factor
        corrected[:, :, 2] *= r_factor
        
        # Clip values to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size with optional aspect ratio preservation.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        if self.preserve_aspect_ratio:
            # Calculate scaling factors
            h, w = image.shape[:2]
            scale = min(self.target_size[0] / w, self.target_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize maintaining aspect ratio
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Create canvas and center the image
            canvas = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            y_offset = (self.target_size[1] - new_h) // 2
            x_offset = (self.target_size[0] - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            # Direct resize without preserving aspect ratio
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def preprocess_single(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Preprocess a single image through the complete pipeline.
        
        Args:
            image: Input image as numpy array, file path string, or Path object
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
        
        # Ensure image is in BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Apply preprocessing pipeline
        processed = image.copy()
        
        if self.illumination_correction:
            processed = self.correct_illumination(processed)
        
        if self.clahe_enhancement:
            processed = self.apply_clahe(processed)
        
        if self.bilateral_filtering:
            processed = self.apply_bilateral_filter(processed)
        
        if self.color_constancy:
            processed = self.apply_color_constancy(processed)
        
        # Resize to target size
        processed = self.resize_image(processed)
        
        # Convert to RGB (from BGR)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Normalize if requested
        if self.normalize_output:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def preprocess_batch(
        self, 
        images: List[Union[np.ndarray, str, Path]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images (numpy arrays or file paths)
            show_progress: Display progress bar if available
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        processed_batch = []
        
        # Try to import tqdm for progress bar
        try:
            if show_progress:
                from tqdm import tqdm
                images = tqdm(images, desc="Preprocessing images")
        except ImportError:
            pass
        
        for img in images:
            try:
                processed = self.preprocess_single(img)
                processed_batch.append(processed)
            except Exception as e:
                warnings.warn(f"Failed to process image: {e}")
                # Add blank image to maintain batch size
                if self.normalize_output:
                    blank = np.zeros((*self.target_size[::-1], 3), dtype=np.float32)
                else:
                    blank = np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
                processed_batch.append(blank)
        
        return np.array(processed_batch)
    
    def visualize_preprocessing_steps(
        self, 
        image: Union[np.ndarray, str, Path],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize all preprocessing steps for debugging and analysis.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Visualization grid showing each preprocessing step
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            original = cv2.imread(str(image))
        else:
            original = image.copy()
        
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Resize original for visualization
        original_resized = self.resize_image(original)
        
        # Apply each step individually
        steps = [("Original", cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))]
        
        current = original.copy()
        
        if self.illumination_correction:
            current = self.correct_illumination(current)
            current_resized = self.resize_image(current)
            steps.append(("Illumination Corrected", cv2.cvtColor(current_resized, cv2.COLOR_BGR2RGB)))
        
        if self.clahe_enhancement:
            current = self.apply_clahe(current)
            current_resized = self.resize_image(current)
            steps.append(("CLAHE Enhanced", cv2.cvtColor(current_resized, cv2.COLOR_BGR2RGB)))
        
        if self.bilateral_filtering:
            current = self.apply_bilateral_filter(current)
            current_resized = self.resize_image(current)
            steps.append(("Bilateral Filtered", cv2.cvtColor(current_resized, cv2.COLOR_BGR2RGB)))
        
        if self.color_constancy:
            current = self.apply_color_constancy(current)
            current_resized = self.resize_image(current)
            steps.append(("Color Constancy", cv2.cvtColor(current_resized, cv2.COLOR_BGR2RGB)))
        
        # Create visualization grid
        n_steps = len(steps)
        grid_cols = min(3, n_steps)
        grid_rows = (n_steps + grid_cols - 1) // grid_cols
        
        # Calculate grid dimensions
        img_h, img_w = self.target_size[1], self.target_size[0]
        grid_h = grid_rows * (img_h + 40)  # Add space for labels
        grid_w = grid_cols * img_w
        
        # Create grid canvas
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # Place images in grid
        for idx, (label, img) in enumerate(steps):
            row = idx // grid_cols
            col = idx % grid_cols
            
            y_start = row * (img_h + 40) + 30
            x_start = col * img_w
            
            # Place image
            if self.normalize_output and idx == len(steps) - 1:
                img = (img * 255).astype(np.uint8)
            elif img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            grid[y_start:y_start+img_h, x_start:x_start+img_w] = img
            
            # Add label (simplified without cv2.putText to avoid font issues)
            # In production, you would use cv2.putText here
        
        # Save if path provided
        if save_path:
            # Convert RGB to BGR for saving with cv2
            grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, grid_bgr)
        
        return grid


def create_default_preprocessor() -> AdvancedPreprocessor:
    """
    Create a preprocessor with default settings optimized for plant disease detection.
    
    Returns:
        AdvancedPreprocessor instance with recommended settings
    """
    return AdvancedPreprocessor(
        target_size=(224, 224),
        illumination_correction=True,
        gaussian_kernel_size=51,
        clahe_enhancement=True,
        clahe_clip_limit=3.0,
        clahe_grid_size=(8, 8),
        bilateral_filtering=True,
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        color_constancy=True,
        normalize_output=True,
        preserve_aspect_ratio=True
    )


def create_fast_preprocessor() -> AdvancedPreprocessor:
    """
    Create a faster preprocessor with reduced processing for real-time applications.
    
    Returns:
        AdvancedPreprocessor instance optimized for speed
    """
    return AdvancedPreprocessor(
        target_size=(224, 224),
        illumination_correction=False,  # Skip for speed
        clahe_enhancement=True,
        clahe_clip_limit=2.0,  # Lower clip limit
        clahe_grid_size=(4, 4),  # Smaller grid
        bilateral_filtering=False,  # Skip for speed
        color_constancy=True,
        normalize_output=True,
        preserve_aspect_ratio=False  # Direct resize for speed
    )


def create_minimal_preprocessor() -> AdvancedPreprocessor:
    """
    Create a minimal preprocessor with only essential operations.
    
    Returns:
        AdvancedPreprocessor instance with minimal processing
    """
    return AdvancedPreprocessor(
        target_size=(224, 224),
        illumination_correction=False,
        clahe_enhancement=False,
        bilateral_filtering=False,
        color_constancy=False,
        normalize_output=True,
        preserve_aspect_ratio=True
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Advanced Preprocessing Module for Plant Disease Detection")
    print("=" * 60)
    
    # Create preprocessor with default settings
    preprocessor = create_default_preprocessor()
    
    # Example: Process a single image
    # img_path = "path/to/your/image.jpg"
    # processed = preprocessor.preprocess_single(img_path)
    # print(f"Processed image shape: {processed.shape}")
    # print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Example: Process a batch of images
    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # batch = preprocessor.preprocess_batch(image_paths)
    # print(f"Batch shape: {batch.shape}")
    
    # Example: Visualize preprocessing steps
    # viz = preprocessor.visualize_preprocessing_steps(
    #     "test_image.jpg",
    #     save_path="preprocessing_visualization.png"
    # )
    
    print("\nPreprocessor configuration:")
    print(f"  Target size: {preprocessor.target_size}")
    print(f"  Illumination correction: {preprocessor.illumination_correction}")
    print(f"  CLAHE enhancement: {preprocessor.clahe_enhancement}")
    print(f"  Bilateral filtering: {preprocessor.bilateral_filtering}")
    print(f"  Color constancy: {preprocessor.color_constancy}")
    print(f"  Output normalization: {preprocessor.normalize_output}")
    
    print("\nAvailable preset configurations:")
    print("  - create_default_preprocessor(): Full pipeline (best quality)")
    print("  - create_fast_preprocessor(): Optimized for speed")
    print("  - create_minimal_preprocessor(): Minimal processing")