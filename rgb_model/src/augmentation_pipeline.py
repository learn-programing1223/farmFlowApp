"""
Augmentation Pipeline for Realistic Internet Photo Conditions
=============================================================

This module implements comprehensive data augmentation pipelines that simulate
real-world conditions found in internet photos and field conditions. The pipelines
are designed to improve model robustness by exposing it to diverse image quality
and environmental conditions during training.

Key Features:
- Realistic lighting variations (brightness, gamma, CLAHE)
- Camera quality simulation (Gaussian noise, ISO noise, blur)
- Motion and focus issues simulation
- Environmental conditions (rain, sun flare, shadows, fog)
- JPEG compression artifacts
- Separate pipelines for training and validation
- Fully configurable parameters through config dictionaries

Author: PlantPulse Team
Date: 2025
"""

import albumentations as A
from albumentations import (
    RandomBrightnessContrast,
    RandomGamma,
    CLAHE,
    GaussNoise,
    ISONoise,
    MotionBlur,
    Defocus,
    GaussianBlur,
    RandomRain,
    RandomSunFlare,
    RandomShadow,
    RandomFog,
    ImageCompression,
    Sharpen,
    HueSaturationValue,
    ChannelShuffle,
    ColorJitter,
    RGBShift,
    RandomToneCurve,
    Posterize,
    Solarize,
    Downscale,
    MultiplicativeNoise,
    Rotate,
    ShiftScaleRotate,
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    CenterCrop,
    RandomResizedCrop,
    CoarseDropout,
    GridDistortion,
    OpticalDistortion,
    ElasticTransform,
    Normalize,
    ToFloat,
    OneOf,
    Compose
)
import cv2
from typing import Dict, Any, Optional, List
import numpy as np


class AugmentationPipeline:
    """
    Comprehensive augmentation pipeline for simulating real-world photo conditions.
    
    This class creates augmentation pipelines that simulate various conditions
    found in internet photos and field images, helping the model become more
    robust to real-world variations.
    """
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration for augmentation parameters.
        
        Returns:
            Dictionary with default augmentation parameters
        """
        return {
            # Lighting variations
            "brightness_contrast": {
                "brightness_limit": 0.3,
                "contrast_limit": 0.3,
                "probability": 0.7
            },
            "gamma": {
                "gamma_limit": (70, 130),
                "probability": 0.5
            },
            "clahe": {
                "clip_limit": 3.0,
                "tile_grid_size": (8, 8),
                "probability": 0.4
            },
            
            # Camera quality simulation
            "gaussian_noise": {
                "var_limit": (10.0, 50.0),
                "mean": 0,
                "probability": 0.3
            },
            "iso_noise": {
                "color_shift": (0.01, 0.05),
                "intensity": (0.1, 0.5),
                "probability": 0.3
            },
            "multiplicative_noise": {
                "multiplier": (0.9, 1.1),
                "per_channel": True,
                "elementwise": True,
                "probability": 0.2
            },
            
            # Blur and focus issues
            "motion_blur": {
                "blur_limit": 7,
                "probability": 0.2
            },
            "defocus": {
                "radius": (3, 7),
                "alias_blur": (0.1, 0.5),
                "probability": 0.15
            },
            "gaussian_blur": {
                "blur_limit": (3, 7),
                "probability": 0.2
            },
            
            # Environmental conditions
            "rain": {
                "slant_lower": -10,
                "slant_upper": 10,
                "drop_length": 20,
                "drop_width": 1,
                "drop_color": (200, 200, 200),
                "blur_value": 5,
                "brightness_coefficient": 0.7,
                "rain_type": "default",  # Can be 'drizzle', 'heavy', 'torrential', 'default'
                "probability": 0.05
            },
            "sun_flare": {
                "flare_roi": (0, 0, 1, 0.5),
                "angle_lower": 0,
                "angle_upper": 1,
                "num_flare_circles_lower": 6,
                "num_flare_circles_upper": 10,
                "src_radius": 400,
                "src_color": (255, 255, 255),
                "probability": 0.05
            },
            "shadow": {
                "shadow_roi": (0, 0.5, 1, 1),
                "num_shadows_lower": 1,
                "num_shadows_upper": 2,
                "shadow_dimension": 5,
                "probability": 0.1
            },
            "fog": {
                "fog_coef_lower": 0.3,
                "fog_coef_upper": 0.8,
                "alpha_coef": 0.08,
                "probability": 0.05
            },
            
            # JPEG compression artifacts
            "jpeg_compression": {
                "quality_lower": 60,
                "quality_upper": 100,
                "probability": 0.6
            },
            
            # Image quality adjustments
            "downscale": {
                "scale_min": 0.5,
                "scale_max": 0.9,
                "interpolation": cv2.INTER_LINEAR,
                "probability": 0.2
            },
            "sharpen": {
                "alpha": (0.2, 0.5),
                "lightness": (0.5, 1.0),
                "probability": 0.3
            },
            
            # Color adjustments
            "hue_saturation": {
                "hue_shift_limit": 20,
                "sat_shift_limit": 30,
                "val_shift_limit": 20,
                "probability": 0.5
            },
            "rgb_shift": {
                "r_shift_limit": 20,
                "g_shift_limit": 20,
                "b_shift_limit": 20,
                "probability": 0.3
            },
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "probability": 0.4
            },
            
            # Geometric transformations
            "rotate": {
                "limit": 30,
                "probability": 0.5
            },
            "shift_scale_rotate": {
                "shift_limit": 0.1,
                "scale_limit": 0.2,
                "rotate_limit": 30,
                "probability": 0.6
            },
            "horizontal_flip": {
                "probability": 0.5
            },
            "vertical_flip": {
                "probability": 0.2
            },
            
            # Distortions
            "grid_distortion": {
                "num_steps": 5,
                "distort_limit": 0.3,
                "probability": 0.1
            },
            "optical_distortion": {
                "distort_limit": 0.5,
                "shift_limit": 0.5,
                "probability": 0.1
            },
            "elastic_transform": {
                "alpha": 1,
                "sigma": 50,
                "alpha_affine": 50,
                "probability": 0.1
            },
            
            # Cutout/Dropout
            "coarse_dropout": {
                "max_holes": 8,
                "max_height": 32,
                "max_width": 32,
                "min_holes": 1,
                "min_height": 8,
                "min_width": 8,
                "fill_value": 0,
                "probability": 0.2
            },
            
            # Advanced color effects
            "channel_shuffle": {
                "probability": 0.05
            },
            "random_tone_curve": {
                "scale": 0.1,
                "probability": 0.1
            },
            "posterize": {
                "num_bits": 4,
                "probability": 0.05
            },
            "solarize": {
                "threshold": 128,
                "probability": 0.05
            }
        }
    
    @staticmethod
    def create_training_pipeline(
        config: Optional[Dict[str, Any]] = None,
        image_size: int = 224,
        normalize: bool = True
    ) -> A.Compose:
        """
        Create comprehensive training augmentation pipeline.
        
        Args:
            config: Configuration dictionary for augmentation parameters
            image_size: Target image size
            normalize: Whether to normalize images to [0, 1]
            
        Returns:
            Albumentations Compose object with training augmentations
        """
        if config is None:
            config = AugmentationPipeline.get_default_config()
        
        transforms = []
        
        # Geometric transformations
        transforms.append(
            RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=1.0
            )
        )
        
        transforms.append(
            ShiftScaleRotate(
                shift_limit=config["shift_scale_rotate"]["shift_limit"],
                scale_limit=config["shift_scale_rotate"]["scale_limit"],
                rotate_limit=config["shift_scale_rotate"]["rotate_limit"],
                border_mode=cv2.BORDER_REFLECT_101,
                p=config["shift_scale_rotate"]["probability"]
            )
        )
        
        transforms.append(HorizontalFlip(p=config["horizontal_flip"]["probability"]))
        transforms.append(VerticalFlip(p=config["vertical_flip"]["probability"]))
        
        # Lighting variations group
        lighting_transforms = [
            RandomBrightnessContrast(
                brightness_limit=config["brightness_contrast"]["brightness_limit"],
                contrast_limit=config["brightness_contrast"]["contrast_limit"],
                p=1.0
            ),
            RandomGamma(
                gamma_limit=config["gamma"]["gamma_limit"],
                p=1.0
            ),
            CLAHE(
                clip_limit=config["clahe"]["clip_limit"],
                tile_grid_size=config["clahe"]["tile_grid_size"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(lighting_transforms, p=0.8)
        )
        
        # Camera quality simulation group
        noise_transforms = [
            GaussNoise(
                var_limit=config["gaussian_noise"]["var_limit"],
                mean=config["gaussian_noise"]["mean"],
                p=1.0
            ),
            ISONoise(
                color_shift=config["iso_noise"]["color_shift"],
                intensity=config["iso_noise"]["intensity"],
                p=1.0
            ),
            MultiplicativeNoise(
                multiplier=config["multiplicative_noise"]["multiplier"],
                per_channel=config["multiplicative_noise"]["per_channel"],
                elementwise=config["multiplicative_noise"]["elementwise"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(noise_transforms, p=0.4)
        )
        
        # Blur and focus issues group
        blur_transforms = [
            MotionBlur(
                blur_limit=config["motion_blur"]["blur_limit"],
                p=1.0
            ),
            Defocus(
                radius=config["defocus"]["radius"],
                alias_blur=config["defocus"]["alias_blur"],
                p=1.0
            ),
            GaussianBlur(
                blur_limit=config["gaussian_blur"]["blur_limit"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(blur_transforms, p=0.3)
        )
        
        # Environmental conditions (low probability)
        environmental_transforms = [
            RandomRain(
                slant_lower=config["rain"]["slant_lower"],
                slant_upper=config["rain"]["slant_upper"],
                drop_length=config["rain"]["drop_length"],
                drop_width=config["rain"]["drop_width"],
                drop_color=config["rain"]["drop_color"],
                blur_value=config["rain"]["blur_value"],
                brightness_coefficient=config["rain"]["brightness_coefficient"],
                rain_type=config["rain"]["rain_type"],
                p=1.0
            ),
            RandomSunFlare(
                flare_roi=config["sun_flare"]["flare_roi"],
                angle_lower=config["sun_flare"]["angle_lower"],
                angle_upper=config["sun_flare"]["angle_upper"],
                num_flare_circles_lower=config["sun_flare"]["num_flare_circles_lower"],
                num_flare_circles_upper=config["sun_flare"]["num_flare_circles_upper"],
                src_radius=config["sun_flare"]["src_radius"],
                src_color=config["sun_flare"]["src_color"],
                p=1.0
            ),
            RandomShadow(
                shadow_roi=config["shadow"]["shadow_roi"],
                num_shadows_lower=config["shadow"]["num_shadows_lower"],
                num_shadows_upper=config["shadow"]["num_shadows_upper"],
                shadow_dimension=config["shadow"]["shadow_dimension"],
                p=1.0
            ),
            RandomFog(
                fog_coef_lower=config["fog"]["fog_coef_lower"],
                fog_coef_upper=config["fog"]["fog_coef_upper"],
                alpha_coef=config["fog"]["alpha_coef"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(environmental_transforms, p=0.15)
        )
        
        # Color adjustments
        color_transforms = [
            HueSaturationValue(
                hue_shift_limit=config["hue_saturation"]["hue_shift_limit"],
                sat_shift_limit=config["hue_saturation"]["sat_shift_limit"],
                val_shift_limit=config["hue_saturation"]["val_shift_limit"],
                p=1.0
            ),
            RGBShift(
                r_shift_limit=config["rgb_shift"]["r_shift_limit"],
                g_shift_limit=config["rgb_shift"]["g_shift_limit"],
                b_shift_limit=config["rgb_shift"]["b_shift_limit"],
                p=1.0
            ),
            ColorJitter(
                brightness=config["color_jitter"]["brightness"],
                contrast=config["color_jitter"]["contrast"],
                saturation=config["color_jitter"]["saturation"],
                hue=config["color_jitter"]["hue"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(color_transforms, p=0.6)
        )
        
        # Image quality degradation
        quality_transforms = [
            ImageCompression(
                quality_lower=config["jpeg_compression"]["quality_lower"],
                quality_upper=config["jpeg_compression"]["quality_upper"],
                p=1.0
            ),
            Downscale(
                scale_min=config["downscale"]["scale_min"],
                scale_max=config["downscale"]["scale_max"],
                interpolation=config["downscale"]["interpolation"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(quality_transforms, p=0.5)
        )
        
        # Sharpening (sometimes internet images are over-sharpened)
        transforms.append(
            Sharpen(
                alpha=config["sharpen"]["alpha"],
                lightness=config["sharpen"]["lightness"],
                p=config["sharpen"]["probability"]
            )
        )
        
        # Distortions (occasional)
        distortion_transforms = [
            GridDistortion(
                num_steps=config["grid_distortion"]["num_steps"],
                distort_limit=config["grid_distortion"]["distort_limit"],
                p=1.0
            ),
            OpticalDistortion(
                distort_limit=config["optical_distortion"]["distort_limit"],
                shift_limit=config["optical_distortion"]["shift_limit"],
                p=1.0
            ),
            ElasticTransform(
                alpha=config["elastic_transform"]["alpha"],
                sigma=config["elastic_transform"]["sigma"],
                alpha_affine=config["elastic_transform"]["alpha_affine"],
                p=1.0
            )
        ]
        transforms.append(
            OneOf(distortion_transforms, p=0.1)
        )
        
        # Cutout/Dropout for regularization
        transforms.append(
            CoarseDropout(
                max_holes=config["coarse_dropout"]["max_holes"],
                max_height=config["coarse_dropout"]["max_height"],
                max_width=config["coarse_dropout"]["max_width"],
                min_holes=config["coarse_dropout"]["min_holes"],
                min_height=config["coarse_dropout"]["min_height"],
                min_width=config["coarse_dropout"]["min_width"],
                fill_value=config["coarse_dropout"]["fill_value"],
                p=config["coarse_dropout"]["probability"]
            )
        )
        
        # Advanced color effects (rare)
        advanced_color_transforms = [
            ChannelShuffle(p=1.0),
            RandomToneCurve(scale=config["random_tone_curve"]["scale"], p=1.0),
            Posterize(num_bits=config["posterize"]["num_bits"], p=1.0),
            Solarize(threshold=config["solarize"]["threshold"], p=1.0)
        ]
        transforms.append(
            OneOf(advanced_color_transforms, p=0.05)
        )
        
        # Normalization
        if normalize:
            transforms.append(ToFloat(max_value=255.0))
        
        return A.Compose(transforms)
    
    @staticmethod
    def create_validation_pipeline(
        image_size: int = 224,
        normalize: bool = True
    ) -> A.Compose:
        """
        Create validation augmentation pipeline with minimal transformations.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize images to [0, 1]
            
        Returns:
            Albumentations Compose object with validation augmentations
        """
        transforms = [
            CenterCrop(height=image_size, width=image_size, p=1.0)
        ]
        
        if normalize:
            transforms.append(ToFloat(max_value=255.0))
        
        return A.Compose(transforms)
    
    @staticmethod
    def create_test_time_augmentation_pipeline(
        image_size: int = 224,
        normalize: bool = True,
        num_augmentations: int = 5
    ) -> List[A.Compose]:
        """
        Create test-time augmentation (TTA) pipelines for improved predictions.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize images
            num_augmentations: Number of TTA variations
            
        Returns:
            List of augmentation pipelines for TTA
        """
        tta_pipelines = []
        
        # Original image
        tta_pipelines.append(
            AugmentationPipeline.create_validation_pipeline(image_size, normalize)
        )
        
        # Horizontal flip
        transforms = [
            CenterCrop(height=image_size, width=image_size, p=1.0),
            HorizontalFlip(p=1.0)
        ]
        if normalize:
            transforms.append(ToFloat(max_value=255.0))
        tta_pipelines.append(A.Compose(transforms))
        
        # Different crops
        for scale in [0.9, 0.95]:
            transforms = [
                RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(scale, scale),
                    p=1.0
                )
            ]
            if normalize:
                transforms.append(ToFloat(max_value=255.0))
            tta_pipelines.append(A.Compose(transforms))
        
        # Slight rotation
        transforms = [
            CenterCrop(height=image_size, width=image_size, p=1.0),
            Rotate(limit=10, p=1.0)
        ]
        if normalize:
            transforms.append(ToFloat(max_value=255.0))
        tta_pipelines.append(A.Compose(transforms))
        
        return tta_pipelines[:num_augmentations]
    
    @staticmethod
    def create_field_condition_pipeline(
        config: Optional[Dict[str, Any]] = None,
        image_size: int = 224,
        normalize: bool = True
    ) -> A.Compose:
        """
        Create pipeline specifically for field condition simulation.
        
        This pipeline focuses on conditions commonly found in agricultural
        field photography: outdoor lighting, weather conditions, and
        handheld camera artifacts.
        
        Args:
            config: Configuration dictionary
            image_size: Target image size
            normalize: Whether to normalize images
            
        Returns:
            Albumentations Compose object for field conditions
        """
        if config is None:
            config = AugmentationPipeline.get_default_config()
        
        transforms = [
            RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                p=1.0
            )
        ]
        
        # Strong lighting variations (outdoor conditions)
        transforms.append(
            RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.8
            )
        )
        
        # Sun and shadow effects (common in field)
        outdoor_transforms = [
            RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_lower=3,
                num_flare_circles_upper=6,
                src_radius=200,
                p=1.0
            ),
            RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=7,
                p=1.0
            )
        ]
        transforms.append(OneOf(outdoor_transforms, p=0.3))
        
        # Weather conditions
        weather_transforms = [
            RandomRain(drop_length=15, drop_width=1, blur_value=3, p=1.0),
            RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.1, p=1.0)
        ]
        transforms.append(OneOf(weather_transforms, p=0.1))
        
        # Handheld camera artifacts
        transforms.append(
            OneOf([
                MotionBlur(blur_limit=9, p=1.0),
                Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0)
            ], p=0.25)
        )
        
        # Mobile phone camera quality
        transforms.append(
            ImageCompression(
                quality_lower=70,
                quality_upper=95,
                compression_type=ImageCompression.ImageCompressionType.JPEG,
                p=0.7
            )
        )
        
        if normalize:
            transforms.append(ToFloat(max_value=255.0))
        
        return A.Compose(transforms)


def visualize_augmentations(
    image: np.ndarray,
    pipeline: A.Compose,
    num_samples: int = 9,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize multiple augmentation results from a pipeline.
    
    Args:
        image: Original image
        pipeline: Augmentation pipeline
        num_samples: Number of augmented samples to generate
        save_path: Optional path to save visualization
        
    Returns:
        Grid of augmented images
    """
    import matplotlib.pyplot as plt
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        augmented = pipeline(image=image)["image"]
        if augmented.dtype == np.float32 or augmented.dtype == np.float64:
            augmented = (augmented * 255).astype(np.uint8)
        axes[i].imshow(augmented)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    # Convert to numpy array for return
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return img_array


if __name__ == "__main__":
    # Example usage and testing
    print("Augmentation Pipeline for Realistic Internet Photo Conditions")
    print("=" * 60)
    
    # Create training pipeline with default config
    train_pipeline = AugmentationPipeline.create_training_pipeline()
    print(f"\nTraining pipeline created with {len(train_pipeline.transforms)} transform groups")
    
    # Create validation pipeline
    val_pipeline = AugmentationPipeline.create_validation_pipeline()
    print(f"Validation pipeline created with {len(val_pipeline.transforms)} transforms")
    
    # Create field condition pipeline
    field_pipeline = AugmentationPipeline.create_field_condition_pipeline()
    print(f"Field condition pipeline created with {len(field_pipeline.transforms)} transforms")
    
    # Create TTA pipelines
    tta_pipelines = AugmentationPipeline.create_test_time_augmentation_pipeline()
    print(f"Test-time augmentation created with {len(tta_pipelines)} variations")
    
    # Custom configuration example
    custom_config = AugmentationPipeline.get_default_config()
    custom_config["jpeg_compression"]["quality_lower"] = 50  # More aggressive compression
    custom_config["rain"]["probability"] = 0.1  # Higher rain probability
    custom_pipeline = AugmentationPipeline.create_training_pipeline(custom_config)
    print(f"\nCustom pipeline created with modified parameters")
    
    print("\nAugmentation Categories:")
    print("  1. Lighting Variations: brightness, contrast, gamma, CLAHE")
    print("  2. Camera Quality: Gaussian noise, ISO noise, multiplicative noise")
    print("  3. Focus Issues: motion blur, defocus, Gaussian blur")
    print("  4. Environmental: rain, sun flare, shadows, fog")
    print("  5. Compression: JPEG artifacts (quality 60-100)")
    print("  6. Color Effects: hue/saturation, RGB shift, color jitter")
    print("  7. Geometric: rotation, shift, scale, flips")
    print("  8. Distortions: grid, optical, elastic")
    print("  9. Regularization: coarse dropout, cutout")
    
    print("\nUsage Example:")
    print("  from augmentation_pipeline import AugmentationPipeline")
    print("  import cv2")
    print("")
    print("  # Load image")
    print("  image = cv2.imread('plant_image.jpg')")
    print("  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)")
    print("")
    print("  # Create pipeline")
    print("  pipeline = AugmentationPipeline.create_training_pipeline()")
    print("")
    print("  # Apply augmentation")
    print("  augmented = pipeline(image=image)['image']")