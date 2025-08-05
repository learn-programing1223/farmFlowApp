"""
Generate high-quality synthetic thermal dataset for immediate training
While ETH dataset downloads in background
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, Tuple
import random

def generate_synthetic_thermal_dataset(output_dir: str = "data/synthetic_thermal_advanced"):
    """Generate comprehensive synthetic thermal dataset"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("GENERATING SYNTHETIC THERMAL DATASET")
    print("=" * 60)
    print("This will create realistic thermal patterns for training")
    print("while the ETH dataset downloads in the background")
    print("=" * 60)
    
    # Define conditions with realistic parameters
    conditions = {
        "healthy": {
            "temp_mean": 22, 
            "temp_std": 1, 
            "count": 2000,
            "description": "Optimal plant conditions"
        },
        "drought_mild": {
            "temp_mean": 25, 
            "temp_std": 1.5, 
            "count": 1500,
            "description": "Early water stress"
        },
        "drought_severe": {
            "temp_mean": 28, 
            "temp_std": 2, 
            "count": 1500,
            "description": "Severe water stress"
        },
        "disease_bacterial": {
            "temp_mean": 24, 
            "temp_std": 3, 
            "count": 1500,
            "description": "Bacterial infection patterns"
        },
        "disease_fungal": {
            "temp_mean": 23, 
            "temp_std": 2.5, 
            "count": 1500,
            "description": "Fungal infection patterns"
        },
        "nutrient_deficient": {
            "temp_mean": 21, 
            "temp_std": 1.5, 
            "count": 1000,
            "description": "Nutrient stress"
        },
        "pest_damage": {
            "temp_mean": 26, 
            "temp_std": 4, 
            "count": 1000,
            "description": "Insect damage"
        }
    }
    
    total_generated = 0
    
    for condition, params in conditions.items():
        condition_dir = output_path / condition
        condition_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating {params['count']} {condition} images...")
        print(f"  {params['description']}")
        
        for i in range(params['count']):
            # Create thermal image
            img = create_realistic_thermal_image(
                condition=condition,
                temp_mean=params['temp_mean'],
                temp_std=params['temp_std']
            )
            
            # Save image
            filename = condition_dir / f"thermal_{i:05d}.png"
            cv2.imwrite(str(filename), img)
            
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{params['count']}")
            
            total_generated += 1
    
    # Save dataset info
    info = {
        "name": "Synthetic Thermal Advanced",
        "total_images": total_generated,
        "conditions": conditions,
        "resolution": "256x256",
        "temperature_range": [10, 40],
        "features": [
            "Realistic leaf structures",
            "Temperature gradients",
            "Disease patterns",
            "Sensor noise",
            "Environmental effects"
        ]
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Generated {total_generated:,} synthetic thermal images")
    print(f"📁 Saved to: {output_path}")
    
    return output_path

def create_realistic_thermal_image(condition: str, temp_mean: float, temp_std: float) -> np.ndarray:
    """Create a single realistic thermal image"""
    
    # Base temperature map
    img = np.random.normal(temp_mean, temp_std, (256, 256))
    
    # Add environmental gradient
    gradient_type = random.choice(['horizontal', 'vertical', 'radial'])
    if gradient_type == 'horizontal':
        gradient = np.linspace(0, 2, 256).reshape(1, -1)
        img += gradient
    elif gradient_type == 'vertical':
        gradient = np.linspace(0, 2, 256).reshape(-1, 1)
        img += gradient
    else:  # radial
        y, x = np.ogrid[:256, :256]
        center = (128, 128)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        gradient = (r / r.max()) * 2
        img += gradient
    
    # Create plant structure
    num_leaves = random.randint(3, 8)
    plant_mask = np.zeros((256, 256), dtype=bool)
    
    for _ in range(num_leaves):
        # Elliptical leaves
        center_x = random.randint(60, 196)
        center_y = random.randint(60, 196)
        width = random.randint(30, 80)
        height = random.randint(20, 60)
        angle = random.uniform(0, 180)
        
        leaf_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(leaf_mask, (center_x, center_y), (width//2, height//2), 
                   angle, 0, 360, 255, -1)
        
        plant_mask |= (leaf_mask > 0)
        
        # Leaves are cooler than background
        temp_diff = random.uniform(-4, -2)
        img[leaf_mask > 0] += temp_diff
    
    # Add condition-specific patterns
    if "drought" in condition:
        # Water stress: elevated leaf temperature
        stress_level = 0.3 if "mild" in condition else 0.7
        img[plant_mask] += stress_level * 6
        
        # Add hot spots
        num_spots = random.randint(3, 8)
        for _ in range(num_spots):
            if np.any(plant_mask):
                y_coords, x_coords = np.where(plant_mask)
                idx = random.randint(0, len(y_coords) - 1)
                center = (x_coords[idx], y_coords[idx])
                radius = random.randint(10, 25)
                intensity = random.uniform(2, 5)
                cv2.circle(img, center, radius, img[center[1], center[0]] + intensity, -1)
    
    elif "disease" in condition:
        if "bacterial" in condition:
            # Linear infection patterns
            num_lines = random.randint(5, 10)
            for _ in range(num_lines):
                if np.any(plant_mask):
                    y_coords, x_coords = np.where(plant_mask)
                    idx1 = random.randint(0, len(y_coords) - 1)
                    idx2 = random.randint(0, len(y_coords) - 1)
                    pt1 = (x_coords[idx1], y_coords[idx1])
                    pt2 = (x_coords[idx2], y_coords[idx2])
                    thickness = random.randint(2, 5)
                    temp_change = random.uniform(-3, 3)
                    cv2.line(img, pt1, pt2, temp_mean + temp_change, thickness)
        
        elif "fungal" in condition:
            # Circular infection zones
            num_zones = random.randint(8, 15)
            for _ in range(num_zones):
                if np.any(plant_mask):
                    y_coords, x_coords = np.where(plant_mask)
                    idx = random.randint(0, len(y_coords) - 1)
                    center = (x_coords[idx], y_coords[idx])
                    radius = random.randint(8, 20)
                    temp_change = random.uniform(-2, 4)
                    cv2.circle(img, center, radius, temp_mean + temp_change, -1)
    
    elif "nutrient" in condition:
        # Overall temperature reduction
        img[plant_mask] -= random.uniform(2, 4)
        
        # Irregular patches
        num_patches = random.randint(5, 10)
        for _ in range(num_patches):
            if np.any(plant_mask):
                y_coords, x_coords = np.where(plant_mask)
                idx = random.randint(0, len(y_coords) - 1)
                center_x, center_y = x_coords[idx], y_coords[idx]
                w, h = random.randint(20, 40), random.randint(20, 40)
                x1 = max(0, center_x - w//2)
                y1 = max(0, center_y - h//2)
                x2 = min(256, x1 + w)
                y2 = min(256, y1 + h)
                patch_temp = random.uniform(-2, 2)
                img[y1:y2, x1:x2] += patch_temp
    
    elif "pest" in condition:
        # Irregular damage patterns
        num_damages = random.randint(10, 20)
        for _ in range(num_damages):
            if np.any(plant_mask):
                y_coords, x_coords = np.where(plant_mask)
                # Create small irregular damage areas
                num_points = random.randint(3, 6)
                points = []
                base_idx = random.randint(0, len(y_coords) - 1)
                base_x, base_y = x_coords[base_idx], y_coords[base_idx]
                
                for _ in range(num_points):
                    offset_x = random.randint(-20, 20)
                    offset_y = random.randint(-20, 20)
                    px = np.clip(base_x + offset_x, 0, 255)
                    py = np.clip(base_y + offset_y, 0, 255)
                    points.append([px, py])
                
                points = np.array(points)
                temp_change = random.uniform(-5, 5)
                cv2.fillPoly(img, [points], temp_mean + temp_change)
    
    # Add realistic sensor effects
    # 1. Sensor noise
    noise = np.random.normal(0, 0.3, img.shape)
    img += noise
    
    # 2. Dead pixels (occasional)
    if random.random() < 0.1:
        num_dead = random.randint(1, 5)
        for _ in range(num_dead):
            x, y = random.randint(0, 255), random.randint(0, 255)
            img[y, x] = 0
    
    # 3. Edge vignetting
    y, x = np.ogrid[:256, :256]
    center = (128, 128)
    mask = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    vignette = 1 - (mask / mask.max()) * 0.2
    img *= vignette
    
    # 4. Slight blur (thermal camera characteristic)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Normalize to valid temperature range and convert to 8-bit
    img = np.clip(img, 10, 40)
    img_normalized = ((img - 10) / 30 * 255).astype(np.uint8)
    
    return img_normalized

def create_quick_test_set(output_dir: str = "data/quick_thermal_test"):
    """Create small test set for immediate validation"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating quick test set (100 images per condition)...")
    
    conditions = ["healthy", "drought_severe", "disease_fungal", "nutrient_deficient"]
    
    for condition in conditions:
        condition_dir = output_path / condition
        condition_dir.mkdir(exist_ok=True)
        
        temp_params = {
            "healthy": (22, 1),
            "drought_severe": (28, 2),
            "disease_fungal": (23, 2.5),
            "nutrient_deficient": (21, 1.5)
        }
        
        temp_mean, temp_std = temp_params.get(condition, (23, 2))
        
        for i in range(100):
            img = create_realistic_thermal_image(condition, temp_mean, temp_std)
            cv2.imwrite(str(condition_dir / f"test_{i:03d}.png"), img)
    
    print("✅ Quick test set ready!")
    return output_path

if __name__ == "__main__":
    import sys
    
    print("SYNTHETIC THERMAL DATASET GENERATOR")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Default to generating both for maximum data
        choice = "3"
        print("Generating both full dataset and test set...")
    
    if choice in ["1", "3"]:
        generate_synthetic_thermal_dataset()
    
    if choice in ["2", "3"]:
        create_quick_test_set()
    
    print("\n✅ Dataset generation complete!")
    print("\nYou can now train immediately with:")
    print("  python train_with_real_thermal.py")
    print("\nWhile the ETH dataset downloads, use the synthetic data!")