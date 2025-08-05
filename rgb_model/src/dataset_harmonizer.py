import os
import json
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path

class PlantDiseaseHarmonizer:
    """
    Harmonizes plant disease datasets by mapping crop-specific diseases 
    to universal disease categories for cross-crop generalization.
    """
    
    def __init__(self):
        self.universal_categories = [
            'Healthy',
            'Blight',
            'Leaf_Spot', 
            'Powdery_Mildew',
            'Rust',
            'Mosaic_Virus',
            'Nutrient_Deficiency'
        ]
        
        # Initialize comprehensive mapping rules
        self.mapping_rules = self._initialize_mapping_rules()
        
    def _initialize_mapping_rules(self) -> Dict[str, str]:
        """
        Creates comprehensive mapping from specific diseases to universal categories.
        Based on disease characteristics and visual symptoms.
        """
        return {
            # PlantVillage mappings (with triple underscores)
            "apple___apple_scab": "Leaf_Spot",
            "apple___black_rot": "Leaf_Spot",
            "apple___cedar_apple_rust": "Rust",
            "apple___healthy": "Healthy",
            
            "blueberry___healthy": "Healthy",
            
            "cherry_(including_sour)___powdery_mildew": "Powdery_Mildew",
            "cherry_(including_sour)___healthy": "Healthy",
            
            "corn_(maize)___cercospora_leaf_spot gray_leaf_spot": "Leaf_Spot",
            "corn_(maize)___common_rust_": "Rust",
            "corn_(maize)___northern_leaf_blight": "Blight",
            "corn_(maize)___healthy": "Healthy",
            
            "grape___black_rot": "Leaf_Spot",
            "grape___esca_(black_measles)": "Leaf_Spot",
            "grape___leaf_blight_(isariopsis_leaf_spot)": "Blight",
            "grape___healthy": "Healthy",
            
            "orange___haunglongbing_(citrus_greening)": "Nutrient_Deficiency",
            
            "peach___bacterial_spot": "Leaf_Spot",
            "peach___healthy": "Healthy",
            
            "pepper,_bell___bacterial_spot": "Leaf_Spot",
            "pepper,_bell___healthy": "Healthy",
            
            "potato___early_blight": "Blight",
            "potato___late_blight": "Blight",
            "potato___healthy": "Healthy",
            
            "raspberry___healthy": "Healthy",
            
            "soybean___healthy": "Healthy",
            
            "squash___powdery_mildew": "Powdery_Mildew",
            
            "strawberry___leaf_scorch": "Leaf_Spot",
            "strawberry___healthy": "Healthy",
            
            "tomato___bacterial_spot": "Leaf_Spot",
            "tomato___early_blight": "Blight",
            "tomato___late_blight": "Blight",
            "tomato___leaf_mold": "Leaf_Spot",
            "tomato___septoria_leaf_spot": "Leaf_Spot",
            "tomato___spider_mites two-spotted_spider_mite": "Leaf_Spot",  # Previously Pest_Damage
            "tomato___target_spot": "Leaf_Spot",
            "tomato___tomato_yellow_leaf_curl_virus": "Mosaic_Virus",
            "tomato___tomato_mosaic_virus": "Mosaic_Virus",
            "tomato___healthy": "Healthy",
            
            # Additional PlantVillage mappings (variations)
            "apple___apple_scab": "Leaf_Spot",
            "apple___black_rot": "Leaf_Spot",
            "apple___cedar_apple_rust": "Rust",
            "apple___healthy": "Healthy",
            "corn_(maize)___cercospora_leaf_spot_gray_leaf_spot": "Leaf_Spot",
            "corn_(maize)___common_rust": "Rust",
            "pepper_bell___bacterial_spot": "Leaf_Spot",
            "pepper_bell___healthy": "Healthy",
            
            # Map pest damage to Leaf_Spot
            "tomato___spider_mites_two-spotted_spider_mite": "Leaf_Spot",  # Previously Pest_Damage
            
            # General mappings (for other datasets)
            "tomato_early_blight": "Blight",
            "tomato_late_blight": "Blight",
            "potato_early_blight": "Blight",
            "potato_late_blight": "Blight",
            
            "tomato_bacterial_spot": "Leaf_Spot",
            "tomato_septoria_leaf_spot": "Leaf_Spot",
            "tomato_target_spot": "Leaf_Spot",
            "apple_black_rot": "Leaf_Spot",
            "pepper_bacterial_spot": "Leaf_Spot",
            "strawberry_leaf_scorch": "Leaf_Spot",
            
            "grape_black_measles": "Leaf_Spot",
            "apple_cedar_rust": "Rust",
            "corn_common_rust": "Rust",
            
            "tomato_mosaic_virus": "Mosaic_Virus",
            "tomato_yellow_leaf_curl_virus": "Mosaic_Virus",
            "pepper_bell_bacterial_spot": "Leaf_Spot",
            
            "squash_powdery_mildew": "Powdery_Mildew",
            "grape_leaf_blight": "Blight",
            
            "tomato_spider_mites": "Leaf_Spot",  # Previously Pest_Damage
            "corn_northern_leaf_blight": "Blight",
            
            # PlantDoc mappings
            "apple_scab": "Leaf_Spot",
            "apple_frogeye_spot": "Leaf_Spot",
            "cherry_powdery_mildew": "Powdery_Mildew",
            "peach_bacterial_spot": "Leaf_Spot",
            
            # More PlantDoc mappings
            "apple scab": "Leaf_Spot",
            "apple rust": "Rust",
            "apple healthy": "Healthy",
            "apple_leaf": "Healthy",  # Generic apple leaf
            "bell_pepper leaf spot": "Leaf_Spot",
            "bell_pepper healthy": "Healthy",
            "bell_pepper_leaf": "Healthy",
            "blueberry healthy": "Healthy",
            "blueberry_leaf": "Healthy",
            "cherry healthy": "Healthy",
            "cherry_leaf": "Healthy",
            "corn gray leaf spot": "Leaf_Spot",
            "corn common rust": "Rust",
            "corn healthy": "Healthy",
            "corn_leaf": "Healthy",
            "corn northern leaf blight": "Blight",
            "grape black measles": "Leaf_Spot",
            "grape black rot": "Leaf_Spot",
            "grape healthy": "Healthy",
            "grape_leaf": "Healthy",
            "grape leaf blight": "Blight",
            "peach healthy": "Healthy",
            "peach_leaf": "Healthy",
            "potato early blight": "Blight",
            "potato late blight": "Blight",
            "potato healthy": "Healthy",
            "potato_leaf": "Healthy",
            "raspberry healthy": "Healthy",
            "raspberry_leaf": "Healthy",
            "soybean bacterial blight": "Blight",
            "soybean caterpillar": "Leaf_Spot",  # Previously Pest_Damage
            "soybean diabrotica speciosa": "Leaf_Spot",  # Previously Pest_Damage
            "soybean downy mildew": "Powdery_Mildew",
            "soybean healthy": "Healthy",
            "soybean_leaf": "Healthy",
            "soybean mosaic virus": "Mosaic_Virus",
            "soybean powdery mildew": "Powdery_Mildew",
            "soybean rust": "Rust",
            "soybean southern blight": "Blight",
            "squash powdery mildew": "Powdery_Mildew",
            "squash healthy": "Healthy",
            "strawberry healthy": "Healthy",
            "strawberry_leaf": "Healthy",
            "strawberry leaf scorch": "Leaf_Spot",
            "tomato early blight": "Blight",
            "tomato healthy": "Healthy",
            "tomato_leaf": "Healthy",
            "tomato late blight": "Blight",
            "tomato leaf mold": "Leaf_Spot",
            "tomato mosaic virus": "Mosaic_Virus",
            "tomato septoria leaf spot": "Leaf_Spot",
            "tomato spider mites": "Leaf_Spot",  # Previously Pest_Damage
            "tomato target spot": "Leaf_Spot",
            "tomato yellow leaf curl virus": "Mosaic_Virus",
            
            # PlantDoc specific labels (based on common patterns)
            "alstonia_scholaris_diseased": "Leaf_Spot",
            "alstonia_scholaris_healthy": "Healthy",
            "arjun_diseased": "Leaf_Spot",
            "arjun_healthy": "Healthy",
            "bael_diseased": "Leaf_Spot",
            "basil_healthy": "Healthy",
            "chinar_diseased": "Leaf_Spot",
            "chinar_healthy": "Healthy",
            "gauva_diseased": "Leaf_Spot",
            "gauva_healthy": "Healthy",
            "jamun_diseased": "Leaf_Spot",
            "jamun_healthy": "Healthy",
            "jatropha_diseased": "Leaf_Spot",
            "jatropha_healthy": "Healthy",
            "lemon_diseased": "Leaf_Spot",
            "lemon_healthy": "Healthy",
            "mango_diseased": "Leaf_Spot",
            "mango_healthy": "Healthy",
            "pomegranate_diseased": "Leaf_Spot",
            "pomegranate_healthy": "Healthy",
            
            # General mappings
            "healthy": "Healthy",
            "leaf_mold": "Leaf_Spot",
            "septoria_leaf_spot": "Leaf_Spot",
            "spider_mites": "Leaf_Spot",  # Previously Pest_Damage
            "two_spotted_spider_mite": "Leaf_Spot",  # Previously Pest_Damage
            "yellow_leaf_curl_virus": "Mosaic_Virus",
            "mosaic_virus": "Mosaic_Virus",
            "bacterial_spot": "Leaf_Spot",
            "black_rot": "Leaf_Spot",
            "powdery_mildew": "Powdery_Mildew",
            "leaf_scorch": "Leaf_Spot",
            "rust": "Rust",
            "common_rust": "Rust",
            "northern_leaf_blight": "Blight",
            "cercospora_leaf_spot": "Leaf_Spot",
            "early_blight": "Blight",
            "late_blight": "Blight",
            "leaf_miner": "Leaf_Spot",  # Previously Pest_Damage
            "nutrient_deficiency": "Nutrient_Deficiency",
            "nitrogen_deficiency": "Nutrient_Deficiency",
            "phosphorus_deficiency": "Nutrient_Deficiency",
            "potassium_deficiency": "Nutrient_Deficiency",
        }
    
    def harmonize_dataset(self, dataset_name: str, images: List[str], 
                         labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Harmonizes a dataset by mapping specific disease labels to universal categories.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'PlantVillage', 'PlantDoc')
            images: List of image paths
            labels: List of original labels
            
        Returns:
            Tuple of (images, harmonized_labels)
        """
        harmonized_labels = []
        
        for label in labels:
            # Ensure label is a string
            label_str = str(label) if not isinstance(label, str) else label
            
            # Normalize label format - convert to lowercase and replace spaces/dashes
            normalized_label = label_str.lower().replace(' ', '_').replace('-', '_')
            
            # Try direct mapping first
            if normalized_label in self.mapping_rules:
                harmonized_label = self.mapping_rules[normalized_label]
            else:
                # Try to find partial matches
                harmonized_label = self._fuzzy_match_label(normalized_label)
                
            harmonized_labels.append(harmonized_label)
        
        # Log mapping statistics
        self._log_mapping_stats(dataset_name, labels, harmonized_labels)
        
        return images, harmonized_labels
    
    def _fuzzy_match_label(self, label: str) -> str:
        """
        Attempts to match a label using keyword matching when exact match fails.
        """
        label_lower = label.lower()
        
        # Check for keywords in label
        if 'healthy' in label_lower:
            return 'Healthy'
        elif 'blight' in label_lower:
            return 'Blight'
        elif 'spot' in label_lower or 'rot' in label_lower:
            return 'Leaf_Spot'
        elif 'rust' in label_lower:
            return 'Rust'
        elif 'mildew' in label_lower:
            return 'Powdery_Mildew'
        elif 'virus' in label_lower or 'mosaic' in label_lower or 'curl' in label_lower:
            return 'Mosaic_Virus'
        elif 'deficiency' in label_lower or 'nutrient' in label_lower:
            return 'Nutrient_Deficiency'
        elif 'pest' in label_lower or 'mite' in label_lower or 'insect' in label_lower or 'caterpillar' in label_lower:
            # Map pest damage to leaf spot since we don't have pest damage examples
            return 'Leaf_Spot'
        elif 'diseased' in label_lower:
            # Generic diseased - map to most common disease category
            return 'Leaf_Spot'
        elif label_lower.endswith('_leaf') or label_lower.endswith(' leaf'):
            # Generic plant leaf without disease indication
            return 'Healthy'
        else:
            print(f"Warning: Could not map label '{label}' to universal category")
            # Default to Healthy for unmapped labels rather than Unknown
            return 'Healthy'
    
    def _log_mapping_stats(self, dataset_name: str, original_labels: List[str], 
                          harmonized_labels: List[str]):
        """Logs statistics about the harmonization process."""
        unique_original = set(original_labels)
        unique_harmonized = set(harmonized_labels)
        
        print(f"\n{dataset_name} Harmonization Statistics:")
        print(f"Original unique labels: {len(unique_original)}")
        print(f"Harmonized unique labels: {len(unique_harmonized)}")
        
        if len(harmonized_labels) > 0:
            print(f"Distribution:")
            for category in self.universal_categories:
                count = harmonized_labels.count(category)
                percentage = (count / len(harmonized_labels)) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
            
            # Check for any "Unknown" labels
            unknown_count = harmonized_labels.count('Unknown')
            if unknown_count > 0:
                print(f"\nWarning: {unknown_count} labels could not be mapped!")
                # Show examples of unmapped labels
                unmapped_examples = []
                for orig, harm in zip(original_labels, harmonized_labels):
                    if harm == 'Unknown' and orig not in unmapped_examples:
                        unmapped_examples.append(orig)
                        if len(unmapped_examples) >= 5:
                            break
                print(f"Examples of unmapped labels: {unmapped_examples}")
        else:
            print("No labels to process.")
    
    def create_balanced_dataset(self, datasets: Dict[str, Tuple[List[str], List[str]]],
                               samples_per_class: int = 500) -> Tuple[List[str], List[str]]:
        """
        Creates a balanced dataset by sampling equally from each universal category.
        
        Args:
            datasets: Dictionary mapping dataset names to (images, labels) tuples
            samples_per_class: Number of samples per universal category
            
        Returns:
            Tuple of balanced (images, labels)
        """
        category_samples = {category: [] for category in self.universal_categories}
        
        # Collect all samples by category
        for dataset_name, (images, labels) in datasets.items():
            for img, label in zip(images, labels):
                if label in category_samples:
                    category_samples[label].append((img, label, dataset_name))
        
        # Balance sampling
        balanced_images = []
        balanced_labels = []
        
        for category, samples in category_samples.items():
            if len(samples) >= samples_per_class:
                # Random sample if we have enough
                selected = np.random.choice(len(samples), samples_per_class, replace=False)
                for idx in selected:
                    balanced_images.append(samples[idx][0])
                    balanced_labels.append(samples[idx][1])
            else:
                # Use all samples and oversample if needed
                for sample in samples:
                    balanced_images.append(sample[0])
                    balanced_labels.append(sample[1])
                
                # Oversample to reach target
                if len(samples) > 0:
                    oversample_count = samples_per_class - len(samples)
                    oversampled = np.random.choice(len(samples), oversample_count, replace=True)
                    for idx in oversampled:
                        balanced_images.append(samples[idx][0])
                        balanced_labels.append(samples[idx][1])
        
        return balanced_images, balanced_labels
    
    def save_mapping_config(self, output_path: str):
        """Saves the mapping configuration for reproducibility."""
        config = {
            'universal_categories': self.universal_categories,
            'mapping_rules': self.mapping_rules
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Mapping configuration saved to {output_path}")


def test_harmonizer():
    """Test the harmonizer with sample data."""
    harmonizer = PlantDiseaseHarmonizer()
    
    # Test with sample PlantVillage labels
    test_labels = [
        "Tomato_Early_blight",
        "Tomato_healthy",
        "Apple_Black_rot",
        "Corn_Common_rust",
        "Grape_Black_Measles"
    ]
    
    test_images = [f"img_{i}.jpg" for i in range(len(test_labels))]
    
    images, harmonized = harmonizer.harmonize_dataset(
        "PlantVillage_Test", 
        test_images, 
        test_labels
    )
    
    print("\nTest Results:")
    for original, harmonized_label in zip(test_labels, harmonized):
        print(f"{original} -> {harmonized_label}")


if __name__ == "__main__":
    test_harmonizer()