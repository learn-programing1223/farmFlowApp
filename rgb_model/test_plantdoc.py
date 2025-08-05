#!/usr/bin/env python3
"""
Test PlantDoc dataset loading
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import MultiDatasetLoader

def test_plantdoc_loading():
    """Test loading PlantDoc dataset"""
    print("Testing PlantDoc dataset loading...")
    
    # Initialize loader
    loader = MultiDatasetLoader(base_data_dir='./data')
    
    # Load PlantDoc only
    images, labels = loader.load_plantdoc()
    
    if images:
        print(f"\nSuccessfully loaded {len(images)} images")
        print(f"Unique labels after harmonization: {len(set(labels))}")
        
        # Count distribution
        from collections import Counter
        label_counts = Counter(labels)
        print("\nLabel distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")
    else:
        print("No images loaded from PlantDoc")

if __name__ == "__main__":
    test_plantdoc_loading()