#!/usr/bin/env python3
"""
Augment plant disease dataset using CycleGAN-style transformations
"""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from cyclegan_augmentor import augment_dataset_with_cyclegan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Augment plant disease dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Base data directory')
    parser.add_argument('--target-samples', type=int, default=7000,
                       help='Target samples per class')
    parser.add_argument('--output-dir', type=str, default='./data/augmented',
                       help='Output directory for augmented images')
    
    args = parser.parse_args()
    
    print("Starting data augmentation with CycleGAN-style transformations...")
    print(f"Target: {args.target_samples} samples per class")
    
    augmented_dir = augment_dataset_with_cyclegan(
        data_dir=args.data_dir,
        target_samples_per_class=args.target_samples,
        output_dir=args.output_dir
    )
    
    print(f"\nAugmentation complete! Synthetic images saved to: {augmented_dir}")
    print("\nYou can now train with augmented data by adding --use-augmented flag to training script")


if __name__ == "__main__":
    main()