#!/usr/bin/env python3
"""
PlantVillage Data Quality Analysis Script
========================================

This script analyzes the processed PlantVillage dataset to:
1. Verify data quality and consistency
2. Generate distribution statistics
3. Sample and visualize images from each category
4. Check for potential data issues
5. Validate the mapping accuracy

Author: Claude Code
Date: 2025-01-11
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from collections import defaultdict, Counter
# import seaborn as sns  # Not needed for this analysis

class PlantVillageAnalyzer:
    def __init__(self, processed_path="datasets/plantvillage_processed"):
        self.processed_path = processed_path
        self.target_categories = ['Blight', 'Healthy', 'Leaf_Spot', 'Mosaic_Virus', 
                                'Nutrient_Deficiency', 'Powdery_Mildew', 'Rust']
        self.stats = {}
        
        # Load metadata
        metadata_path = os.path.join(processed_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def analyze_dataset_structure(self):
        """Analyze the structure and distribution of the processed dataset"""
        print("=" * 60)
        print("PLANTVILLAGE DATASET STRUCTURE ANALYSIS")
        print("=" * 60)
        
        splits = ['train', 'val', 'test']
        total_stats = defaultdict(int)
        split_stats = {}
        
        for split in splits:
            split_path = os.path.join(self.processed_path, split)
            if not os.path.exists(split_path):
                print(f"Warning: {split} split not found")
                continue
                
            split_data = {}
            print(f"\n{split.upper()} SPLIT:")
            print("-" * 20)
            
            for category in self.target_categories:
                cat_path = os.path.join(split_path, category)
                if os.path.exists(cat_path):
                    files = [f for f in os.listdir(cat_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    count = len(files)
                    split_data[category] = count
                    total_stats[category] += count
                    
                    print(f"  {category:18}: {count:4d} images")
                else:
                    split_data[category] = 0
                    print(f"  {category:18}: {0:4d} images")
            
            split_total = sum(split_data.values())
            split_data['total'] = split_total
            split_stats[split] = split_data
            print(f"  {'TOTAL':18}: {split_total:4d} images")
        
        print(f"\nOVERALL DISTRIBUTION:")
        print("-" * 25)
        grand_total = 0
        for category in self.target_categories:
            total = total_stats[category]
            grand_total += total
            print(f"  {category:18}: {total:4d} images")
        
        print(f"  {'GRAND TOTAL':18}: {grand_total:4d} images")
        
        self.stats['split_stats'] = split_stats
        self.stats['total_stats'] = total_stats
        self.stats['grand_total'] = grand_total
        
        return split_stats
    
    def check_data_balance(self):
        """Check class balance within and across splits"""
        print(f"\nCLASS BALANCE ANALYSIS:")
        print("-" * 25)
        
        # Calculate balance metrics
        counts = list(self.stats['total_stats'].values())
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = np.min(counts)
            max_count = np.max(counts)
            
            balance_ratio = min_count / max_count if max_count > 0 else 0
            cv = std_count / mean_count if mean_count > 0 else 0
            
            print(f"  Mean images per class: {mean_count:.1f}")
            print(f"  Standard deviation: {std_count:.1f}")
            print(f"  Coefficient of variation: {cv:.3f}")
            print(f"  Balance ratio (min/max): {balance_ratio:.3f}")
            
            if balance_ratio > 0.8:
                print("  [OK] Dataset is well balanced")
            elif balance_ratio > 0.5:
                print("  [WARNING] Dataset has moderate imbalance")
            else:
                print("  [ERROR] Dataset is significantly imbalanced")
    
    def analyze_image_properties(self, sample_size=50):
        """Analyze properties of sample images from each category"""
        print(f"\nIMAGE PROPERTIES ANALYSIS:")
        print("-" * 30)
        
        properties = defaultdict(list)
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.processed_path, split)
            if not os.path.exists(split_path):
                continue
                
            for category in self.target_categories:
                cat_path = os.path.join(split_path, category)
                if not os.path.exists(cat_path):
                    continue
                
                files = [f for f in os.listdir(cat_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not files:
                    continue
                
                # Sample random files
                sample_files = random.sample(files, min(sample_size, len(files)))
                
                for filename in sample_files:
                    img_path = os.path.join(cat_path, filename)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            mode = img.mode
                            
                            properties['category'].append(category)
                            properties['width'].append(width)
                            properties['height'].append(height)
                            properties['aspect_ratio'].append(width/height)
                            properties['mode'].append(mode)
                            properties['file_size'].append(os.path.getsize(img_path))
                            
                    except Exception as e:
                        print(f"    Error reading {img_path}: {e}")
        
        if properties['width']:
            print(f"  Analyzed {len(properties['width'])} images")
            print(f"  Width range: {min(properties['width'])} - {max(properties['width'])}")
            print(f"  Height range: {min(properties['height'])} - {max(properties['height'])}")
            print(f"  Average aspect ratio: {np.mean(properties['aspect_ratio']):.2f}")
            
            modes = Counter(properties['mode'])
            print(f"  Color modes: {dict(modes)}")
            
            avg_size_mb = np.mean(properties['file_size']) / (1024 * 1024)
            print(f"  Average file size: {avg_size_mb:.2f} MB")
        
        return properties
    
    def visualize_sample_images(self, samples_per_category=4):
        """Create a visualization showing sample images from each category"""
        print(f"\nCREATING SAMPLE VISUALIZATION...")
        
        fig, axes = plt.subplots(len(self.target_categories), samples_per_category, 
                               figsize=(16, 20))
        
        if len(self.target_categories) == 1:
            axes = axes.reshape(1, -1)
        
        for cat_idx, category in enumerate(self.target_categories):
            category_images = []
            
            # Collect images from all splits
            for split in ['train', 'val', 'test']:
                cat_path = os.path.join(self.processed_path, split, category)
                if os.path.exists(cat_path):
                    files = [f for f in os.listdir(cat_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    for filename in files[:samples_per_category]:
                        img_path = os.path.join(cat_path, filename)
                        category_images.append((img_path, filename))
            
            # Display sample images
            for img_idx in range(samples_per_category):
                ax = axes[cat_idx, img_idx]
                
                if img_idx < len(category_images):
                    img_path, filename = category_images[img_idx]
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(f"{category}\n{filename[:20]}...", fontsize=8)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error loading\n{filename}", 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No image", ha='center', va='center', 
                           transform=ax.transAxes)
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.processed_path, 'sample_images.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Sample visualization saved to: sample_images.png")
    
    def create_distribution_charts(self):
        """Create charts showing data distribution"""
        print(f"\nCREATING DISTRIBUTION CHARTS...")
        
        # Prepare data for plotting
        categories = []
        train_counts = []
        val_counts = []
        test_counts = []
        
        for category in self.target_categories:
            categories.append(category)
            train_counts.append(self.stats['split_stats'].get('train', {}).get(category, 0))
            val_counts.append(self.stats['split_stats'].get('val', {}).get(category, 0))
            test_counts.append(self.stats['split_stats'].get('test', {}).get(category, 0))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Stacked bar chart
        x = np.arange(len(categories))
        width = 0.6
        
        ax1.bar(x, train_counts, width, label='Train', color='lightblue')
        ax1.bar(x, val_counts, width, bottom=train_counts, label='Val', color='lightgreen')
        ax1.bar(x, test_counts, width, 
               bottom=np.array(train_counts) + np.array(val_counts), 
               label='Test', color='lightcoral')
        
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Dataset Distribution by Split')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pie chart of total distribution
        total_counts = [train_counts[i] + val_counts[i] + test_counts[i] 
                       for i in range(len(categories))]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax2.pie(total_counts, labels=categories, 
                                          autopct='%1.1f%%', colors=colors)
        ax2.set_title('Overall Category Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.processed_path, 'distribution_charts.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Distribution charts saved to: distribution_charts.png")
    
    def validate_mapping_accuracy(self):
        """Validate the accuracy of category mapping by checking filenames"""
        print(f"\nVALIDATING CATEGORY MAPPING:")
        print("-" * 32)
        
        mapping_validation = defaultdict(list)
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.processed_path, split)
            if not os.path.exists(split_path):
                continue
                
            for category in self.target_categories:
                cat_path = os.path.join(split_path, category)
                if not os.path.exists(cat_path):
                    continue
                
                files = [f for f in os.listdir(cat_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Extract original categories from filenames
                original_categories = set()
                for filename in files:
                    # Extract original category from filename pattern
                    parts = filename.split('___')
                    if len(parts) > 0:
                        original_cat = parts[0].replace('_', ' ')
                        original_categories.add(original_cat)
                
                mapping_validation[category].extend(original_categories)
        
        # Print mapping validation results
        for category in self.target_categories:
            if category in mapping_validation:
                original_cats = set(mapping_validation[category])
                print(f"  {category:18}: {len(original_cats)} original categories")
                for orig_cat in sorted(original_cats):
                    print(f"    - {orig_cat}")
            else:
                print(f"  {category:18}: No images found")
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print(f"\nGENERATING COMPREHENSIVE REPORT...")
        
        report_path = os.path.join(self.processed_path, 'data_quality_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("PlantVillage Dataset Quality Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {self.metadata.get('created_at', 'Unknown')}\n")
            f.write(f"Total images processed: {self.stats.get('grand_total', 0)}\n\n")
            
            f.write("Dataset Structure:\n")
            f.write("-" * 20 + "\n")
            
            for split in ['train', 'val', 'test']:
                if split in self.stats['split_stats']:
                    split_data = self.stats['split_stats'][split]
                    f.write(f"\n{split.upper()}:\n")
                    for category in self.target_categories:
                        count = split_data.get(category, 0)
                        f.write(f"  {category:18}: {count:4d} images\n")
                    f.write(f"  {'TOTAL':18}: {split_data.get('total', 0):4d} images\n")
            
            f.write(f"\nOverall Distribution:\n")
            f.write("-" * 20 + "\n")
            for category in self.target_categories:
                total = self.stats['total_stats'].get(category, 0)
                f.write(f"  {category:18}: {total:4d} images\n")
            f.write(f"  {'GRAND TOTAL':18}: {self.stats.get('grand_total', 0):4d} images\n")
            
            # Class balance analysis
            counts = list(self.stats['total_stats'].values())
            if counts:
                mean_count = np.mean(counts)
                std_count = np.std(counts)
                balance_ratio = np.min(counts) / np.max(counts) if np.max(counts) > 0 else 0
                
                f.write(f"\nClass Balance Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"  Mean images per class: {mean_count:.1f}\n")
                f.write(f"  Standard deviation: {std_count:.1f}\n")
                f.write(f"  Balance ratio (min/max): {balance_ratio:.3f}\n")
                
                if balance_ratio > 0.8:
                    f.write("  Status: Well balanced [OK]\n")
                elif balance_ratio > 0.5:
                    f.write("  Status: Moderately imbalanced [WARNING]\n")
                else:
                    f.write("  Status: Significantly imbalanced [ERROR]\n")
        
        print(f"  Comprehensive report saved to: data_quality_report.txt")
        print(f"  Report location: {os.path.abspath(report_path)}")
    
    def run_full_analysis(self):
        """Run complete data quality analysis"""
        print("Starting comprehensive PlantVillage data analysis...")
        
        # Set random seed for reproducible sampling
        random.seed(42)
        np.random.seed(42)
        
        # Run all analysis components
        self.analyze_dataset_structure()
        self.check_data_balance()
        self.analyze_image_properties()
        self.validate_mapping_accuracy()
        self.create_distribution_charts()
        self.visualize_sample_images()
        self.generate_report()
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Dataset is ready for training with {self.stats.get('grand_total', 0)} images")
        print(f"across {len(self.target_categories)} categories.")
        print(f"\nAll analysis files saved to: {os.path.abspath(self.processed_path)}")


def main():
    """Main function"""
    
    # Check if processed dataset exists
    processed_path = "datasets/plantvillage_processed"
    if not os.path.exists(processed_path):
        print(f"Error: Processed dataset not found at {processed_path}")
        print("Please run prepare_plantvillage_data.py first.")
        return
    
    try:
        # Create analyzer and run analysis
        analyzer = PlantVillageAnalyzer(processed_path)
        analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()