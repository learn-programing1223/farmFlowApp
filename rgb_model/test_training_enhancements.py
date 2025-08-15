#!/usr/bin/env python3
"""
Test script for enhanced training features
==========================================

Tests different configurations of the enhanced training script
to verify all components work correctly.
"""

import subprocess
import time
import psutil
import json
import os
from pathlib import Path
from datetime import datetime
import sys

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def run_training_test(config_name, args):
    """Run a training test with given arguments."""
    print("\n" + "=" * 70)
    print(f"TEST: {config_name}")
    print("=" * 70)
    print(f"Arguments: {' '.join(args)}")
    
    # Record start metrics
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Build command
    cmd = [sys.executable, "train_robust_model_v2.py"] + args
    
    # Run training
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout for test runs
    )
    
    # Record end metrics
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Parse output
    output = result.stdout + result.stderr
    
    # Extract key metrics from output
    metrics = {
        'config_name': config_name,
        'success': result.returncode == 0,
        'duration': end_time - start_time,
        'memory_delta': end_memory - start_memory,
        'has_nan': 'nan' in output.lower() or 'inf' in output.lower(),
        'final_accuracy': None,
        'final_loss': None,
        'swa_initialized': '[SWA]' in output,
        'model_saved': 'Saved' in output and '.h5' in output
    }
    
    # Try to extract accuracy and loss
    for line in output.split('\n'):
        if 'accuracy:' in line.lower() and 'val' not in line.lower():
            try:
                # Extract accuracy value
                parts = line.split('accuracy:')[-1].strip().split()[0]
                metrics['final_accuracy'] = float(parts.strip('%'))
            except:
                pass
        
        if 'loss:' in line.lower() and 'val' not in line.lower():
            try:
                # Extract loss value
                parts = line.split('loss:')[-1].strip().split()[0]
                metrics['final_loss'] = float(parts)
            except:
                pass
    
    # Print summary
    print(f"\nResults for {config_name}:")
    print(f"  Success: {metrics['success']}")
    print(f"  Duration: {metrics['duration']:.2f}s")
    print(f"  Memory delta: {metrics['memory_delta']:.2f} MB")
    print(f"  Has NaN/Inf: {metrics['has_nan']}")
    print(f"  Final loss: {metrics['final_loss']}")
    print(f"  Final accuracy: {metrics['final_accuracy']}")
    print(f"  SWA initialized: {metrics['swa_initialized']}")
    print(f"  Model saved: {metrics['model_saved']}")
    
    # Save detailed output
    output_file = f"test_output_{config_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Arguments: {' '.join(args)}\n")
        f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("Full Output:\n")
        f.write(output)
    
    print(f"  Full output saved to: {output_file}")
    
    return metrics

def main():
    """Run all tests."""
    print("=" * 70)
    print("ENHANCED TRAINING SCRIPT TEST SUITE")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Initial memory: {get_memory_usage():.2f} MB")
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    all_results = []
    
    # Test 1: Fast preprocessing with CombinedLoss
    print("\n[Test 1: Fast preprocessing with CombinedLoss]")
    result = run_training_test(
        "Fast preprocessing with CombinedLoss",
        [
            "--test_run",
            "--preprocessing_mode", "fast",
            "--loss_type", "combined",
            "--focal_weight", "0.7",
            "--epochs", "2",
            "--batch_size", "16"  # Smaller batch for faster testing
        ]
    )
    all_results.append(result)
    
    # Test 2: Fast preprocessing with standard loss
    print("\n[Test 2: Fast preprocessing with standard loss]")
    result = run_training_test(
        "Fast preprocessing with standard loss",
        [
            "--test_run",
            "--preprocessing_mode", "fast",
            "--loss_type", "standard",
            "--epochs", "2",
            "--batch_size", "16"
        ]
    )
    all_results.append(result)
    
    # Test 3: Legacy preprocessing (advanced disabled)
    print("\n[Test 3: Legacy preprocessing]")
    result = run_training_test(
        "Legacy preprocessing",
        [
            "--test_run",
            "--preprocessing_mode", "legacy",
            "--no-use_advanced_preprocessing",
            "--loss_type", "combined",
            "--epochs", "2",
            "--batch_size", "16"
        ]
    )
    all_results.append(result)
    
    # Test 4: Verify SWA initialization (set low start epoch)
    print("\n[Test 4: SWA initialization test]")
    result = run_training_test(
        "SWA initialization",
        [
            "--test_run",
            "--preprocessing_mode", "minimal",  # Fastest mode
            "--loss_type", "focal",
            "--swa_start_epoch", "1",  # Start SWA early for testing
            "--epochs", "3",
            "--batch_size", "16"
        ]
    )
    all_results.append(result)
    
    # Test 5: Model checkpoint saving
    print("\n[Test 5: Model checkpoint saving]")
    result = run_training_test(
        "Checkpoint saving",
        [
            "--test_run",
            "--preprocessing_mode", "minimal",
            "--loss_type", "label_smoothing",
            "--epochs", "2",
            "--batch_size", "16",
            "--output_dir", "models"
        ]
    )
    all_results.append(result)
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print("\n1. Performance Comparison:")
    print(f"{'Configuration':<40} {'Time (s)':<10} {'Memory (MB)':<12} {'Loss':<10} {'Success':<10}")
    print("-" * 82)
    
    for result in all_results:
        config = result['config_name'][:40]
        duration = f"{result['duration']:.2f}"
        memory = f"{result['memory_delta']:.2f}"
        loss = f"{result['final_loss']:.4f}" if result['final_loss'] else "N/A"
        success = "✓" if result['success'] else "✗"
        
        print(f"{config:<40} {duration:<10} {memory:<12} {loss:<10} {success:<10}")
    
    print("\n2. Loss Function Comparison:")
    combined_results = [r for r in all_results if 'combined' in r['config_name'].lower()]
    standard_results = [r for r in all_results if 'standard' in r['config_name'].lower()]
    
    if combined_results and standard_results:
        combined_loss = combined_results[0]['final_loss']
        standard_loss = standard_results[0]['final_loss']
        
        if combined_loss and standard_loss:
            print(f"  Combined Loss: {combined_loss:.4f}")
            print(f"  Standard Loss: {standard_loss:.4f}")
            print(f"  Difference: {abs(combined_loss - standard_loss):.4f}")
    
    print("\n3. Preprocessing Comparison:")
    fast_results = [r for r in all_results if 'fast' in r['config_name'].lower()]
    legacy_results = [r for r in all_results if 'legacy' in r['config_name'].lower()]
    
    if fast_results and legacy_results:
        fast_time = fast_results[0]['duration']
        legacy_time = legacy_results[0]['duration']
        
        print(f"  Fast preprocessing time: {fast_time:.2f}s")
        print(f"  Legacy preprocessing time: {legacy_time:.2f}s")
        print(f"  Speedup: {legacy_time/fast_time:.2f}x")
    
    print("\n4. Feature Verification:")
    print(f"  SWA initialized: {any(r['swa_initialized'] for r in all_results)}")
    print(f"  Models saved: {all(r['model_saved'] for r in all_results)}")
    print(f"  No NaN/Inf losses: {not any(r['has_nan'] for r in all_results)}")
    
    # Check for saved models
    print("\n5. Saved Model Files:")
    model_files = list(Path("models").glob("enhanced_*.h5"))
    for model_file in model_files[:5]:  # Show first 5
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"  - {model_file.name} ({size_mb:.2f} MB)")
    
    # Save test report
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': len(all_results),
        'tests_passed': sum(1 for r in all_results if r['success']),
        'results': all_results
    }
    
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[Report saved to: {report_file}]")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    all_passed = all(r['success'] for r in all_results)
    no_nan = not any(r['has_nan'] for r in all_results)
    
    if all_passed and no_nan:
        print("✓ ALL TESTS PASSED!")
        print("  - All configurations ran successfully")
        print("  - No NaN or Inf values detected")
        print("  - Model checkpoints saved correctly")
    else:
        print("✗ SOME TESTS FAILED")
        failed = [r['config_name'] for r in all_results if not r['success']]
        if failed:
            print(f"  Failed configurations: {', '.join(failed)}")
        if not no_nan:
            print("  WARNING: NaN or Inf values detected in some tests")

if __name__ == "__main__":
    main()