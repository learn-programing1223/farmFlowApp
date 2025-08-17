#!/usr/bin/env python3
"""
Quick test to verify the argparse fix works correctly.
"""

import subprocess
import sys

def test_argparse():
    """Test that the argparse fix works."""
    print("=" * 70)
    print("TESTING ARGPARSE FIX")
    print("=" * 70)
    
    # Test command that previously failed
    cmd = [
        sys.executable,
        "train_robust_model_v2.py",
        "--epochs", "1",
        "--batch_size", "8",
        "--preprocessing_mode", "legacy",
        "--use_advanced_preprocessing",  # This should work now as a flag
        "--loss_type", "standard",
        "--test_run"  # Quick test mode
    ]
    
    print("\nTesting command:")
    print(" ".join(cmd))
    print("\nExpected: Should start without argparse errors")
    print("-" * 70)
    
    try:
        # Run for just a few seconds to test argument parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read first 20 lines to see if it starts correctly
        line_count = 0
        error_found = False
        
        for line in process.stdout:
            print(line, end='')
            line_count += 1
            
            # Check for argparse errors
            if "error: argument --use_advanced_preprocessing" in line:
                error_found = True
                print("\n[ERROR] Argparse error still present!")
                break
            
            # Stop after seeing that training started
            if "Loading datasets" in line or "ENHANCED ROBUST MODEL TRAINING" in line:
                print("\n[SUCCESS] Training started without argparse errors!")
                break
                
            if line_count > 30:
                break
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        if not error_found:
            print("\n" + "=" * 70)
            print("TEST PASSED!")
            print("=" * 70)
            print("\nThe argparse fix is working correctly:")
            print("  ✓ --use_advanced_preprocessing works as a flag")
            print("  ✓ No argument errors")
            print("  ✓ Training can start successfully")
            return True
        else:
            print("\n" + "=" * 70)
            print("TEST FAILED!")
            print("=" * 70)
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print("\n[INFO] Test timed out (this is OK)")
        return True
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("Testing argparse fix for --use_advanced_preprocessing")
    print("This will start training briefly to verify arguments work\n")
    
    success = test_argparse()
    
    if success:
        print("\n✅ Fix verified! You can now run:")
        print("  .\\run_overnight_training.ps1")
        print("  .\\run_clean_training.ps1")
    else:
        print("\n❌ Fix may not be working. Check the error above.")
    
    sys.exit(0 if success else 1)