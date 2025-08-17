#!/usr/bin/env python3
"""
Clean training launcher that suppresses non-critical warnings and provides cleaner output.
This wrapper handles all the warning suppressions and environment setup.
"""

import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors, not warnings/info
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations messages

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')

# Suppress other common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Now import TensorFlow with warnings suppressed
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import and run the main training script
print("=" * 70)
print("PLANT DISEASE DETECTION - CLEAN TRAINING")
print("=" * 70)
print()

# Check TensorFlow version
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    # Set memory growth to avoid OOM errors
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
else:
    print("No GPU detected - using CPU (will be slower)")

print()
print("-" * 70)
print("Starting training with clean output...")
print("-" * 70)
print()

# Import the training module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Pass through command line arguments to the training script
if __name__ == "__main__":
    # Import here to ensure all suppressions are active
    import train_robust_model_v2
    
    # The training script will handle its own argument parsing
    # Just need to ensure it runs
    try:
        train_robust_model_v2.main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)