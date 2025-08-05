"""
TensorFlow Lite Converter for Plant Disease Detection Model
Handles conversion and optimization for mobile deployment
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, Union, Dict
import json

logger = logging.getLogger(__name__)


class TFLiteConverter:
    """
    Converts trained Keras models to TensorFlow Lite format with various optimization options
    """
    
    def __init__(self, model_path: str, output_dir: str = './tflite'):
        """
        Initialize the converter
        
        Args:
            model_path: Path to the saved Keras model
            output_dir: Directory to save TFLite models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the model
        logger.info(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        
    def convert_to_tflite(self, 
                         quantization_type: str = 'int8',
                         representative_data: Optional[np.ndarray] = None,
                         optimize_for_size: bool = True) -> str:
        """
        Convert model to TFLite format with specified optimizations
        
        Args:
            quantization_type: Type of quantization ('none', 'dynamic', 'int8', 'float16')
            representative_data: Representative dataset for full integer quantization
            optimize_for_size: Whether to optimize for model size
            
        Returns:
            Path to the saved TFLite model
        """
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        if optimize_for_size:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure quantization
        if quantization_type == 'dynamic':
            # Dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            output_file = 'model_dynamic_quant.tflite'
            
        elif quantization_type == 'int8':
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_data is not None:
                def representative_dataset():
                    for i in range(min(100, len(representative_data))):
                        # Ensure data is in the right format
                        data = np.expand_dims(representative_data[i], axis=0).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
                
                # Set input/output to int8
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                
            output_file = 'model_int8_quant.tflite'
            
        elif quantization_type == 'float16':
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            output_file = 'model_float16_quant.tflite'
            
        else:
            # No quantization
            output_file = 'model_float32.tflite'
        
        # Convert the model
        try:
            logger.info(f"Converting model with {quantization_type} quantization...")
            tflite_model = converter.convert()
            
            # Save the model
            output_path = self.output_dir / output_file
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Calculate size reduction
            original_size = self.model_path.stat().st_size / (1024 * 1024)  # MB
            tflite_size = len(tflite_model) / (1024 * 1024)  # MB
            reduction = (1 - tflite_size / original_size) * 100
            
            logger.info(f"Model converted successfully!")
            logger.info(f"Original size: {original_size:.2f} MB")
            logger.info(f"TFLite size: {tflite_size:.2f} MB")
            logger.info(f"Size reduction: {reduction:.1f}%")
            logger.info(f"Saved to: {output_path}")
            
            # Save conversion info
            info = {
                'original_model': str(self.model_path),
                'tflite_model': str(output_path),
                'quantization_type': quantization_type,
                'original_size_mb': original_size,
                'tflite_size_mb': tflite_size,
                'size_reduction_percent': reduction
            }
            
            with open(self.output_dir / 'conversion_info.json', 'w') as f:
                json.dump(info, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return None
    
    def evaluate_tflite_model(self, 
                            tflite_path: str,
                            test_data: np.ndarray,
                            test_labels: np.ndarray,
                            batch_size: int = 32) -> Dict:
        """
        Evaluate the TFLite model accuracy
        
        Args:
            tflite_path: Path to TFLite model
            test_data: Test images
            test_labels: Test labels (one-hot encoded)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get input shape and type
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        logger.info(f"Evaluating TFLite model: {tflite_path}")
        logger.info(f"Input shape: {input_shape}, dtype: {input_dtype}")
        
        # Prepare for evaluation
        predictions = []
        
        # Process in batches
        num_samples = len(test_data)
        for i in range(0, num_samples, batch_size):
            batch_data = test_data[i:i+batch_size]
            
            batch_predictions = []
            for sample in batch_data:
                # Prepare input
                input_data = np.expand_dims(sample, axis=0)
                
                # Quantize input if needed
                if input_dtype == np.uint8:
                    input_data = (input_data * 255).astype(np.uint8)
                else:
                    input_data = input_data.astype(input_dtype)
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                batch_predictions.append(output_data[0])
            
            predictions.extend(batch_predictions)
            
            # Progress update
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, num_samples)}/{num_samples} samples")
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Dequantize if needed
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.uint8:
            # Get quantization parameters
            output_quant = output_details[0].get('quantization', (0, 0))
            scale, zero_point = output_quant
            if scale != 0:
                predictions = (predictions.astype(np.float32) - zero_point) * scale
        
        # Apply softmax if not already applied
        if predictions.shape[-1] > 1 and not np.allclose(predictions.sum(axis=-1), 1.0):
            predictions = tf.nn.softmax(predictions).numpy()
        
        # Calculate metrics
        y_true = np.argmax(test_labels, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class accuracy
        num_classes = test_labels.shape[1]
        per_class_accuracy = []
        for c in range(num_classes):
            mask = y_true == c
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == c)
                per_class_accuracy.append(class_acc)
        
        results = {
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_accuracy,
            'mean_per_class_accuracy': float(np.mean(per_class_accuracy)),
            'num_test_samples': num_samples
        }
        
        logger.info(f"\nTFLite Model Evaluation:")
        logger.info(f"Overall Accuracy: {accuracy:.2%}")
        logger.info(f"Mean Per-Class Accuracy: {np.mean(per_class_accuracy):.2%}")
        
        return results
    
    def convert_all_variants(self, representative_data: Optional[np.ndarray] = None):
        """
        Convert model to multiple TFLite variants for comparison
        
        Args:
            representative_data: Representative dataset for quantization
        """
        variants = ['none', 'dynamic', 'float16']
        if representative_data is not None:
            variants.append('int8')
        
        results = {}
        
        for variant in variants:
            logger.info(f"\nConverting with {variant} quantization...")
            tflite_path = self.convert_to_tflite(
                quantization_type=variant,
                representative_data=representative_data
            )
            
            if tflite_path:
                results[variant] = tflite_path
        
        # Create comparison summary
        summary = []
        for variant, path in results.items():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            summary.append({
                'variant': variant,
                'path': path,
                'size_mb': size_mb
            })
        
        # Sort by size
        summary.sort(key=lambda x: x['size_mb'])
        
        logger.info("\n" + "="*50)
        logger.info("TFLITE CONVERSION SUMMARY")
        logger.info("="*50)
        for item in summary:
            logger.info(f"{item['variant']:>10}: {item['size_mb']:.2f} MB")
        
        return results


def main():
    """Test the converter"""
    import numpy as np
    
    # Create a dummy model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    # Save dummy model
    model.save('test_model.keras')
    
    # Create converter
    converter = TFLiteConverter('test_model.keras', './test_tflite')
    
    # Create dummy representative data
    representative_data = np.random.rand(100, 224, 224, 3).astype(np.float32)
    
    # Convert all variants
    converter.convert_all_variants(representative_data)
    
    # Clean up
    import os
    os.remove('test_model.keras')
    
    print("\nTFLite converter test completed!")


if __name__ == "__main__":
    main()