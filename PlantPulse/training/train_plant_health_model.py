"""
PlantPulse Model Training Pipeline
Trains a MobileNetV3-based model for plant health analysis from thermal images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Skip tensorflow_datasets if not available
try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None
from typing import Tuple, Dict
import json
import cv2

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Temperature ranges for thermal images
TEMP_MIN = 15.0  # Celsius
TEMP_MAX = 40.0  # Celsius

class ThermalDataGenerator:
    """Generate synthetic thermal data for training"""
    
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        
    def generate_healthy_plant(self) -> np.ndarray:
        """Generate thermal pattern for healthy plant"""
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Base temperature (cooler than ambient)
        ambient_temp = np.random.uniform(25, 30)
        leaf_temp = ambient_temp - np.random.uniform(4, 8)
        
        # Create leaf shape with gaussian
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        for y in range(IMG_SIZE):
            for x in range(IMG_SIZE):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < IMG_SIZE // 3:
                    # Leaf area with slight variation
                    img[y, x] = leaf_temp + np.random.normal(0, 0.5)
                else:
                    # Background
                    img[y, x] = ambient_temp + np.random.normal(0, 0.3)
                    
        return img
    
    def generate_water_stressed_plant(self, stress_level: float) -> np.ndarray:
        """Generate thermal pattern for water-stressed plant"""
        img = self.generate_healthy_plant()
        
        # Increase temperature based on stress level
        temp_increase = stress_level * 6.0  # Up to 6°C increase
        leaf_mask = img < np.mean(img) - 2
        img[leaf_mask] += temp_increase
        
        return img
    
    def generate_diseased_plant(self, disease_type: str) -> np.ndarray:
        """Generate thermal pattern for diseased plant"""
        img = self.generate_healthy_plant()
        
        if disease_type == 'fungal':
            # Circular hotspots
            num_spots = np.random.randint(3, 8)
            for _ in range(num_spots):
                x = np.random.randint(50, IMG_SIZE - 50)
                y = np.random.randint(50, IMG_SIZE - 50)
                radius = np.random.randint(10, 30)
                temp_delta = np.random.uniform(2, 5)
                
                # Draw circle with temperature change
                circle_temp = float(img[y, x] + temp_delta)
                cv2.circle(img, (x, y), radius, circle_temp, -1)
                
        elif disease_type == 'bacterial':
            # Linear patterns with cooling
            num_lines = np.random.randint(2, 5)
            for _ in range(num_lines):
                x1 = np.random.randint(0, IMG_SIZE)
                y1 = np.random.randint(0, IMG_SIZE)
                x2 = np.random.randint(0, IMG_SIZE)
                y2 = np.random.randint(0, IMG_SIZE)
                temp_delta = np.random.uniform(-3, -1)
                
                # Draw line with temperature change
                line_temp = float(img[y1, x1] + temp_delta)
                cv2.line(img, (x1, y1), (x2, y2), line_temp, 3)
                
        elif disease_type == 'viral':
            # Mosaic pattern
            block_size = 20
            for y in range(0, IMG_SIZE, block_size):
                for x in range(0, IMG_SIZE, block_size):
                    if np.random.random() > 0.5:
                        temp_delta = np.random.uniform(-2, 3)
                        img[y:y+block_size, x:x+block_size] += temp_delta
                        
        return img
    
    def generate_nutrient_deficient_plant(self, nutrient: str) -> np.ndarray:
        """Generate thermal pattern for nutrient deficiency"""
        img = self.generate_healthy_plant()
        
        if nutrient == 'nitrogen':
            # Overall cooling, uniform temperature
            img -= np.random.uniform(2, 4)
            # Reduce variation
            img = cv2.GaussianBlur(img, (21, 21), 0)
            
        elif nutrient == 'phosphorus':
            # Irregular patches
            num_patches = np.random.randint(5, 15)
            for _ in range(num_patches):
                x = np.random.randint(0, IMG_SIZE)
                y = np.random.randint(0, IMG_SIZE)
                w = np.random.randint(20, 50)
                h = np.random.randint(20, 50)
                temp_delta = np.random.uniform(-1, 2)
                img[y:y+h, x:x+w] += temp_delta
                
        elif nutrient == 'potassium':
            # Edge burn pattern
            mask = np.zeros_like(img)
            cv2.rectangle(mask, (20, 20), (IMG_SIZE-20, IMG_SIZE-20), 1, -1)
            edge_mask = mask == 0
            img[edge_mask] += np.random.uniform(3, 5)
            
        return img

def create_dataset(generator: ThermalDataGenerator, num_samples: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create a balanced dataset with all conditions"""
    images = []
    labels = {
        'water_stress': [],
        'disease': [],
        'nutrients': [],
        'segmentation': []
    }
    
    samples_per_class = num_samples // 10
    
    # Healthy plants
    for _ in range(samples_per_class):
        img = generator.generate_healthy_plant()
        images.append(img)
        labels['water_stress'].append(0.0)
        labels['disease'].append([1.0, 0.0, 0.0, 0.0])  # [healthy, bacterial, fungal, viral]
        labels['nutrients'].append([0.5, 0.5, 0.5])  # [N, P, K] optimal
        labels['segmentation'].append(np.zeros((IMG_SIZE, IMG_SIZE)))
    
    # Water stressed plants
    for _ in range(samples_per_class * 3):
        stress_level = np.random.uniform(0.1, 1.0)
        img = generator.generate_water_stressed_plant(stress_level)
        images.append(img)
        labels['water_stress'].append(stress_level)
        labels['disease'].append([1.0, 0.0, 0.0, 0.0])
        labels['nutrients'].append([0.5, 0.5, 0.5])
        
        # Create segmentation mask
        seg_mask = (img > np.mean(img) + 2).astype(np.float32)
        labels['segmentation'].append(seg_mask)
    
    # Diseased plants
    for disease_type in ['bacterial', 'fungal', 'viral']:
        for _ in range(samples_per_class):
            img = generator.generate_diseased_plant(disease_type)
            images.append(img)
            labels['water_stress'].append(np.random.uniform(0, 0.3))
            
            disease_vec = [0.0, 0.0, 0.0, 0.0]
            disease_idx = ['healthy', 'bacterial', 'fungal', 'viral'].index(disease_type)
            disease_vec[disease_idx] = 1.0
            labels['disease'].append(disease_vec)
            
            labels['nutrients'].append([0.5, 0.5, 0.5])
            
            # Disease segmentation
            seg_mask = np.abs(img - generator.generate_healthy_plant()) > 2
            labels['segmentation'].append(seg_mask.astype(np.float32))
    
    # Nutrient deficient plants
    for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
        for _ in range(samples_per_class):
            img = generator.generate_nutrient_deficient_plant(nutrient)
            images.append(img)
            labels['water_stress'].append(np.random.uniform(0, 0.2))
            labels['disease'].append([1.0, 0.0, 0.0, 0.0])
            
            nutrient_vec = [0.5, 0.5, 0.5]
            nutrient_idx = ['nitrogen', 'phosphorus', 'potassium'].index(nutrient)
            nutrient_vec[nutrient_idx] = 0.1  # Deficient
            labels['nutrients'].append(nutrient_vec)
            
            labels['segmentation'].append(np.zeros((IMG_SIZE, IMG_SIZE)))
    
    # Convert to numpy arrays
    images = np.array(images)
    for key in labels:
        labels[key] = np.array(labels[key])
    
    return images, labels

def build_multi_task_model() -> keras.Model:
    """Build MobileNetV3-based model for multi-task plant health analysis"""
    
    # Input layer - single channel thermal image
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='thermal_input')
    
    # Expand to 3 channels for MobileNetV3
    x = layers.Conv2D(3, 1, padding='same')(inputs)
    
    # Try to load MobileNetV3, fall back to simpler model if it fails
    try:
        # Create a new input for MobileNetV3
        mobilenet_input = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        backbone_model = keras.applications.MobileNetV3Small(
            input_tensor=mobilenet_input,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        # Apply backbone to our processed input
        features = backbone_model(x)
        
        # Freeze early layers
        for layer in backbone_model.layers[:50]:
            layer.trainable = False
    except Exception as e:
        print(f"Warning: Could not load MobileNetV3, using custom CNN: {e}")
        # Fallback to custom lightweight CNN
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        features = x
    
    # Task-specific heads
    # 1. Water stress regression
    water_stress = layers.Dense(64, activation='relu')(features)
    water_stress = layers.Dropout(0.3)(water_stress)
    water_stress_output = layers.Dense(1, activation='sigmoid', name='water_stress')(water_stress)
    
    # 2. Disease classification
    disease = layers.Dense(128, activation='relu')(features)
    disease = layers.Dropout(0.3)(disease)
    disease_output = layers.Dense(4, activation='softmax', name='disease')(disease)
    
    # 3. Nutrient levels
    nutrients = layers.Dense(64, activation='relu')(features)
    nutrients = layers.Dropout(0.3)(nutrients)
    nutrients_output = layers.Dense(3, activation='sigmoid', name='nutrients')(nutrients)
    
    # 4. Segmentation decoder
    # Simplified decoder without skip connections
    seg = layers.Dense(28 * 28 * 32)(features)
    seg = layers.Reshape((28, 28, 32))(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Activation('relu')(seg)
    
    seg = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Activation('relu')(seg)
    
    seg = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Activation('relu')(seg)
    
    seg = layers.Conv2DTranspose(16, 3, strides=2, padding='same')(seg)
    seg = layers.BatchNormalization()(seg)
    seg = layers.Activation('relu')(seg)
    
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(seg)
    
    # Create model
    model = keras.Model(
        inputs=inputs,
        outputs=[water_stress_output, disease_output, nutrients_output, segmentation_output]
    )
    
    return model

def train_model(model: keras.Model, train_data: Tuple[np.ndarray, Dict], 
                val_data: Tuple[np.ndarray, Dict]) -> keras.Model:
    """Train the multi-task model"""
    
    # Compile model with task-specific losses
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss={
            'water_stress': 'mse',
            'disease': 'categorical_crossentropy',
            'nutrients': 'mse',
            'segmentation': 'binary_crossentropy'
        },
        loss_weights={
            'water_stress': 1.0,
            'disease': 1.0,
            'nutrients': 0.5,
            'segmentation': 0.3
        },
        metrics={
            'water_stress': ['mae'],
            'disease': ['accuracy'],
            'nutrients': ['mae'],
            'segmentation': ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
        }
    )
    
    # Prepare data
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    # Normalize images
    train_images = (train_images - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    val_images = (val_images - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    
    # Expand dimensions for Conv2D
    train_images = np.expand_dims(train_images, -1)
    val_images = np.expand_dims(val_images, -1)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'plant_health_best.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        train_images,
        {
            'water_stress': train_labels['water_stress'],
            'disease': train_labels['disease'],
            'nutrients': train_labels['nutrients'],
            'segmentation': train_labels['segmentation']
        },
        validation_data=(
            val_images,
            {
                'water_stress': val_labels['water_stress'],
                'disease': val_labels['disease'],
                'nutrients': val_labels['nutrients'],
                'segmentation': val_labels['segmentation']
            }
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def convert_to_tflite(model: keras.Model, quantize: bool = True) -> bytes:
    """Convert Keras model to TFLite format"""
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    return tflite_model

def representative_dataset_gen():
    """Generate representative dataset for quantization"""
    generator = ThermalDataGenerator()
    for _ in range(100):
        img = generator.generate_healthy_plant()
        img = (img - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        img = np.expand_dims(img, axis=(0, -1)).astype(np.float32)
        yield [img]

def main():
    """Main training pipeline"""
    print("PlantPulse Model Training Pipeline")
    print("==================================")
    
    # Create data generator
    generator = ThermalDataGenerator()
    
    # Generate datasets
    print("\n1. Generating training data...")
    train_images, train_labels = create_dataset(generator, num_samples=10000)
    val_images, val_labels = create_dataset(generator, num_samples=2000)
    print(f"   Training samples: {len(train_images)}")
    print(f"   Validation samples: {len(val_images)}")
    
    # Build model
    print("\n2. Building model...")
    model = build_multi_task_model()
    model.summary()
    
    # Train model
    print("\n3. Training model...")
    model, history = train_model(
        model,
        (train_images, train_labels),
        (val_images, val_labels)
    )
    
    # Save Keras model
    print("\n4. Saving Keras model...")
    model.save('plant_health_model.h5')
    
    # Convert to TFLite
    print("\n5. Converting to TFLite...")
    tflite_model = convert_to_tflite(model, quantize=True)
    
    # Save TFLite model
    with open('plant_health_v1.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Model saved: plant_health_v1.tflite ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    print("\n6. Training complete!")
    print("   - Keras model: plant_health_model.h5")
    print("   - TFLite model: plant_health_v1.tflite")
    print("   - Training history: training_history.json")

if __name__ == "__main__":
    main()