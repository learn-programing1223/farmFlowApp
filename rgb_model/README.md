# PlantPulse RGB Model - 95% Accuracy Plant Disease Detection

## Overview
This is the **production-ready** RGB model for PlantPulse that achieves **95.52% validation accuracy** and **95.10% test accuracy** on plant disease detection across 7 disease categories.

## Model Performance
- **Validation Accuracy**: 95.52%
- **Test Accuracy**: 95.10% (on completely unseen data)
- **Precision**: 96.03%
- **Recall**: 94.57%
- **F1 Score**: 95.30%
- **Model Size**: 17.4 MB (H5), ~5 MB (TFLite)

## Disease Categories
The model detects 7 universal plant disease categories:
1. **Blight** (94.67% accuracy)
2. **Healthy** (98.00% accuracy)
3. **Leaf Spot** (89.00% accuracy)
4. **Mosaic Virus** (99.33% accuracy)
5. **Nutrient Deficiency** (99.67% accuracy)
6. **Powdery Mildew** (94.67% accuracy)
7. **Rust** (90.33% accuracy)

## Key Innovation
The breakthrough was discovering that **[-1, 1] normalization** instead of [0, 1] was critical for this dataset. This is implemented as the first Lambda layer in the model.

## Project Structure
```
rgb_model/
├── train_working_solution.py   # The winning training script (95.52% accuracy)
├── evaluate_final_model.py     # Test set evaluation script
├── convert_to_mobile.py        # Convert to TFLite for mobile deployment
├── models/
│   ├── best_working_model.h5   # The trained model (95.52% accuracy)
│   ├── final_working_model.h5  # Backup of the model
│   ├── training_history_working.json
│   ├── test_evaluation_results.json
│   └── confusion_matrix_test.png
├── data/
│   └── splits/                 # Preprocessed data (X_train.npy, etc.)
├── src/
│   └── data_loader.py          # Data loading utilities
├── summary.md                  # Detailed documentation
└── README.md                   # This file
```

## Quick Start

### 1. Test the Model
```python
python evaluate_final_model.py
```

### 2. Convert for Mobile Deployment
```python
python convert_to_mobile.py
```

### 3. Use for Inference
```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('models/best_working_model.h5')

# Preprocess image (224x224x3, values 0-1)
image = preprocess_your_image()  

# Predict
prediction = model.predict(np.expand_dims(image, axis=0))
disease_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])

print(f"Disease: {class_names[disease_class]}")
print(f"Confidence: {confidence:.2%}")
```

## Model Architecture
Custom CNN with 4 convolutional blocks:
- Input normalization: [-1, 1] scaling (critical!)
- Conv blocks: 32 → 64 → 128 → 256 filters
- BatchNormalization and Dropout for regularization
- GlobalAveragePooling instead of Flatten
- Dense layers: 512 → 256 → 7 (output)
- Total parameters: 1.44M

## Training Details
- **Dataset**: PlantVillage (14,000 images)
- **Split**: 70% train, 15% validation, 15% test
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam with learning rate scheduling
- **Data Augmentation**: Random flip, brightness, contrast
- **Training Time**: ~9 hours on CPU

## Requirements
```
tensorflow>=2.12.0
numpy
scikit-learn
matplotlib
```

## Next Steps
1. Convert to TensorFlow Lite for mobile deployment ✅
2. Integrate into React Native app
3. Deploy on-device for offline functionality

## Why This Model Works
After testing 20+ different approaches including EfficientNet and ResNet50 transfer learning, we found that:
1. Simple custom CNNs outperform complex transfer learning for plant diseases
2. [-1, 1] normalization is critical for this specific dataset
3. Transfer learning from ImageNet doesn't help with plant leaves
4. Starting from scratch allows proper feature learning

## License
Part of the PlantPulse project - for plant health monitoring

## Contact
Repository: https://github.com/[your-username]/farmFlowApp