import { TensorflowLite, Tensor } from 'react-native-fast-tflite';
import { Platform } from 'react-native';

export interface DiseaseDetectionResult {
  disease: string;
  confidence: number;
  severity: 'mild' | 'moderate' | 'severe';
  recommendations: string[];
  additionalInfo: string;
}

export class RGBDiseaseModel {
  private model: TensorflowLite | null = null;
  private isLoaded = false;
  
  // Updated for new CycleGAN robust model - same classes, better real-world performance
  private readonly diseaseClasses = [
    'Blight',
    'Healthy',
    'Leaf_Spot',
    'Mosaic_Virus',
    'Nutrient_Deficiency',
    'Powdery_Mildew',
    'Rust'
  ];

  private readonly diseaseInfo: Record<string, {
    description: string;
    severity: (confidence: number) => 'mild' | 'moderate' | 'severe';
    recommendations: string[];
  }> = {
    'Blight': {
      description: 'Fungal disease causing dark spots and wilting',
      severity: (conf) => conf > 0.8 ? 'severe' : conf > 0.5 ? 'moderate' : 'mild',
      recommendations: [
        'Remove affected leaves immediately',
        'Apply copper-based fungicide',
        'Improve air circulation around plant',
        'Avoid overhead watering'
      ]
    },
    'Healthy': {
      description: 'Plant appears healthy with no visible disease',
      severity: () => 'mild',
      recommendations: [
        'Continue current care routine',
        'Monitor regularly for changes',
        'Maintain proper watering schedule'
      ]
    },
    'Leaf_Spot': {
      description: 'Bacterial or fungal spots on leaves',
      severity: (conf) => conf > 0.7 ? 'moderate' : 'mild',
      recommendations: [
        'Remove infected leaves',
        'Avoid wetting foliage when watering',
        'Apply neem oil spray',
        'Ensure proper spacing between plants'
      ]
    },
    'Mosaic_Virus': {
      description: 'Viral infection causing mottled leaf patterns',
      severity: (conf) => conf > 0.6 ? 'severe' : 'moderate',
      recommendations: [
        'Isolate infected plant immediately',
        'No cure - focus on prevention',
        'Control aphids and other vectors',
        'Consider removing severely infected plants'
      ]
    },
    'Nutrient_Deficiency': {
      description: 'Lack of essential nutrients causing discoloration',
      severity: (conf) => conf > 0.7 ? 'moderate' : 'mild',
      recommendations: [
        'Apply balanced fertilizer',
        'Check soil pH levels',
        'Add compost or organic matter',
        'Consider specific nutrient supplements'
      ]
    },
    'Powdery_Mildew': {
      description: 'White powdery fungal growth on leaves',
      severity: (conf) => conf > 0.75 ? 'severe' : conf > 0.5 ? 'moderate' : 'mild',
      recommendations: [
        'Improve air circulation',
        'Apply sulfur or potassium bicarbonate spray',
        'Remove affected leaves',
        'Avoid overhead watering'
      ]
    },
    'Rust': {
      description: 'Orange/brown pustules on leaf undersides',
      severity: (conf) => conf > 0.7 ? 'severe' : conf > 0.4 ? 'moderate' : 'mild',
      recommendations: [
        'Remove and destroy infected leaves',
        'Apply fungicide containing myclobutanil',
        'Water at soil level only',
        'Improve air circulation'
      ]
    }
  };

  async loadModel(): Promise<void> {
    if (this.isLoaded) return;

    try {
      // Load model from assets
      const modelPath = Platform.select({
        android: 'plant_disease_model.tflite',
        ios: 'plant_disease_model.tflite'
      });

      if (!modelPath) {
        throw new Error('Unsupported platform');
      }

      this.model = await TensorflowLite.loadModel({
        model: modelPath,
        delegate: 'metal' // Use GPU acceleration on iOS
      });

      this.isLoaded = true;
      console.log('RGB Disease model loaded successfully');
    } catch (error) {
      console.error('Failed to load RGB disease model:', error);
      throw error;
    }
  }

  preprocessImage(imageData: Uint8Array, width: number, height: number): Float32Array {
    // Resize to 224x224 and normalize to [0, 1]
    const targetSize = 224;
    const output = new Float32Array(targetSize * targetSize * 3);
    
    // Simple bilinear interpolation for resizing
    const scaleX = width / targetSize;
    const scaleY = height / targetSize;
    
    for (let y = 0; y < targetSize; y++) {
      for (let x = 0; x < targetSize; x++) {
        const srcX = Math.floor(x * scaleX);
        const srcY = Math.floor(y * scaleY);
        const srcIdx = (srcY * width + srcX) * 4; // RGBA
        const dstIdx = (y * targetSize + x) * 3; // RGB
        
        // Normalize to [0, 1] - the model has internal normalization to [-1, 1]
        output[dstIdx] = imageData[srcIdx] / 255.0;
        output[dstIdx + 1] = imageData[srcIdx + 1] / 255.0;
        output[dstIdx + 2] = imageData[srcIdx + 2] / 255.0;
      }
    }
    
    return output;
  }

  async detectDisease(imageData: Uint8Array, width: number, height: number): Promise<DiseaseDetectionResult> {
    if (!this.model || !this.isLoaded) {
      await this.loadModel();
    }

    try {
      // Preprocess image
      const preprocessed = this.preprocessImage(imageData, width, height);
      
      // Create input tensor
      const inputTensor = new Tensor(
        new Float32Array(preprocessed),
        [1, 224, 224, 3]
      );

      // Run inference
      const output = await this.model!.run([inputTensor]);
      const predictions = output[0].data as Float32Array;
      
      // Find top prediction
      let maxIndex = 0;
      let maxConfidence = predictions[0];
      
      for (let i = 1; i < predictions.length; i++) {
        if (predictions[i] > maxConfidence) {
          maxConfidence = predictions[i];
          maxIndex = i;
        }
      }

      const disease = this.diseaseClasses[maxIndex];
      const info = this.diseaseInfo[disease];
      
      return {
        disease,
        confidence: maxConfidence,
        severity: info.severity(maxConfidence),
        recommendations: info.recommendations,
        additionalInfo: info.description
      };
    } catch (error) {
      console.error('Disease detection failed:', error);
      throw error;
    }
  }

  async detectMultipleDiseases(imageData: Uint8Array, width: number, height: number, threshold = 0.3): Promise<DiseaseDetectionResult[]> {
    if (!this.model || !this.isLoaded) {
      await this.loadModel();
    }

    try {
      const preprocessed = this.preprocessImage(imageData, width, height);
      
      const inputTensor = new Tensor(
        new Float32Array(preprocessed),
        [1, 224, 224, 3]
      );

      const output = await this.model!.run([inputTensor]);
      const predictions = output[0].data as Float32Array;
      
      // Get all predictions above threshold
      const results: DiseaseDetectionResult[] = [];
      
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] >= threshold) {
          const disease = this.diseaseClasses[i];
          const info = this.diseaseInfo[disease];
          
          results.push({
            disease,
            confidence: predictions[i],
            severity: info.severity(predictions[i]),
            recommendations: info.recommendations,
            additionalInfo: info.description
          });
        }
      }
      
      // Sort by confidence
      results.sort((a, b) => b.confidence - a.confidence);
      
      return results;
    } catch (error) {
      console.error('Multiple disease detection failed:', error);
      throw error;
    }
  }

  dispose(): void {
    if (this.model) {
      this.model.close();
      this.model = null;
      this.isLoaded = false;
    }
  }
}

// Singleton instance
export const rgbDiseaseModel = new RGBDiseaseModel();