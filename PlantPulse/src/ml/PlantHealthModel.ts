import {
  TensorflowLite,
  Tensor,
  Delegate,
} from 'react-native-fast-tflite';
import { ThermalFrame } from '../types/thermal';
import { PlantHealthAnalysis } from '../types/plant';
import { runPlaceholderInference } from './models/plant_health_v1_placeholder';

export class PlantHealthModel {
  private tflite: TensorflowLite | null = null;
  private modelPath: string;
  private isLoaded: boolean = false;
  private delegate: Delegate = 'default';

  constructor(modelPath: string = 'plant_health_v1.tflite') {
    this.modelPath = modelPath;
  }

  async initialize(useGPU: boolean = true): Promise<void> {
    try {
      // Set delegate for GPU acceleration
      if (useGPU) {
        this.delegate = 'gpu';
      }

      // For development, use placeholder if TFLite fails to load
      try {
        // Load the model
        this.tflite = await TensorflowLite.loadModel({
          path: this.modelPath,
          delegate: this.delegate,
        });
      } catch (modelError) {
        console.warn('Using placeholder model for development:', modelError);
        // Continue with placeholder
      }

      this.isLoaded = true;
      console.log('Plant health model initialized');
    } catch (error) {
      console.error('Failed to initialize plant health model:', error);
      throw error;
    }
  }

  async analyze(thermalFrame: ThermalFrame): Promise<PlantHealthAnalysis> {
    if (!this.isLoaded) {
      throw new Error('Model not initialized');
    }

    // Preprocess thermal data
    const input = await this.preprocessThermalData(thermalFrame);

    // Run inference
    const startTime = Date.now();
    let outputs: Tensor[];
    
    if (this.tflite) {
      outputs = await this.tflite.run([input]);
    } else {
      // Use placeholder for development
      const placeholderResults = await runPlaceholderInference(input);
      outputs = [
        { data: placeholderResults.water_stress, shape: [1, 1], dataType: 'float32' },
        { data: placeholderResults.disease_class, shape: [1, 4], dataType: 'float32' },
        { data: placeholderResults.nutrients, shape: [1, 3], dataType: 'float32' },
        { data: placeholderResults.segmentation, shape: [1, 224, 224, 1], dataType: 'float32' },
      ];
    }
    
    const inferenceTime = Date.now() - startTime;
    console.log(`Inference completed in ${inferenceTime}ms`);

    // Post-process outputs
    return this.postprocessOutputs(outputs, thermalFrame);
  }

  private async preprocessThermalData(thermalFrame: ThermalFrame): Promise<Tensor> {
    const { temperatureData } = thermalFrame;
    
    // Model expects 224x224 input, so we need to resize
    const resized = this.resizeThermalData(temperatureData, 256, 192, 224, 224);
    
    // Normalize to 0-1 range
    const normalized = this.normalizeTemperatureData(resized);
    
    // Create tensor with shape [1, 224, 224, 1]
    return {
      data: normalized,
      shape: [1, 224, 224, 1],
      dataType: 'float32',
    };
  }

  private resizeThermalData(
    data: Float32Array,
    srcWidth: number,
    srcHeight: number,
    dstWidth: number,
    dstHeight: number
  ): Float32Array {
    const resized = new Float32Array(dstWidth * dstHeight);
    
    // Simple bilinear interpolation
    const xRatio = srcWidth / dstWidth;
    const yRatio = srcHeight / dstHeight;
    
    for (let y = 0; y < dstHeight; y++) {
      for (let x = 0; x < dstWidth; x++) {
        const srcX = x * xRatio;
        const srcY = y * yRatio;
        
        const x0 = Math.floor(srcX);
        const x1 = Math.min(x0 + 1, srcWidth - 1);
        const y0 = Math.floor(srcY);
        const y1 = Math.min(y0 + 1, srcHeight - 1);
        
        const dx = srcX - x0;
        const dy = srcY - y0;
        
        const v00 = data[y0 * srcWidth + x0];
        const v01 = data[y0 * srcWidth + x1];
        const v10 = data[y1 * srcWidth + x0];
        const v11 = data[y1 * srcWidth + x1];
        
        const v0 = v00 * (1 - dx) + v01 * dx;
        const v1 = v10 * (1 - dx) + v11 * dx;
        const value = v0 * (1 - dy) + v1 * dy;
        
        resized[y * dstWidth + x] = value;
      }
    }
    
    return resized;
  }

  private normalizeTemperatureData(data: Float32Array): Float32Array {
    const normalized = new Float32Array(data.length);
    
    // Find min and max for normalization
    let min = Infinity;
    let max = -Infinity;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        min = Math.min(min, data[i]);
        max = Math.max(max, data[i]);
      }
    }
    
    // Normalize to 0-1 range
    const range = max - min;
    if (range > 0) {
      for (let i = 0; i < data.length; i++) {
        normalized[i] = (data[i] - min) / range;
      }
    }
    
    return normalized;
  }

  private postprocessOutputs(
    outputs: Tensor[],
    thermalFrame: ThermalFrame
  ): PlantHealthAnalysis {
    // Model outputs:
    // 0: Water stress index (1 value)
    // 1: Disease classification (4 values: healthy, bacterial, fungal, viral)
    // 2: Nutrient levels (3 values: N, P, K)
    // 3: Affected area mask (224x224)
    
    const waterStressOutput = outputs[0].data as Float32Array;
    const diseaseOutput = outputs[1].data as Float32Array;
    const nutrientOutput = outputs[2].data as Float32Array;
    const affectedAreaMask = outputs[3].data as Float32Array;
    
    // Process water stress
    const waterStressIndex = waterStressOutput[0];
    const stressLevel = this.categorizeStressLevel(waterStressIndex);
    
    // Process disease detection
    const diseaseType = this.getDiseaseType(diseaseOutput);
    const diseaseConfidence = Math.max(...diseaseOutput);
    const affectedArea = this.calculateAffectedArea(affectedAreaMask);
    
    // Process nutrient status
    const nutrientStatus = {
      nitrogen: this.getNutrientLevel(nutrientOutput[0]),
      phosphorus: this.getNutrientLevel(nutrientOutput[1]),
      potassium: this.getNutrientLevel(nutrientOutput[2]),
    };
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(
      stressLevel,
      diseaseType,
      nutrientStatus
    );
    
    return {
      id: `analysis_${Date.now()}`,
      waterStressIndex,
      stressLevel,
      diseaseDetection: {
        type: diseaseType,
        confidence: diseaseConfidence,
        affectedArea,
        patterns: [], // Would be extracted from segmentation mask
      },
      nutrientStatus,
      recommendations,
      timestamp: thermalFrame.timestamp,
      thermalFrameId: thermalFrame.deviceId,
    };
  }

  private categorizeStressLevel(
    waterStressIndex: number
  ): 'none' | 'mild' | 'moderate' | 'severe' {
    if (waterStressIndex < 0.2) return 'none';
    if (waterStressIndex < 0.36) return 'mild';
    if (waterStressIndex < 0.6) return 'moderate';
    return 'severe';
  }

  private getDiseaseType(
    diseaseOutput: Float32Array
  ): 'healthy' | 'bacterial' | 'fungal' | 'viral' {
    const classes = ['healthy', 'bacterial', 'fungal', 'viral'] as const;
    let maxIndex = 0;
    let maxValue = diseaseOutput[0];
    
    for (let i = 1; i < diseaseOutput.length; i++) {
      if (diseaseOutput[i] > maxValue) {
        maxValue = diseaseOutput[i];
        maxIndex = i;
      }
    }
    
    return classes[maxIndex];
  }

  private getNutrientLevel(value: number): 'deficient' | 'optimal' | 'excess' {
    if (value < 0.3) return 'deficient';
    if (value > 0.7) return 'excess';
    return 'optimal';
  }

  private calculateAffectedArea(mask: Float32Array): number {
    let affectedPixels = 0;
    
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0.5) {
        affectedPixels++;
      }
    }
    
    return (affectedPixels / mask.length) * 100;
  }

  private generateRecommendations(
    stressLevel: string,
    diseaseType: string,
    nutrientStatus: any
  ): string[] {
    const recommendations: string[] = [];
    
    // Water stress recommendations
    switch (stressLevel) {
      case 'mild':
        recommendations.push('Increase watering frequency slightly');
        recommendations.push('Check soil moisture levels');
        break;
      case 'moderate':
        recommendations.push('Water immediately');
        recommendations.push('Consider adding mulch to retain moisture');
        break;
      case 'severe':
        recommendations.push('Water thoroughly and immediately');
        recommendations.push('Check for root problems or poor drainage');
        recommendations.push('Consider relocating plant if stress persists');
        break;
    }
    
    // Disease recommendations
    switch (diseaseType) {
      case 'bacterial':
        recommendations.push('Remove affected leaves');
        recommendations.push('Improve air circulation');
        recommendations.push('Avoid overhead watering');
        break;
      case 'fungal':
        recommendations.push('Apply fungicide if severe');
        recommendations.push('Reduce humidity around plant');
        recommendations.push('Remove infected plant material');
        break;
      case 'viral':
        recommendations.push('Isolate plant to prevent spread');
        recommendations.push('Monitor for insect vectors');
        recommendations.push('Consider removing if severely affected');
        break;
    }
    
    // Nutrient recommendations
    if (nutrientStatus.nitrogen === 'deficient') {
      recommendations.push('Apply nitrogen-rich fertilizer');
    }
    if (nutrientStatus.phosphorus === 'deficient') {
      recommendations.push('Add phosphorus supplement or bone meal');
    }
    if (nutrientStatus.potassium === 'deficient') {
      recommendations.push('Apply potassium-rich fertilizer');
    }
    
    return recommendations;
  }

  async warmup(): Promise<void> {
    if (!this.isLoaded) {
      await this.initialize();
    }
    
    // Run a dummy inference to warm up the model
    const dummyData = new Float32Array(256 * 192).fill(20);
    const dummyFrame: ThermalFrame = {
      temperatureData: dummyData,
      timestamp: Date.now(),
      deviceId: 'warmup',
      calibrationOffset: 0,
    };
    
    await this.analyze(dummyFrame);
    console.log('Model warmed up');
  }

  dispose(): void {
    if (this.tflite) {
      this.tflite.close();
      this.tflite = null;
      this.isLoaded = false;
    }
  }

  getModelInfo(): {
    isLoaded: boolean;
    delegate: string;
    modelPath: string;
  } {
    return {
      isLoaded: this.isLoaded,
      delegate: this.delegate,
      modelPath: this.modelPath,
    };
  }
}