import { ThermalFrame } from '../types/thermal';
import { NutrientLevel, PlantProfile } from '../types/plant';

export class NutrientAnalyzer {
  analyzeNutrientStatus(
    thermalFrame: ThermalFrame,
    plant: PlantProfile,
    historicalData: ThermalFrame[]
  ): {
    nitrogen: NutrientLevel;
    phosphorus: NutrientLevel;
    potassium: NutrientLevel;
    confidence: number;
  } {
    const { temperatureData } = thermalFrame;
    
    // Extract temperature features
    const features = this.extractTemperatureFeatures(temperatureData);
    
    // Analyze temperature patterns specific to nutrient deficiencies
    const nitrogenStatus = this.analyzeNitrogenStatus(features, historicalData);
    const phosphorusStatus = this.analyzePhosphorusStatus(features, historicalData);
    const potassiumStatus = this.analyzePotassiumStatus(features, historicalData);
    
    // Calculate overall confidence based on data quality and consistency
    const confidence = this.calculateConfidence(features, historicalData);
    
    return {
      nitrogen: nitrogenStatus,
      phosphorus: phosphorusStatus,
      potassium: potassiumStatus,
      confidence,
    };
  }

  private extractTemperatureFeatures(temperatureData: Float32Array): {
    mean: number;
    std: number;
    skewness: number;
    kurtosis: number;
    gradient: { x: number; y: number };
    hotspotCount: number;
    coldspotCount: number;
    uniformity: number;
  } {
    const width = 256;
    const height = 192;
    
    // Calculate basic statistics
    const { mean, std } = this.calculateMeanStd(temperatureData);
    const skewness = this.calculateSkewness(temperatureData, mean, std);
    const kurtosis = this.calculateKurtosis(temperatureData, mean, std);
    
    // Calculate temperature gradient
    const gradient = this.calculateGradient(temperatureData, width, height);
    
    // Count hot and cold spots
    const hotspotCount = this.countHotspots(temperatureData, mean, std);
    const coldspotCount = this.countColdspots(temperatureData, mean, std);
    
    // Calculate temperature uniformity
    const uniformity = this.calculateUniformity(temperatureData, width, height);
    
    return {
      mean,
      std,
      skewness,
      kurtosis,
      gradient,
      hotspotCount,
      coldspotCount,
      uniformity,
    };
  }

  private calculateMeanStd(data: Float32Array): { mean: number; std: number } {
    let sum = 0;
    let count = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        sum += data[i];
        count++;
      }
    }
    
    const mean = sum / count;
    
    let sumSquaredDiff = 0;
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        sumSquaredDiff += Math.pow(data[i] - mean, 2);
      }
    }
    
    const std = Math.sqrt(sumSquaredDiff / count);
    
    return { mean, std };
  }

  private calculateSkewness(
    data: Float32Array,
    mean: number,
    std: number
  ): number {
    if (std === 0) return 0;
    
    let sumCubedDiff = 0;
    let count = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        sumCubedDiff += Math.pow((data[i] - mean) / std, 3);
        count++;
      }
    }
    
    return sumCubedDiff / count;
  }

  private calculateKurtosis(
    data: Float32Array,
    mean: number,
    std: number
  ): number {
    if (std === 0) return 0;
    
    let sumQuadDiff = 0;
    let count = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        sumQuadDiff += Math.pow((data[i] - mean) / std, 4);
        count++;
      }
    }
    
    return (sumQuadDiff / count) - 3; // Excess kurtosis
  }

  private calculateGradient(
    data: Float32Array,
    width: number,
    height: number
  ): { x: number; y: number } {
    let gradX = 0;
    let gradY = 0;
    let count = 0;
    
    // Sobel operator for gradient calculation
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        
        // Horizontal gradient (Sobel X)
        const gx = (
          -1 * data[(y - 1) * width + (x - 1)] +
           1 * data[(y - 1) * width + (x + 1)] +
          -2 * data[y * width + (x - 1)] +
           2 * data[y * width + (x + 1)] +
          -1 * data[(y + 1) * width + (x - 1)] +
           1 * data[(y + 1) * width + (x + 1)]
        ) / 8;
        
        // Vertical gradient (Sobel Y)
        const gy = (
          -1 * data[(y - 1) * width + (x - 1)] +
          -2 * data[(y - 1) * width + x] +
          -1 * data[(y - 1) * width + (x + 1)] +
           1 * data[(y + 1) * width + (x - 1)] +
           2 * data[(y + 1) * width + x] +
           1 * data[(y + 1) * width + (x + 1)]
        ) / 8;
        
        gradX += Math.abs(gx);
        gradY += Math.abs(gy);
        count++;
      }
    }
    
    return {
      x: gradX / count,
      y: gradY / count,
    };
  }

  private countHotspots(
    data: Float32Array,
    mean: number,
    std: number
  ): number {
    let count = 0;
    const threshold = mean + 2 * std;
    
    for (let i = 0; i < data.length; i++) {
      if (data[i] > threshold) {
        count++;
      }
    }
    
    return count;
  }

  private countColdspots(
    data: Float32Array,
    mean: number,
    std: number
  ): number {
    let count = 0;
    const threshold = mean - 2 * std;
    
    for (let i = 0; i < data.length; i++) {
      if (data[i] < threshold) {
        count++;
      }
    }
    
    return count;
  }

  private calculateUniformity(
    data: Float32Array,
    width: number,
    height: number
  ): number {
    const blockSize = 16;
    const blocksX = Math.floor(width / blockSize);
    const blocksY = Math.floor(height / blockSize);
    const blockMeans: number[] = [];
    
    for (let by = 0; by < blocksY; by++) {
      for (let bx = 0; bx < blocksX; bx++) {
        let sum = 0;
        let count = 0;
        
        for (let y = 0; y < blockSize; y++) {
          for (let x = 0; x < blockSize; x++) {
            const px = bx * blockSize + x;
            const py = by * blockSize + y;
            
            if (px < width && py < height) {
              sum += data[py * width + px];
              count++;
            }
          }
        }
        
        if (count > 0) {
          blockMeans.push(sum / count);
        }
      }
    }
    
    // Calculate coefficient of variation
    const { mean, std } = this.calculateMeanStd(new Float32Array(blockMeans));
    return std === 0 ? 1 : 1 - (std / mean);
  }

  private analyzeNitrogenStatus(
    features: ReturnType<typeof this.extractTemperatureFeatures>,
    historicalData: ThermalFrame[]
  ): NutrientLevel {
    // Nitrogen deficiency characteristics:
    // - Lower overall leaf temperature (reduced metabolic activity)
    // - More uniform temperature distribution (less variation)
    // - Starts from older leaves (bottom of plant)
    
    const tempScore = this.normalizeScore(features.mean, 18, 25, true);
    const uniformityScore = this.normalizeScore(features.uniformity, 0.6, 0.9, false);
    const gradientScore = this.normalizeScore(
      features.gradient.y,
      0.1,
      0.5,
      true
    ); // Higher gradient suggests deficiency pattern
    
    const nitrogenScore = (tempScore + uniformityScore + gradientScore) / 3;
    
    if (nitrogenScore < 0.3) {
      return 'deficient';
    } else if (nitrogenScore > 0.7) {
      return 'excess';
    }
    
    return 'optimal';
  }

  private analyzePhosphorusStatus(
    features: ReturnType<typeof this.extractTemperatureFeatures>,
    historicalData: ThermalFrame[]
  ): NutrientLevel {
    // Phosphorus deficiency characteristics:
    // - Dark green/purple coloration (affects temperature absorption)
    // - Reduced growth rate (lower metabolic heat)
    // - Affects older leaves first
    
    const hotspotScore = this.normalizeScore(features.hotspotCount, 10, 100, false);
    const skewnessScore = this.normalizeScore(Math.abs(features.skewness), 0, 1, true);
    const stdScore = this.normalizeScore(features.std, 1, 4, false);
    
    const phosphorusScore = (hotspotScore + skewnessScore + stdScore) / 3;
    
    if (phosphorusScore < 0.35) {
      return 'deficient';
    } else if (phosphorusScore > 0.65) {
      return 'excess';
    }
    
    return 'optimal';
  }

  private analyzePotassiumStatus(
    features: ReturnType<typeof this.extractTemperatureFeatures>,
    historicalData: ThermalFrame[]
  ): NutrientLevel {
    // Potassium deficiency characteristics:
    // - Edge burn patterns (higher edge temperatures)
    // - Irregular temperature distribution
    // - Affects water regulation (stomatal function)
    
    const kurtosisScore = this.normalizeScore(features.kurtosis, -1, 2, true);
    const coldspotScore = this.normalizeScore(features.coldspotCount, 10, 100, true);
    const gradientXScore = this.normalizeScore(features.gradient.x, 0.1, 0.5, true);
    
    const potassiumScore = (kurtosisScore + coldspotScore + gradientXScore) / 3;
    
    if (potassiumScore < 0.3) {
      return 'deficient';
    } else if (potassiumScore > 0.7) {
      return 'excess';
    }
    
    return 'optimal';
  }

  private normalizeScore(
    value: number,
    min: number,
    max: number,
    inverse: boolean = false
  ): number {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return inverse ? 1 - normalized : normalized;
  }

  private calculateConfidence(
    features: ReturnType<typeof this.extractTemperatureFeatures>,
    historicalData: ThermalFrame[]
  ): number {
    let confidence = 0.5; // Base confidence
    
    // Increase confidence with more historical data
    if (historicalData.length > 10) {
      confidence += 0.1;
    }
    if (historicalData.length > 50) {
      confidence += 0.1;
    }
    
    // Increase confidence with better data quality
    if (features.std > 0.5 && features.std < 5) {
      confidence += 0.1;
    }
    
    // Decrease confidence with extreme values
    if (Math.abs(features.skewness) > 2 || Math.abs(features.kurtosis) > 5) {
      confidence -= 0.1;
    }
    
    return Math.max(0.2, Math.min(0.95, confidence));
  }

  generateNutrientDeficiencyMap(
    thermalFrame: ThermalFrame,
    nutrientType: 'nitrogen' | 'phosphorus' | 'potassium'
  ): Float32Array {
    const { temperatureData } = thermalFrame;
    const width = 256;
    const height = 192;
    const mapSize = width * height;
    const deficiencyMap = new Float32Array(mapSize);
    
    switch (nutrientType) {
      case 'nitrogen':
        return this.generateNitrogenDeficiencyMap(temperatureData, width, height);
      case 'phosphorus':
        return this.generatePhosphorusDeficiencyMap(temperatureData, width, height);
      case 'potassium':
        return this.generatePotassiumDeficiencyMap(temperatureData, width, height);
      default:
        return deficiencyMap;
    }
  }

  private generateNitrogenDeficiencyMap(
    temperatureData: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    const deficiencyMap = new Float32Array(width * height);
    const { mean, std } = this.calculateMeanStd(temperatureData);
    
    // Nitrogen deficiency shows as cooler, more uniform areas
    for (let i = 0; i < temperatureData.length; i++) {
      const temp = temperatureData[i];
      const deviation = (mean - temp) / std;
      
      // Higher score for cooler temperatures
      const score = Math.max(0, Math.min(1, (deviation + 2) / 4));
      deficiencyMap[i] = score;
    }
    
    return this.smoothMap(deficiencyMap, width, height);
  }

  private generatePhosphorusDeficiencyMap(
    temperatureData: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    const deficiencyMap = new Float32Array(width * height);
    
    // Phosphorus deficiency affects older leaves (typically bottom)
    // and shows as irregular temperature patterns
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const temp = temperatureData[idx];
        
        // Calculate local variance
        let localVariance = 0;
        let count = 0;
        
        for (let dy = -3; dy <= 3; dy++) {
          for (let dx = -3; dx <= 3; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              const nidx = ny * width + nx;
              localVariance += Math.pow(temp - temperatureData[nidx], 2);
              count++;
            }
          }
        }
        
        localVariance /= count;
        
        // Higher variance suggests deficiency
        const score = Math.max(0, Math.min(1, localVariance / 5));
        deficiencyMap[idx] = score;
      }
    }
    
    return deficiencyMap;
  }

  private generatePotassiumDeficiencyMap(
    temperatureData: Float32Array,
    width: number,
    height: number
  ): Float32Array {
    const deficiencyMap = new Float32Array(width * height);
    
    // Potassium deficiency shows as edge burn (higher edge temperatures)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        
        // Calculate distance from edge
        const edgeDistX = Math.min(x, width - 1 - x);
        const edgeDistY = Math.min(y, height - 1 - y);
        const edgeDist = Math.min(edgeDistX, edgeDistY);
        
        // Edge proximity factor
        const edgeFactor = 1 - (edgeDist / Math.min(width, height) * 2);
        
        // Temperature factor
        const temp = temperatureData[idx];
        const tempFactor = (temp - 20) / 10; // Normalize around expected temp
        
        // Combine factors
        const score = Math.max(0, Math.min(1, edgeFactor * tempFactor));
        deficiencyMap[idx] = score;
      }
    }
    
    return deficiencyMap;
  }

  private smoothMap(
    map: Float32Array,
    width: number,
    height: number,
    kernelSize: number = 5
  ): Float32Array {
    const smoothed = new Float32Array(width * height);
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let count = 0;
        
        for (let dy = -halfKernel; dy <= halfKernel; dy++) {
          for (let dx = -halfKernel; dx <= halfKernel; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              sum += map[ny * width + nx];
              count++;
            }
          }
        }
        
        smoothed[y * width + x] = sum / count;
      }
    }
    
    return smoothed;
  }
}