import { ThermalFrame } from '../types/thermal';

export class ThermalPreprocessor {
  augmentThermalData(
    thermalFrame: ThermalFrame,
    augmentations: {
      rotate?: boolean;
      flip?: boolean;
      noise?: boolean;
      temperatureShift?: boolean;
    } = {}
  ): ThermalFrame[] {
    const augmented: ThermalFrame[] = [thermalFrame];
    
    if (augmentations.rotate) {
      augmented.push(...this.rotateAugmentation(thermalFrame));
    }
    
    if (augmentations.flip) {
      augmented.push(...this.flipAugmentation(thermalFrame));
    }
    
    if (augmentations.noise) {
      augmented.push(this.addNoise(thermalFrame));
    }
    
    if (augmentations.temperatureShift) {
      augmented.push(...this.temperatureShiftAugmentation(thermalFrame));
    }
    
    return augmented;
  }

  private rotateAugmentation(frame: ThermalFrame): ThermalFrame[] {
    const rotated: ThermalFrame[] = [];
    const { temperatureData } = frame;
    const width = 256;
    const height = 192;
    
    // 90 degree rotation
    const rotated90 = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = y * width + x;
        const dstIdx = x * height + (height - 1 - y);
        rotated90[dstIdx] = temperatureData[srcIdx];
      }
    }
    
    rotated.push({
      ...frame,
      temperatureData: rotated90,
      timestamp: frame.timestamp + 1,
    });
    
    // 180 degree rotation
    const rotated180 = new Float32Array(width * height);
    for (let i = 0; i < temperatureData.length; i++) {
      rotated180[temperatureData.length - 1 - i] = temperatureData[i];
    }
    
    rotated.push({
      ...frame,
      temperatureData: rotated180,
      timestamp: frame.timestamp + 2,
    });
    
    return rotated;
  }

  private flipAugmentation(frame: ThermalFrame): ThermalFrame[] {
    const flipped: ThermalFrame[] = [];
    const { temperatureData } = frame;
    const width = 256;
    const height = 192;
    
    // Horizontal flip
    const flippedH = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = y * width + x;
        const dstIdx = y * width + (width - 1 - x);
        flippedH[dstIdx] = temperatureData[srcIdx];
      }
    }
    
    flipped.push({
      ...frame,
      temperatureData: flippedH,
      timestamp: frame.timestamp + 3,
    });
    
    // Vertical flip
    const flippedV = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = y * width + x;
        const dstIdx = (height - 1 - y) * width + x;
        flippedV[dstIdx] = temperatureData[srcIdx];
      }
    }
    
    flipped.push({
      ...frame,
      temperatureData: flippedV,
      timestamp: frame.timestamp + 4,
    });
    
    return flipped;
  }

  private addNoise(frame: ThermalFrame): ThermalFrame {
    const { temperatureData } = frame;
    const noisy = new Float32Array(temperatureData.length);
    const noiseLevel = 0.5; // ±0.5°C
    
    for (let i = 0; i < temperatureData.length; i++) {
      const noise = (Math.random() - 0.5) * 2 * noiseLevel;
      noisy[i] = temperatureData[i] + noise;
    }
    
    return {
      ...frame,
      temperatureData: noisy,
      timestamp: frame.timestamp + 5,
    };
  }

  private temperatureShiftAugmentation(frame: ThermalFrame): ThermalFrame[] {
    const shifted: ThermalFrame[] = [];
    const { temperatureData } = frame;
    const shifts = [-2, -1, 1, 2]; // Temperature shifts in °C
    
    for (const shift of shifts) {
      const shiftedData = new Float32Array(temperatureData.length);
      for (let i = 0; i < temperatureData.length; i++) {
        shiftedData[i] = temperatureData[i] + shift;
      }
      
      shifted.push({
        ...frame,
        temperatureData: shiftedData,
        timestamp: frame.timestamp + 6 + shift,
      });
    }
    
    return shifted;
  }

  extractFeatures(thermalFrame: ThermalFrame): Float32Array {
    const { temperatureData } = thermalFrame;
    const features: number[] = [];
    
    // Statistical features
    const stats = this.calculateStatistics(temperatureData);
    features.push(
      stats.mean,
      stats.std,
      stats.min,
      stats.max,
      stats.skewness,
      stats.kurtosis
    );
    
    // Texture features (simplified GLCM)
    const texture = this.calculateTextureFeatures(temperatureData, 256, 192);
    features.push(
      texture.contrast,
      texture.correlation,
      texture.energy,
      texture.homogeneity
    );
    
    // Histogram features
    const histogram = this.calculateHistogram(temperatureData, 32);
    features.push(...histogram);
    
    // Gradient features
    const gradient = this.calculateGradientFeatures(temperatureData, 256, 192);
    features.push(
      gradient.meanMagnitude,
      gradient.stdMagnitude,
      gradient.dominantDirection
    );
    
    return new Float32Array(features);
  }

  private calculateStatistics(data: Float32Array): {
    mean: number;
    std: number;
    min: number;
    max: number;
    skewness: number;
    kurtosis: number;
  } {
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    let count = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        sum += data[i];
        min = Math.min(min, data[i]);
        max = Math.max(max, data[i]);
        count++;
      }
    }
    
    const mean = sum / count;
    
    let variance = 0;
    let skewness = 0;
    let kurtosis = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        const diff = data[i] - mean;
        variance += diff * diff;
        skewness += diff * diff * diff;
        kurtosis += diff * diff * diff * diff;
      }
    }
    
    variance /= count;
    const std = Math.sqrt(variance);
    
    if (std > 0) {
      skewness = (skewness / count) / Math.pow(std, 3);
      kurtosis = (kurtosis / count) / Math.pow(std, 4) - 3;
    }
    
    return { mean, std, min, max, skewness, kurtosis };
  }

  private calculateTextureFeatures(
    data: Float32Array,
    width: number,
    height: number
  ): {
    contrast: number;
    correlation: number;
    energy: number;
    homogeneity: number;
  } {
    // Simplified GLCM calculation
    const levels = 8; // Quantization levels
    const glcm = new Float32Array(levels * levels);
    
    // Quantize temperatures
    const { min, max } = this.calculateStatistics(data);
    const range = max - min;
    
    // Build co-occurrence matrix
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const idx = y * width + x;
        const rightIdx = idx + 1;
        
        const level1 = Math.floor(((data[idx] - min) / range) * (levels - 1));
        const level2 = Math.floor(((data[rightIdx] - min) / range) * (levels - 1));
        
        if (level1 >= 0 && level1 < levels && level2 >= 0 && level2 < levels) {
          glcm[level1 * levels + level2]++;
        }
      }
    }
    
    // Normalize GLCM
    const total = glcm.reduce((sum, val) => sum + val, 0);
    for (let i = 0; i < glcm.length; i++) {
      glcm[i] /= total;
    }
    
    // Calculate texture features
    let contrast = 0;
    let correlation = 0;
    let energy = 0;
    let homogeneity = 0;
    
    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = glcm[i * levels + j];
        contrast += p * (i - j) * (i - j);
        energy += p * p;
        homogeneity += p / (1 + Math.abs(i - j));
      }
    }
    
    return { contrast, correlation, energy, homogeneity };
  }

  private calculateHistogram(data: Float32Array, bins: number): number[] {
    const histogram = new Array(bins).fill(0);
    const { min, max } = this.calculateStatistics(data);
    const range = max - min;
    
    for (let i = 0; i < data.length; i++) {
      if (!isNaN(data[i]) && isFinite(data[i])) {
        const binIndex = Math.min(
          bins - 1,
          Math.floor(((data[i] - min) / range) * bins)
        );
        histogram[binIndex]++;
      }
    }
    
    // Normalize
    const total = histogram.reduce((sum, val) => sum + val, 0);
    return histogram.map(val => val / total);
  }

  private calculateGradientFeatures(
    data: Float32Array,
    width: number,
    height: number
  ): {
    meanMagnitude: number;
    stdMagnitude: number;
    dominantDirection: number;
  } {
    const magnitudes: number[] = [];
    const directions: number[] = [];
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        
        // Sobel operators
        const gx = (
          data[idx + 1] - data[idx - 1] +
          2 * (data[idx + width + 1] - data[idx + width - 1]) +
          data[idx + 2 * width + 1] - data[idx + 2 * width - 1]
        ) / 8;
        
        const gy = (
          data[idx + width] - data[idx - width] +
          2 * (data[idx + width + 1] - data[idx - width + 1]) +
          data[idx + width - 1] - data[idx - width - 1]
        ) / 8;
        
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        const direction = Math.atan2(gy, gx);
        
        magnitudes.push(magnitude);
        directions.push(direction);
      }
    }
    
    const meanMagnitude = magnitudes.reduce((sum, val) => sum + val, 0) / magnitudes.length;
    const stdMagnitude = Math.sqrt(
      magnitudes.reduce((sum, val) => sum + (val - meanMagnitude) ** 2, 0) / magnitudes.length
    );
    
    // Find dominant direction using histogram
    const dirBins = 8;
    const dirHistogram = new Array(dirBins).fill(0);
    
    for (const dir of directions) {
      const binIndex = Math.floor(((dir + Math.PI) / (2 * Math.PI)) * dirBins) % dirBins;
      dirHistogram[binIndex]++;
    }
    
    const maxBin = dirHistogram.indexOf(Math.max(...dirHistogram));
    const dominantDirection = (maxBin / dirBins) * 2 * Math.PI - Math.PI;
    
    return { meanMagnitude, stdMagnitude, dominantDirection };
  }

  createBatchTensor(frames: ThermalFrame[], targetSize: number = 224): {
    data: Float32Array;
    shape: number[];
  } {
    const batchSize = frames.length;
    const tensorSize = batchSize * targetSize * targetSize;
    const batchData = new Float32Array(tensorSize);
    
    for (let b = 0; b < batchSize; b++) {
      const { temperatureData } = frames[b];
      
      // Resize to target size
      const resized = this.resize(temperatureData, 256, 192, targetSize, targetSize);
      
      // Normalize
      const normalized = this.normalize(resized);
      
      // Copy to batch tensor
      const offset = b * targetSize * targetSize;
      for (let i = 0; i < normalized.length; i++) {
        batchData[offset + i] = normalized[i];
      }
    }
    
    return {
      data: batchData,
      shape: [batchSize, targetSize, targetSize, 1],
    };
  }

  private resize(
    data: Float32Array,
    srcWidth: number,
    srcHeight: number,
    dstWidth: number,
    dstHeight: number
  ): Float32Array {
    const resized = new Float32Array(dstWidth * dstHeight);
    const xScale = srcWidth / dstWidth;
    const yScale = srcHeight / dstHeight;
    
    for (let y = 0; y < dstHeight; y++) {
      for (let x = 0; x < dstWidth; x++) {
        const srcX = x * xScale;
        const srcY = y * yScale;
        
        // Bilinear interpolation
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
        
        resized[y * dstWidth + x] = v0 * (1 - dy) + v1 * dy;
      }
    }
    
    return resized;
  }

  private normalize(data: Float32Array): Float32Array {
    const { min, max } = this.calculateStatistics(data);
    const range = max - min;
    const normalized = new Float32Array(data.length);
    
    if (range > 0) {
      for (let i = 0; i < data.length; i++) {
        normalized[i] = (data[i] - min) / range;
      }
    }
    
    return normalized;
  }
}