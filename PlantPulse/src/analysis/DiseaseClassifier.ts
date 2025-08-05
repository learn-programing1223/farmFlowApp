import { ThermalFrame } from '../types/thermal';
import { DiseasePattern } from '../types/plant';

export class DiseaseClassifier {
  private readonly PATTERN_DETECTION_THRESHOLD = 2.0; // °C difference
  
  detectDiseasePatterns(
    thermalFrame: ThermalFrame,
    healthyBaselineTemp: number
  ): {
    diseaseType: 'healthy' | 'bacterial' | 'fungal' | 'viral';
    confidence: number;
    patterns: DiseasePattern[];
    affectedArea: number;
  } {
    const { temperatureData } = thermalFrame;
    const width = 256;
    const height = 192;
    
    // Detect temperature anomalies
    const anomalyMap = this.createAnomalyMap(
      temperatureData,
      width,
      height,
      healthyBaselineTemp
    );
    
    // Find patterns in anomaly map
    const patterns = this.findPatterns(anomalyMap, width, height);
    
    // Classify disease based on patterns
    const classification = this.classifyDisease(patterns, anomalyMap, width, height);
    
    // Calculate affected area percentage
    const affectedArea = this.calculateAffectedArea(anomalyMap);
    
    return {
      diseaseType: classification.type,
      confidence: classification.confidence,
      patterns,
      affectedArea,
    };
  }

  private createAnomalyMap(
    temperatureData: Float32Array,
    width: number,
    height: number,
    baseline: number
  ): Float32Array {
    const anomalyMap = new Float32Array(width * height);
    
    for (let i = 0; i < temperatureData.length; i++) {
      const tempDiff = temperatureData[i] - baseline;
      anomalyMap[i] = tempDiff;
    }
    
    return anomalyMap;
  }

  private findPatterns(
    anomalyMap: Float32Array,
    width: number,
    height: number
  ): DiseasePattern[] {
    const patterns: DiseasePattern[] = [];
    const visited = new Set<number>();
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        
        if (visited.has(index)) continue;
        
        const anomaly = anomalyMap[index];
        if (Math.abs(anomaly) < this.PATTERN_DETECTION_THRESHOLD) continue;
        
        // Detect pattern type at this location
        const pattern = this.detectPatternType(
          anomalyMap,
          x,
          y,
          width,
          height,
          visited
        );
        
        if (pattern) {
          patterns.push(pattern);
        }
      }
    }
    
    return patterns;
  }

  private detectPatternType(
    anomalyMap: Float32Array,
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Set<number>
  ): DiseasePattern | null {
    const startIndex = startY * width + startX;
    const startAnomaly = anomalyMap[startIndex];
    
    // Check for circular pattern (common in fungal infections)
    const circular = this.checkCircularPattern(
      anomalyMap,
      startX,
      startY,
      width,
      height,
      visited
    );
    
    if (circular) {
      return {
        type: 'circular',
        location: circular.location,
        temperatureDelta: startAnomaly,
      };
    }
    
    // Check for linear pattern (common in bacterial infections)
    const linear = this.checkLinearPattern(
      anomalyMap,
      startX,
      startY,
      width,
      height,
      visited
    );
    
    if (linear) {
      return {
        type: 'linear',
        location: linear.location,
        temperatureDelta: startAnomaly,
      };
    }
    
    // Check for mosaic pattern (common in viral infections)
    const mosaic = this.checkMosaicPattern(
      anomalyMap,
      startX,
      startY,
      width,
      height,
      visited
    );
    
    if (mosaic) {
      return {
        type: 'mosaic',
        location: mosaic.location,
        temperatureDelta: startAnomaly,
      };
    }
    
    // Default to irregular pattern
    return {
      type: 'irregular',
      location: { x: startX, y: startY, radius: 5 },
      temperatureDelta: startAnomaly,
    };
  }

  private checkCircularPattern(
    anomalyMap: Float32Array,
    centerX: number,
    centerY: number,
    width: number,
    height: number,
    visited: Set<number>
  ): { location: { x: number; y: number; radius: number } } | null {
    const centerAnomaly = anomalyMap[centerY * width + centerX];
    let maxRadius = 0;
    
    // Check increasing radii
    for (let radius = 1; radius < 50; radius++) {
      let matchCount = 0;
      let totalCount = 0;
      
      // Sample points on circle
      const samples = radius * 8;
      for (let i = 0; i < samples; i++) {
        const angle = (i / samples) * Math.PI * 2;
        const x = Math.round(centerX + radius * Math.cos(angle));
        const y = Math.round(centerY + radius * Math.sin(angle));
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
          totalCount++;
          const index = y * width + x;
          const anomaly = anomalyMap[index];
          
          // Check if temperature anomaly is similar
          if (Math.abs(anomaly - centerAnomaly) < 1.0) {
            matchCount++;
            visited.add(index);
          }
        }
      }
      
      // If >70% of points match, consider it part of circle
      if (totalCount > 0 && matchCount / totalCount > 0.7) {
        maxRadius = radius;
      } else if (maxRadius > 5) {
        // Circle ended, return result
        return {
          location: { x: centerX, y: centerY, radius: maxRadius },
        };
      }
    }
    
    return null;
  }

  private checkLinearPattern(
    anomalyMap: Float32Array,
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Set<number>
  ): { location: { x: number; y: number; radius: number } } | null {
    const startAnomaly = anomalyMap[startY * width + startX];
    
    // Check 8 directions
    const directions = [
      [0, 1], [1, 0], [0, -1], [-1, 0],
      [1, 1], [-1, -1], [1, -1], [-1, 1],
    ];
    
    let bestDirection: number[] | null = null;
    let bestLength = 0;
    
    for (const [dx, dy] of directions) {
      let length = 0;
      let x = startX;
      let y = startY;
      
      while (x >= 0 && x < width && y >= 0 && y < height) {
        const index = y * width + x;
        const anomaly = anomalyMap[index];
        
        if (Math.abs(anomaly - startAnomaly) > 1.5) break;
        
        visited.add(index);
        length++;
        x += dx;
        y += dy;
      }
      
      if (length > bestLength) {
        bestLength = length;
        bestDirection = [dx, dy];
      }
    }
    
    if (bestLength > 10) {
      return {
        location: { x: startX, y: startY, radius: bestLength / 2 },
      };
    }
    
    return null;
  }

  private checkMosaicPattern(
    anomalyMap: Float32Array,
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Set<number>
  ): { location: { x: number; y: number; radius: number } } | null {
    const patchSize = 5;
    let patchCount = 0;
    let totalVariance = 0;
    
    // Check surrounding area for mosaic-like variance
    for (let dy = -20; dy <= 20; dy += patchSize) {
      for (let dx = -20; dx <= 20; dx += patchSize) {
        const px = startX + dx;
        const py = startY + dy;
        
        if (px >= 0 && px < width - patchSize && 
            py >= 0 && py < height - patchSize) {
          
          // Calculate variance in this patch
          const temps: number[] = [];
          for (let y = 0; y < patchSize; y++) {
            for (let x = 0; x < patchSize; x++) {
              const index = (py + y) * width + (px + x);
              temps.push(anomalyMap[index]);
              visited.add(index);
            }
          }
          
          const variance = this.calculateVariance(temps);
          if (variance > 2.0) {
            patchCount++;
            totalVariance += variance;
          }
        }
      }
    }
    
    // Mosaic pattern has high variance across patches
    if (patchCount > 5 && totalVariance / patchCount > 3.0) {
      return {
        location: { x: startX, y: startY, radius: 20 },
      };
    }
    
    return null;
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private classifyDisease(
    patterns: DiseasePattern[],
    anomalyMap: Float32Array,
    width: number,
    height: number
  ): { type: 'healthy' | 'bacterial' | 'fungal' | 'viral'; confidence: number } {
    if (patterns.length === 0) {
      return { type: 'healthy', confidence: 0.9 };
    }
    
    // Count pattern types
    const patternCounts = {
      circular: 0,
      linear: 0,
      mosaic: 0,
      irregular: 0,
    };
    
    let totalTempDelta = 0;
    
    for (const pattern of patterns) {
      patternCounts[pattern.type]++;
      totalTempDelta += Math.abs(pattern.temperatureDelta);
    }
    
    const avgTempDelta = totalTempDelta / patterns.length;
    
    // Fungal infections: circular patterns, moderate temperature changes
    if (patternCounts.circular > patterns.length * 0.5) {
      const confidence = Math.min(0.95, 0.6 + (patternCounts.circular / patterns.length) * 0.35);
      
      // Biotrophic vs necrotrophic based on temperature
      if (avgTempDelta > -2.5 && avgTempDelta < 2) {
        return { type: 'fungal', confidence };
      } else if (avgTempDelta > 2 && avgTempDelta < 9) {
        return { type: 'fungal', confidence: confidence * 0.9 };
      }
    }
    
    // Bacterial infections: linear patterns, initial cooling
    if (patternCounts.linear > patterns.length * 0.4) {
      const confidence = Math.min(0.9, 0.5 + (patternCounts.linear / patterns.length) * 0.4);
      
      if (avgTempDelta < 0) {
        return { type: 'bacterial', confidence };
      }
    }
    
    // Viral infections: mosaic patterns, variable temperature
    if (patternCounts.mosaic > patterns.length * 0.3) {
      const confidence = Math.min(0.85, 0.5 + (patternCounts.mosaic / patterns.length) * 0.35);
      return { type: 'viral', confidence };
    }
    
    // Default classification based on temperature patterns
    if (avgTempDelta < -1) {
      return { type: 'bacterial', confidence: 0.6 };
    } else if (avgTempDelta > 2) {
      return { type: 'fungal', confidence: 0.6 };
    }
    
    return { type: 'healthy', confidence: 0.7 };
  }

  private calculateAffectedArea(anomalyMap: Float32Array): number {
    let affectedPixels = 0;
    
    for (let i = 0; i < anomalyMap.length; i++) {
      if (Math.abs(anomalyMap[i]) > this.PATTERN_DETECTION_THRESHOLD) {
        affectedPixels++;
      }
    }
    
    return (affectedPixels / anomalyMap.length) * 100;
  }

  trackDiseaseProgression(
    currentFrame: ThermalFrame,
    previousFrames: ThermalFrame[],
    healthyBaseline: number
  ): {
    spreadRate: number; // pixels per minute
    direction: { x: number; y: number }; // normalized vector
    severity: 'stable' | 'spreading' | 'aggressive';
  } {
    if (previousFrames.length === 0) {
      return {
        spreadRate: 0,
        direction: { x: 0, y: 0 },
        severity: 'stable',
      };
    }
    
    // Get disease patterns from current and previous frames
    const currentPatterns = this.detectDiseasePatterns(currentFrame, healthyBaseline);
    const previousPatterns = this.detectDiseasePatterns(
      previousFrames[previousFrames.length - 1],
      healthyBaseline
    );
    
    // Calculate spread metrics
    const areaIncrease = currentPatterns.affectedArea - previousPatterns.affectedArea;
    const timeDelta = (currentFrame.timestamp - previousFrames[previousFrames.length - 1].timestamp) / 60000; // minutes
    
    const spreadRate = Math.abs(areaIncrease / timeDelta);
    
    // Determine severity
    let severity: 'stable' | 'spreading' | 'aggressive';
    if (spreadRate < 0.5) {
      severity = 'stable';
    } else if (spreadRate < 2.0) {
      severity = 'spreading';
    } else {
      severity = 'aggressive';
    }
    
    // Calculate spread direction (simplified)
    const direction = this.calculateSpreadDirection(
      currentPatterns.patterns,
      previousPatterns.patterns
    );
    
    return { spreadRate, direction, severity };
  }

  private calculateSpreadDirection(
    currentPatterns: DiseasePattern[],
    previousPatterns: DiseasePattern[]
  ): { x: number; y: number } {
    if (currentPatterns.length === 0 || previousPatterns.length === 0) {
      return { x: 0, y: 0 };
    }
    
    // Calculate center of mass for current and previous patterns
    const currentCenter = this.calculatePatternCenter(currentPatterns);
    const previousCenter = this.calculatePatternCenter(previousPatterns);
    
    // Calculate direction vector
    const dx = currentCenter.x - previousCenter.x;
    const dy = currentCenter.y - previousCenter.y;
    
    // Normalize
    const magnitude = Math.sqrt(dx * dx + dy * dy);
    if (magnitude === 0) {
      return { x: 0, y: 0 };
    }
    
    return {
      x: dx / magnitude,
      y: dy / magnitude,
    };
  }

  private calculatePatternCenter(
    patterns: DiseasePattern[]
  ): { x: number; y: number } {
    if (patterns.length === 0) {
      return { x: 0, y: 0 };
    }
    
    const sumX = patterns.reduce((sum, p) => sum + p.location.x, 0);
    const sumY = patterns.reduce((sum, p) => sum + p.location.y, 0);
    
    return {
      x: sumX / patterns.length,
      y: sumY / patterns.length,
    };
  }
}