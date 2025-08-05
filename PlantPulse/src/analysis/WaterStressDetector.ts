import { ThermalFrame } from '../types/thermal';
import { PlantProfile } from '../types/plant';

export class WaterStressDetector {
  private readonly STRESS_THRESHOLD = 0.36;
  private readonly BASELINE_BUFFER_SIZE = 10;
  private baselineTemperatures: Map<string, number[]> = new Map();

  calculateCWSI(
    leafTemp: number,
    airTemp: number,
    vpd: number // Vapor Pressure Deficit in kPa
  ): number {
    // Theoretical minimum temperature (well-watered plant)
    const wetLeafTemp = airTemp - (2.0 * vpd);
    
    // Theoretical maximum temperature (water-stressed plant)
    const dryLeafTemp = airTemp + 5.0;
    
    // Crop Water Stress Index calculation
    const cwsi = (leafTemp - wetLeafTemp) / (dryLeafTemp - wetLeafTemp);
    
    // Clamp between 0 and 1
    return Math.max(0, Math.min(1, cwsi));
  }

  detectWaterStress(
    thermalFrame: ThermalFrame,
    plant: PlantProfile,
    ambientTemp: number,
    humidity: number
  ): {
    cwsi: number;
    stressLevel: 'none' | 'mild' | 'moderate' | 'severe';
    leafTemperature: number;
    temperatureDelta: number;
  } {
    // Extract leaf temperatures from thermal data
    const leafTemps = this.extractLeafTemperatures(thermalFrame);
    const avgLeafTemp = this.calculateAverageLeafTemperature(leafTemps);
    
    // Calculate VPD from ambient conditions
    const vpd = this.calculateVPD(ambientTemp, humidity);
    
    // Calculate CWSI
    const cwsi = this.calculateCWSI(avgLeafTemp, ambientTemp, vpd);
    
    // Update baseline temperatures for this plant
    this.updateBaseline(plant.id, avgLeafTemp);
    
    // Calculate temperature delta from baseline
    const baseline = this.getBaselineTemperature(plant.id);
    const temperatureDelta = avgLeafTemp - baseline;
    
    // Determine stress level
    const stressLevel = this.categorizeStressLevel(cwsi, temperatureDelta);
    
    return {
      cwsi,
      stressLevel,
      leafTemperature: avgLeafTemp,
      temperatureDelta,
    };
  }

  private extractLeafTemperatures(thermalFrame: ThermalFrame): number[] {
    const temps: number[] = [];
    const { temperatureData } = thermalFrame;
    const width = 256;
    const height = 192;
    
    // Focus on central region where plant is likely to be
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 3;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dist = Math.sqrt(
          Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2)
        );
        
        if (dist <= radius) {
          const index = y * width + x;
          const temp = temperatureData[index];
          
          // Filter out obvious non-leaf temperatures
          if (temp > 10 && temp < 50 && !isNaN(temp)) {
            temps.push(temp);
          }
        }
      }
    }
    
    return temps;
  }

  private calculateAverageLeafTemperature(temperatures: number[]): number {
    if (temperatures.length === 0) return 20; // Default room temperature
    
    // Remove outliers using IQR method
    const sorted = [...temperatures].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    
    const filtered = temperatures.filter(
      temp => temp >= q1 - 1.5 * iqr && temp <= q3 + 1.5 * iqr
    );
    
    return filtered.reduce((sum, temp) => sum + temp, 0) / filtered.length;
  }

  private calculateVPD(temperature: number, humidity: number): number {
    // Saturation vapor pressure (kPa) using Magnus formula
    const svp = 0.6108 * Math.exp((17.27 * temperature) / (temperature + 237.3));
    
    // Actual vapor pressure
    const avp = (humidity / 100) * svp;
    
    // Vapor Pressure Deficit
    return svp - avp;
  }

  private updateBaseline(plantId: string, temperature: number): void {
    if (!this.baselineTemperatures.has(plantId)) {
      this.baselineTemperatures.set(plantId, []);
    }
    
    const baseline = this.baselineTemperatures.get(plantId)!;
    baseline.push(temperature);
    
    // Keep only recent measurements
    if (baseline.length > this.BASELINE_BUFFER_SIZE) {
      baseline.shift();
    }
  }

  private getBaselineTemperature(plantId: string): number {
    const baseline = this.baselineTemperatures.get(plantId);
    
    if (!baseline || baseline.length === 0) {
      return 20; // Default baseline
    }
    
    return baseline.reduce((sum, temp) => sum + temp, 0) / baseline.length;
  }

  private categorizeStressLevel(
    cwsi: number,
    temperatureDelta: number
  ): 'none' | 'mild' | 'moderate' | 'severe' {
    if (cwsi < 0.2 && temperatureDelta < 1.5) {
      return 'none';
    } else if (cwsi < 0.36 && temperatureDelta < 3) {
      return 'mild';
    } else if (cwsi < 0.6 && temperatureDelta < 5) {
      return 'moderate';
    } else {
      return 'severe';
    }
  }

  detectStomatalClosure(
    thermalFrames: ThermalFrame[],
    timeWindow: number = 300000 // 5 minutes
  ): boolean {
    if (thermalFrames.length < 2) return false;
    
    // Sort frames by timestamp
    const sorted = [...thermalFrames].sort((a, b) => a.timestamp - b.timestamp);
    
    // Get frames within time window
    const recentFrames = sorted.filter(
      frame => frame.timestamp > Date.now() - timeWindow
    );
    
    if (recentFrames.length < 2) return false;
    
    // Calculate temperature trend
    const temps = recentFrames.map(frame => {
      const leafTemps = this.extractLeafTemperatures(frame);
      return this.calculateAverageLeafTemperature(leafTemps);
    });
    
    // Check for rapid temperature increase (indicates stomatal closure)
    const tempIncrease = temps[temps.length - 1] - temps[0];
    const timeElapsed = (
      recentFrames[recentFrames.length - 1].timestamp - 
      recentFrames[0].timestamp
    ) / 60000; // Convert to minutes
    
    const rateOfChange = tempIncrease / timeElapsed;
    
    // Stomatal closure typically causes >1°C/min increase
    return rateOfChange > 1.0;
  }

  generateWaterStressMap(
    thermalFrame: ThermalFrame,
    resolution: number = 16
  ): Float32Array {
    const { temperatureData } = thermalFrame;
    const width = 256;
    const height = 192;
    const mapWidth = Math.floor(width / resolution);
    const mapHeight = Math.floor(height / resolution);
    const stressMap = new Float32Array(mapWidth * mapHeight);
    
    // Calculate ambient temperature (edges of frame)
    const ambientTemp = this.estimateAmbientTemperature(temperatureData, width, height);
    
    for (let my = 0; my < mapHeight; my++) {
      for (let mx = 0; mx < mapWidth; mx++) {
        // Sample temperature in this region
        let sumTemp = 0;
        let count = 0;
        
        for (let dy = 0; dy < resolution; dy++) {
          for (let dx = 0; dx < resolution; dx++) {
            const x = mx * resolution + dx;
            const y = my * resolution + dy;
            
            if (x < width && y < height) {
              const index = y * width + x;
              sumTemp += temperatureData[index];
              count++;
            }
          }
        }
        
        const avgTemp = sumTemp / count;
        
        // Simple stress calculation based on temperature differential
        const tempDiff = avgTemp - ambientTemp;
        const stress = Math.max(0, Math.min(1, (tempDiff + 6) / 12)); // Normalize to 0-1
        
        stressMap[my * mapWidth + mx] = stress;
      }
    }
    
    return stressMap;
  }

  private estimateAmbientTemperature(
    temperatureData: Float32Array,
    width: number,
    height: number
  ): number {
    const edgeTemps: number[] = [];
    const margin = 10;
    
    // Sample temperatures from frame edges
    for (let x = 0; x < width; x++) {
      // Top edge
      if (x < margin || x >= width - margin) {
        for (let y = 0; y < margin; y++) {
          edgeTemps.push(temperatureData[y * width + x]);
        }
      }
      
      // Bottom edge
      if (x < margin || x >= width - margin) {
        for (let y = height - margin; y < height; y++) {
          edgeTemps.push(temperatureData[y * width + x]);
        }
      }
    }
    
    // Calculate median of edge temperatures
    const sorted = edgeTemps.filter(t => !isNaN(t)).sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)] || 20;
  }
}