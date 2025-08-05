import { ThermalCalibration } from '../types/thermal';

export class TemperatureParser {
  private readonly FRAME_WIDTH = 256;
  private readonly FRAME_HEIGHT = 384;
  private readonly TEMP_DATA_HEIGHT = 192;
  
  parseFrame(
    rawData: Uint8Array,
    deviceModel: string,
    calibration: ThermalCalibration
  ): Float32Array {
    const temperatureData = new Float32Array(this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT);
    
    switch (deviceModel) {
      case 'InfiRay_P2_Pro':
        return this.parseInfiRayFrame(rawData, calibration);
      case 'TOPDON_TC002C':
        return this.parseTopdonFrame(rawData, calibration);
      default:
        return this.parseGenericFrame(rawData, calibration);
    }
  }

  private parseInfiRayFrame(
    rawData: Uint8Array,
    calibration: ThermalCalibration
  ): Float32Array {
    const temperatureData = new Float32Array(this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT);
    
    // InfiRay P2 Pro specific parsing
    // Temperature data is in the bottom portion of the frame
    const tempDataOffset = this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT * 2; // Skip visual data
    
    for (let y = 0; y < this.TEMP_DATA_HEIGHT; y++) {
      for (let x = 0; x < this.FRAME_WIDTH; x++) {
        const pixelIndex = y * this.FRAME_WIDTH + x;
        const dataIndex = tempDataOffset + (pixelIndex * 2);
        
        // 16-bit temperature value
        const rawTemp = (rawData[dataIndex + 1] << 8) | rawData[dataIndex];
        
        // Convert to Celsius
        const tempKelvin = rawTemp / 10.0;
        const tempCelsius = tempKelvin - 273.15;
        
        // Apply calibration
        temperatureData[pixelIndex] = this.applyCalibration(
          tempCelsius,
          calibration
        );
      }
    }
    
    return temperatureData;
  }

  private parseTopdonFrame(
    rawData: Uint8Array,
    calibration: ThermalCalibration
  ): Float32Array {
    const temperatureData = new Float32Array(this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT);
    
    // TOPDON TC002C specific parsing
    // Similar format but different temperature encoding
    const tempDataOffset = this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT * 2;
    
    for (let y = 0; y < this.TEMP_DATA_HEIGHT; y++) {
      for (let x = 0; x < this.FRAME_WIDTH; x++) {
        const pixelIndex = y * this.FRAME_WIDTH + x;
        const dataIndex = tempDataOffset + (pixelIndex * 2);
        
        // 16-bit temperature value with different scaling
        const rawTemp = (rawData[dataIndex + 1] << 8) | rawData[dataIndex];
        
        // TOPDON uses different scaling factor
        const tempCelsius = (rawTemp / 100.0) - 40.0;
        
        temperatureData[pixelIndex] = this.applyCalibration(
          tempCelsius,
          calibration
        );
      }
    }
    
    return temperatureData;
  }

  private parseGenericFrame(
    rawData: Uint8Array,
    calibration: ThermalCalibration
  ): Float32Array {
    const temperatureData = new Float32Array(this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT);
    
    // Generic UVC thermal camera parsing
    const tempDataOffset = this.FRAME_WIDTH * this.TEMP_DATA_HEIGHT * 2;
    
    for (let y = 0; y < this.TEMP_DATA_HEIGHT; y++) {
      for (let x = 0; x < this.FRAME_WIDTH; x++) {
        const pixelIndex = y * this.FRAME_WIDTH + x;
        const dataIndex = tempDataOffset + (pixelIndex * 2);
        
        const rawTemp = (rawData[dataIndex + 1] << 8) | rawData[dataIndex];
        const tempKelvin = rawTemp / 10.0;
        const tempCelsius = tempKelvin - 273.15;
        
        temperatureData[pixelIndex] = this.applyCalibration(
          tempCelsius,
          calibration
        );
      }
    }
    
    return temperatureData;
  }

  private applyCalibration(
    temperature: number,
    calibration: ThermalCalibration
  ): number {
    // Apply emissivity correction
    const emissivityCorrected = temperature / calibration.emissivity;
    
    // Apply reflected temperature compensation
    const reflectedComponent = 
      (1 - calibration.emissivity) * calibration.reflectedTemperature;
    
    // Apply atmospheric transmission correction based on distance
    const atmosphericTransmission = this.calculateAtmosphericTransmission(
      calibration.distance,
      calibration.humidity,
      calibration.ambientTemperature
    );
    
    const correctedTemp = 
      (emissivityCorrected - reflectedComponent) / atmosphericTransmission;
    
    return correctedTemp;
  }

  private calculateAtmosphericTransmission(
    distance: number,
    humidity: number,
    ambientTemp: number
  ): number {
    // Simplified atmospheric transmission calculation
    // In reality, this would use more complex models
    const waterVaporDensity = this.calculateWaterVaporDensity(humidity, ambientTemp);
    const attenuationCoeff = 0.006 + (0.002 * waterVaporDensity);
    
    return Math.exp(-attenuationCoeff * distance);
  }

  private calculateWaterVaporDensity(
    humidity: number,
    temperature: number
  ): number {
    // Magnus formula for saturation vapor pressure
    const a = 17.27;
    const b = 237.7;
    const saturationPressure = 
      0.61078 * Math.exp((a * temperature) / (b + temperature));
    
    // Actual vapor pressure
    const actualPressure = (humidity / 100) * saturationPressure;
    
    // Water vapor density in g/m³
    const R = 461.5; // Specific gas constant for water vapor
    const tempKelvin = temperature + 273.15;
    
    return (actualPressure * 1000) / (R * tempKelvin);
  }

  extractSpotTemperature(
    temperatureData: Float32Array,
    x: number,
    y: number,
    radius: number = 3
  ): number {
    const temps: number[] = [];
    
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = Math.max(0, Math.min(this.FRAME_WIDTH - 1, x + dx));
        const py = Math.max(0, Math.min(this.TEMP_DATA_HEIGHT - 1, y + dy));
        
        if (dx * dx + dy * dy <= radius * radius) {
          temps.push(temperatureData[py * this.FRAME_WIDTH + px]);
        }
      }
    }
    
    // Return average temperature in the spot
    return temps.reduce((a, b) => a + b, 0) / temps.length;
  }

  findHotspots(
    temperatureData: Float32Array,
    threshold: number = 5
  ): Array<{ x: number; y: number; temperature: number }> {
    const hotspots: Array<{ x: number; y: number; temperature: number }> = [];
    const avgTemp = this.calculateAverageTemperature(temperatureData);
    
    for (let y = 0; y < this.TEMP_DATA_HEIGHT; y++) {
      for (let x = 0; x < this.FRAME_WIDTH; x++) {
        const index = y * this.FRAME_WIDTH + x;
        const temp = temperatureData[index];
        
        if (temp > avgTemp + threshold) {
          hotspots.push({ x, y, temperature: temp });
        }
      }
    }
    
    return this.clusterHotspots(hotspots);
  }

  private calculateAverageTemperature(temperatureData: Float32Array): number {
    let sum = 0;
    let count = 0;
    
    for (let i = 0; i < temperatureData.length; i++) {
      if (!isNaN(temperatureData[i]) && isFinite(temperatureData[i])) {
        sum += temperatureData[i];
        count++;
      }
    }
    
    return count > 0 ? sum / count : 0;
  }

  private clusterHotspots(
    hotspots: Array<{ x: number; y: number; temperature: number }>
  ): Array<{ x: number; y: number; temperature: number }> {
    // Simple clustering to merge nearby hotspots
    const clustered: Array<{ x: number; y: number; temperature: number }> = [];
    const visited = new Set<number>();
    
    for (let i = 0; i < hotspots.length; i++) {
      if (visited.has(i)) continue;
      
      const cluster = [hotspots[i]];
      visited.add(i);
      
      for (let j = i + 1; j < hotspots.length; j++) {
        if (visited.has(j)) continue;
        
        const dist = Math.sqrt(
          Math.pow(hotspots[i].x - hotspots[j].x, 2) +
          Math.pow(hotspots[i].y - hotspots[j].y, 2)
        );
        
        if (dist < 10) {
          cluster.push(hotspots[j]);
          visited.add(j);
        }
      }
      
      // Calculate cluster center
      const centerX = cluster.reduce((sum, p) => sum + p.x, 0) / cluster.length;
      const centerY = cluster.reduce((sum, p) => sum + p.y, 0) / cluster.length;
      const maxTemp = Math.max(...cluster.map(p => p.temperature));
      
      clustered.push({
        x: Math.round(centerX),
        y: Math.round(centerY),
        temperature: maxTemp,
      });
    }
    
    return clustered;
  }
}