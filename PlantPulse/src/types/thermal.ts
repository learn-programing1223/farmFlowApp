export interface ThermalFrame {
  temperatureData: Float32Array; // 256x192 temperature values in Celsius
  timestamp: number;
  deviceId: string;
  calibrationOffset: number;
}

export interface TemperaturePoint {
  x: number;
  y: number;
  temperature: number;
}

export interface ThermalDevice {
  id: string;
  name: string;
  model: 'InfiRay_P2_Pro' | 'TOPDON_TC002C' | 'Unknown';
  resolution: {
    width: number;
    height: number;
  };
  temperatureRange: {
    min: number;
    max: number;
  };
  isConnected: boolean;
}

export interface ThermalCalibration {
  emissivity: number; // 0.0 - 1.0
  reflectedTemperature: number; // Celsius
  distance: number; // meters
  humidity: number; // percentage
  ambientTemperature: number; // Celsius
}