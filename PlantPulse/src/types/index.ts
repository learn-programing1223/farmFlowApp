export * from './thermal';
export * from './plant';
export * from './navigation';

export interface AppSettings {
  temperatureUnit: 'celsius' | 'fahrenheit';
  autoSaveAnalysis: boolean;
  frameProcessingRate: number; // FPS
  enableGPUAcceleration: boolean;
  notificationsEnabled: boolean;
  waterStressAlertThreshold: number;
  diseaseDetectionSensitivity: 'low' | 'medium' | 'high';
}

export interface Alert {
  id: string;
  type: 'water_stress' | 'disease' | 'nutrient' | 'temperature';
  severity: 'info' | 'warning' | 'critical';
  plantId: string;
  message: string;
  timestamp: number;
  isRead: boolean;
}

export interface AnalysisSession {
  id: string;
  startTime: number;
  endTime?: number;
  framesProcessed: number;
  plantsAnalyzed: string[]; // Plant IDs
  alerts: Alert[];
}