export interface PlantHealthAnalysis {
  id: string;
  waterStressIndex: number; // 0-1 scale, >0.36 indicates stress
  stressLevel: 'none' | 'mild' | 'moderate' | 'severe';
  diseaseDetection: {
    type: 'healthy' | 'bacterial' | 'fungal' | 'viral';
    confidence: number;
    affectedArea: number; // percentage
    patterns: DiseasePattern[];
  };
  nutrientStatus: {
    nitrogen: NutrientLevel;
    phosphorus: NutrientLevel;
    potassium: NutrientLevel;
  };
  recommendations: string[];
  timestamp: number;
  thermalFrameId?: string;
}

export type NutrientLevel = 'deficient' | 'optimal' | 'excess';

export interface DiseasePattern {
  type: 'circular' | 'linear' | 'mosaic' | 'irregular';
  location: {
    x: number;
    y: number;
    radius: number;
  };
  temperatureDelta: number; // Temperature difference from healthy tissue
}

export interface PlantProfile {
  id: string;
  species: string;
  nickname: string;
  optimalTempRange: {
    min: number;
    max: number;
  };
  waterStressThreshold: number;
  lastAnalysis?: PlantHealthAnalysis;
  history: PlantHealthAnalysis[];
  createdAt: number;
  updatedAt: number;
  imageUri?: string;
}

export interface PlantSpecies {
  id: string;
  scientificName: string;
  commonName: string;
  optimalTemperature: {
    min: number;
    max: number;
  };
  optimalHumidity: {
    min: number;
    max: number;
  };
  waterStressThreshold: number;
  diseaseResistance: {
    bacterial: number; // 0-1
    fungal: number;
    viral: number;
  };
}