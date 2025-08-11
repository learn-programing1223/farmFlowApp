import AsyncStorage from '@react-native-async-storage/async-storage';
import { DiseaseDetectionResult } from '../ml/RGBDiseaseModel';

export interface PlantScanHistory {
  id: string;
  plantId?: string;
  plantName?: string;
  scanDate: string;
  imagePath?: string;
  result: DiseaseDetectionResult;
  notes?: string;
  location?: {
    latitude: number;
    longitude: number;
  };
  weather?: {
    temperature: number;
    humidity: number;
    conditions: string;
  };
}

export interface Plant {
  id: string;
  name: string;
  species?: string;
  dateAdded: string;
  lastScanned?: string;
  scanHistory: string[]; // Array of scan IDs
  currentHealth?: 'healthy' | 'warning' | 'critical';
  notes?: string;
  imageUri?: string;
}

const HISTORY_KEY = '@PlantPulse:scanHistory';
const PLANTS_KEY = '@PlantPulse:plants';
const MAX_HISTORY_ITEMS = 100;

class PlantHistoryService {
  /**
   * Save a new scan to history
   */
  async saveScan(scan: Omit<PlantScanHistory, 'id'>): Promise<PlantScanHistory> {
    try {
      const scanWithId: PlantScanHistory = {
        ...scan,
        id: this.generateId(),
        scanDate: scan.scanDate || new Date().toISOString()
      };

      const history = await this.getHistory();
      history.unshift(scanWithId);

      // Limit history size
      if (history.length > MAX_HISTORY_ITEMS) {
        history.splice(MAX_HISTORY_ITEMS);
      }

      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));

      // Update plant if associated
      if (scan.plantId) {
        await this.updatePlantWithScan(scan.plantId, scanWithId.id);
      }

      return scanWithId;
    } catch (error) {
      console.error('Failed to save scan:', error);
      throw error;
    }
  }

  /**
   * Get all scan history
   */
  async getHistory(): Promise<PlantScanHistory[]> {
    try {
      const historyJson = await AsyncStorage.getItem(HISTORY_KEY);
      return historyJson ? JSON.parse(historyJson) : [];
    } catch (error) {
      console.error('Failed to get history:', error);
      return [];
    }
  }

  /**
   * Get scan history for a specific plant
   */
  async getPlantHistory(plantId: string): Promise<PlantScanHistory[]> {
    const allHistory = await this.getHistory();
    return allHistory.filter(scan => scan.plantId === plantId);
  }

  /**
   * Get recent scans (last 7 days)
   */
  async getRecentScans(days: number = 7): Promise<PlantScanHistory[]> {
    const history = await this.getHistory();
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);

    return history.filter(scan => 
      new Date(scan.scanDate) > cutoffDate
    );
  }

  /**
   * Delete a scan from history
   */
  async deleteScan(scanId: string): Promise<void> {
    try {
      const history = await this.getHistory();
      const filtered = history.filter(scan => scan.id !== scanId);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Failed to delete scan:', error);
      throw error;
    }
  }

  /**
   * Clear all history
   */
  async clearHistory(): Promise<void> {
    try {
      await AsyncStorage.removeItem(HISTORY_KEY);
    } catch (error) {
      console.error('Failed to clear history:', error);
      throw error;
    }
  }

  /**
   * Save a new plant
   */
  async savePlant(plant: Omit<Plant, 'id' | 'dateAdded' | 'scanHistory'>): Promise<Plant> {
    try {
      const newPlant: Plant = {
        ...plant,
        id: this.generateId(),
        dateAdded: new Date().toISOString(),
        scanHistory: []
      };

      const plants = await this.getPlants();
      plants.push(newPlant);

      await AsyncStorage.setItem(PLANTS_KEY, JSON.stringify(plants));
      return newPlant;
    } catch (error) {
      console.error('Failed to save plant:', error);
      throw error;
    }
  }

  /**
   * Get all plants
   */
  async getPlants(): Promise<Plant[]> {
    try {
      const plantsJson = await AsyncStorage.getItem(PLANTS_KEY);
      return plantsJson ? JSON.parse(plantsJson) : [];
    } catch (error) {
      console.error('Failed to get plants:', error);
      return [];
    }
  }

  /**
   * Get a specific plant
   */
  async getPlant(plantId: string): Promise<Plant | null> {
    const plants = await this.getPlants();
    return plants.find(plant => plant.id === plantId) || null;
  }

  /**
   * Update a plant
   */
  async updatePlant(plantId: string, updates: Partial<Plant>): Promise<Plant | null> {
    try {
      const plants = await this.getPlants();
      const index = plants.findIndex(p => p.id === plantId);
      
      if (index === -1) return null;

      plants[index] = { ...plants[index], ...updates };
      await AsyncStorage.setItem(PLANTS_KEY, JSON.stringify(plants));
      
      return plants[index];
    } catch (error) {
      console.error('Failed to update plant:', error);
      throw error;
    }
  }

  /**
   * Delete a plant
   */
  async deletePlant(plantId: string): Promise<void> {
    try {
      const plants = await this.getPlants();
      const filtered = plants.filter(plant => plant.id !== plantId);
      await AsyncStorage.setItem(PLANTS_KEY, JSON.stringify(filtered));

      // Also delete associated scans
      const history = await this.getHistory();
      const filteredHistory = history.filter(scan => scan.plantId !== plantId);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(filteredHistory));
    } catch (error) {
      console.error('Failed to delete plant:', error);
      throw error;
    }
  }

  /**
   * Update plant with new scan
   */
  private async updatePlantWithScan(plantId: string, scanId: string): Promise<void> {
    const plant = await this.getPlant(plantId);
    if (!plant) return;

    const scan = (await this.getHistory()).find(s => s.id === scanId);
    if (!scan) return;

    // Determine health status based on scan
    let currentHealth: 'healthy' | 'warning' | 'critical' = 'healthy';
    if (scan.result.disease !== 'Healthy') {
      currentHealth = scan.result.severity === 'severe' ? 'critical' : 'warning';
    }

    await this.updatePlant(plantId, {
      lastScanned: scan.scanDate,
      scanHistory: [...plant.scanHistory, scanId],
      currentHealth
    });
  }

  /**
   * Get health statistics
   */
  async getHealthStatistics(): Promise<{
    totalScans: number;
    healthyScans: number;
    diseasedScans: number;
    mostCommonDisease: string | null;
    recentTrend: 'improving' | 'stable' | 'declining';
  }> {
    const history = await this.getHistory();
    
    const stats = {
      totalScans: history.length,
      healthyScans: 0,
      diseasedScans: 0,
      diseaseCount: {} as Record<string, number>,
      recentTrend: 'stable' as 'improving' | 'stable' | 'declining'
    };

    history.forEach(scan => {
      if (scan.result.disease === 'Healthy') {
        stats.healthyScans++;
      } else {
        stats.diseasedScans++;
        stats.diseaseCount[scan.result.disease] = 
          (stats.diseaseCount[scan.result.disease] || 0) + 1;
      }
    });

    // Find most common disease
    let mostCommonDisease: string | null = null;
    let maxCount = 0;
    Object.entries(stats.diseaseCount).forEach(([disease, count]) => {
      if (count > maxCount) {
        maxCount = count;
        mostCommonDisease = disease;
      }
    });

    // Analyze recent trend (last 10 scans vs previous 10)
    if (history.length >= 20) {
      const recent = history.slice(0, 10);
      const previous = history.slice(10, 20);
      
      const recentHealthy = recent.filter(s => s.result.disease === 'Healthy').length;
      const previousHealthy = previous.filter(s => s.result.disease === 'Healthy').length;
      
      if (recentHealthy > previousHealthy) {
        stats.recentTrend = 'improving';
      } else if (recentHealthy < previousHealthy) {
        stats.recentTrend = 'declining';
      }
    }

    return {
      totalScans: stats.totalScans,
      healthyScans: stats.healthyScans,
      diseasedScans: stats.diseasedScans,
      mostCommonDisease,
      recentTrend: stats.recentTrend
    };
  }

  /**
   * Export history as JSON
   */
  async exportHistory(): Promise<string> {
    const history = await this.getHistory();
    const plants = await this.getPlants();
    
    return JSON.stringify({
      exportDate: new Date().toISOString(),
      version: '1.0',
      history,
      plants
    }, null, 2);
  }

  /**
   * Import history from JSON
   */
  async importHistory(jsonData: string): Promise<void> {
    try {
      const data = JSON.parse(jsonData);
      
      if (data.history) {
        await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(data.history));
      }
      
      if (data.plants) {
        await AsyncStorage.setItem(PLANTS_KEY, JSON.stringify(data.plants));
      }
    } catch (error) {
      console.error('Failed to import history:', error);
      throw error;
    }
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }
}

// Export singleton instance
export const plantHistoryService = new PlantHistoryService();