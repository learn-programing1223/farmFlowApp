import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CameraScreenNavigationProp } from '../../types';
import { ThermalCameraManager } from '../../camera/ThermalCameraManager';
import { ThermalFrame } from '../../types/thermal';
import { PlantHealthModel } from '../../ml/PlantHealthModel';
import { WaterStressDetector } from '../../analysis/WaterStressDetector';
import { DiseaseClassifier } from '../../analysis/DiseaseClassifier';
import { NutrientAnalyzer } from '../../analysis/NutrientAnalyzer';
import ThermalOverlay from '../components/ThermalOverlay';
import TemperatureSpotMeter from '../components/TemperatureSpotMeter';

const CameraScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();
  const [isConnected, setIsConnected] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTemp, setCurrentTemp] = useState<number | null>(null);
  const [currentFrame, setCurrentFrame] = useState<ThermalFrame | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  
  const cameraManager = useRef<ThermalCameraManager>(ThermalCameraManager.getInstance());
  const mlModel = useRef<PlantHealthModel>(new PlantHealthModel());
  const waterStressDetector = useRef<WaterStressDetector>(new WaterStressDetector());
  const diseaseClassifier = useRef<DiseaseClassifier>(new DiseaseClassifier());
  const nutrientAnalyzer = useRef<NutrientAnalyzer>(new NutrientAnalyzer());

  useEffect(() => {
    initializeCamera();
    return () => {
      cameraManager.current.dispose();
      mlModel.current.dispose();
    };
  }, []);

  const initializeCamera = async () => {
    try {
      setIsLoading(true);
      const initialized = await cameraManager.current.initialize();
      if (initialized) {
        await mlModel.current.initialize();
        checkForConnectedCamera();
      }
    } catch (error) {
      console.error('Failed to initialize camera:', error);
      Alert.alert('Error', 'Failed to initialize thermal camera system');
    } finally {
      setIsLoading(false);
    }
  };

  const checkForConnectedCamera = async () => {
    const devices = await cameraManager.current.scanForDevices();
    if (devices.length > 0) {
      await cameraManager.current.connectToDevice(devices[0]);
      setIsConnected(true);
    }
  };

  const handleConnect = async () => {
    try {
      setIsLoading(true);
      const devices = await cameraManager.current.scanForDevices();
      
      if (devices.length === 0) {
        Alert.alert(
          'No Camera Found',
          'Please connect your InfiRay P2 Pro or TOPDON TC002C via USB-C',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Retry', onPress: handleConnect },
          ],
        );
        return;
      }

      await cameraManager.current.connectToDevice(devices[0]);
      setIsConnected(true);
      Alert.alert('Success', `Connected to ${devices[0].name}`);
    } catch (error) {
      console.error('Failed to connect camera:', error);
      Alert.alert('Connection Failed', 'Unable to connect to thermal camera');
    } finally {
      setIsLoading(false);
    }
  };

  const handleScan = async () => {
    if (!isConnected) {
      Alert.alert('No Camera', 'Please connect a thermal camera first');
      return;
    }
    
    if (isScanning) {
      await cameraManager.current.stopCapture();
      setIsScanning(false);
    } else {
      setIsScanning(true);
      try {
        await cameraManager.current.startCapture(handleThermalFrame);
      } catch (error) {
        console.error('Failed to start capture:', error);
        Alert.alert('Error', 'Failed to start thermal capture');
        setIsScanning(false);
      }
    }
  };

  const handleThermalFrame = async (frame: ThermalFrame) => {
    setCurrentFrame(frame);
    
    // Extract center temperature for display
    const centerTemp = frame.temperatureData[
      Math.floor(frame.temperatureData.length / 2)
    ];
    setCurrentTemp(centerTemp);
    
    // Run ML analysis
    try {
      const analysis = await mlModel.current.analyze(frame);
      setAnalysisResult(analysis);
      
      // Check for alerts
      if (analysis.stressLevel === 'severe' || 
          analysis.diseaseDetection.type !== 'healthy') {
        // Navigate to analysis screen with results
        navigation.navigate('Analysis', { analysis });
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const handleCalibrate = () => {
    navigation.navigate('Calibration');
  };

  return (
    <View style={styles.container}>
      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Initializing thermal camera...</Text>
        </View>
      )}
      
      <View style={styles.cameraView}>
        {!isConnected ? (
          <View style={styles.placeholder}>
            <Icon name="camera-off" size={80} color="#ccc" />
            <Text style={styles.placeholderText}>No thermal camera connected</Text>
          </View>
        ) : (
          <View style={styles.thermalView}>
            {currentFrame ? (
              <View style={styles.thermalImageContainer}>
                <ThermalOverlay
                  thermalFrame={currentFrame}
                  colorMap="iron"
                  showHotspots={true}
                  showGrid={false}
                />
                <TemperatureSpotMeter
                  thermalFrame={currentFrame}
                  onTemperatureRead={(temp) => setCurrentTemp(temp)}
                />
              </View>
            ) : (
              <View style={styles.waitingForData}>
                <ActivityIndicator size="large" color="#4CAF50" />
                <Text style={styles.waitingText}>Waiting for thermal data...</Text>
              </View>
            )}
            
            {isScanning && analysisResult && (
              <View style={styles.scanOverlay}>
                <View style={styles.analysisInfo}>
                  <View style={[
                    styles.stressIndicator,
                    { backgroundColor: getStressColor(analysisResult.stressLevel) }
                  ]} />
                  <Text style={styles.scanText}>
                    Water Stress: {analysisResult.stressLevel}
                  </Text>
                </View>
                {analysisResult.diseaseDetection.type !== 'healthy' && (
                  <Text style={styles.alertText}>
                    ⚠️ {analysisResult.diseaseDetection.type} detected
                  </Text>
                )}
              </View>
            )}
          </View>
        )}
      </View>

      <View style={styles.controls}>
        {!isConnected ? (
          <TouchableOpacity style={styles.button} onPress={handleConnect}>
            <Icon name="usb" size={24} color="#fff" />
            <Text style={styles.buttonText}>Connect Camera</Text>
          </TouchableOpacity>
        ) : (
          <>
            <TouchableOpacity
              style={[styles.button, isScanning && styles.buttonActive]}
              onPress={handleScan}>
              <Icon name={isScanning ? 'stop' : 'scan-helper'} size={24} color="#fff" />
              <Text style={styles.buttonText}>
                {isScanning ? 'Stop Scan' : 'Start Scan'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, styles.secondaryButton]}
              onPress={handleCalibrate}>
              <Icon name="tune" size={24} color="#4CAF50" />
              <Text style={[styles.buttonText, styles.secondaryButtonText]}>
                Calibrate
              </Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      <View style={styles.info}>
        <View style={styles.infoRow}>
          <Icon name="thermometer" size={20} color="#666" />
          <Text style={styles.infoText}>
            Temperature: {currentTemp ? `${currentTemp.toFixed(1)}°C` : '--°C'}
          </Text>
        </View>
        <View style={styles.infoRow}>
          <Icon name="water-percent" size={20} color="#666" />
          <Text style={styles.infoText}>
            CWSI: {analysisResult ? 
              `${(analysisResult.waterStressIndex * 100).toFixed(0)}%` : '--%'}
          </Text>
        </View>
        {cameraManager.current.getCurrentDevice() && (
          <View style={styles.infoRow}>
            <Icon name="usb" size={20} color="#666" />
            <Text style={styles.infoText}>
              {cameraManager.current.getCurrentDevice()?.name}
            </Text>
          </View>
        )}
      </View>
    </View>
  );
};

const getStressColor = (level: string): string => {
  switch (level) {
    case 'none': return '#4CAF50';
    case 'mild': return '#FFC107';
    case 'moderate': return '#FF9800';
    case 'severe': return '#F44336';
    default: return '#9E9E9E';
  }
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  cameraView: {
    flex: 1,
    margin: 16,
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#ccc',
    fontSize: 16,
    marginTop: 16,
  },
  thermalView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
  },
  thermalText: {
    color: '#fff',
    fontSize: 18,
  },
  scanOverlay: {
    position: 'absolute',
    bottom: 20,
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  scanText: {
    color: '#fff',
    fontSize: 14,
  },
  controls: {
    padding: 16,
    gap: 12,
  },
  button: {
    backgroundColor: '#4CAF50',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
    gap: 8,
  },
  buttonActive: {
    backgroundColor: '#f44336',
  },
  secondaryButton: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButtonText: {
    color: '#4CAF50',
  },
  info: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  thermalImageContainer: {
    flex: 1,
    position: 'relative',
  },
  temperatureOverlay: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  tempText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  analysisInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stressIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  alertText: {
    color: '#fff',
    fontSize: 12,
    marginTop: 4,
  },
  waitingForData: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  waitingText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 12,
  },
});

export default CameraScreen;