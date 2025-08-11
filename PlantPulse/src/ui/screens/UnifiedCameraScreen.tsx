import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  TouchableOpacity,
  Text,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { RGBCameraScreen } from './RGBCameraScreen';
// Import thermal camera when available
// import { ThermalCameraScreen } from './ThermalCameraScreen';

const { width: screenWidth } = Dimensions.get('window');

type CameraMode = 'rgb' | 'thermal';

export const UnifiedCameraScreen: React.FC = () => {
  const [cameraMode, setCameraMode] = useState<CameraMode>('rgb');
  const [thermalAvailable] = useState(false); // Set to true when thermal camera is connected

  const renderModeSelector = () => (
    <View style={styles.modeSelectorContainer}>
      <View style={styles.modeSelector}>
        <TouchableOpacity
          style={[
            styles.modeButton,
            cameraMode === 'rgb' && styles.modeButtonActive,
          ]}
          onPress={() => setCameraMode('rgb')}
        >
          <Icon 
            name="photo-camera" 
            size={24} 
            color={cameraMode === 'rgb' ? '#fff' : '#666'}
          />
          <Text style={[
            styles.modeText,
            cameraMode === 'rgb' && styles.modeTextActive
          ]}>
            RGB Camera
          </Text>
          <Text style={[
            styles.modeSubtext,
            cameraMode === 'rgb' && styles.modeTextActive
          ]}>
            Disease Detection
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.modeButton,
            cameraMode === 'thermal' && styles.modeButtonActive,
            !thermalAvailable && styles.modeButtonDisabled,
          ]}
          onPress={() => thermalAvailable && setCameraMode('thermal')}
          disabled={!thermalAvailable}
        >
          <Icon 
            name="thermostat" 
            size={24} 
            color={cameraMode === 'thermal' ? '#fff' : '#666'}
          />
          <Text style={[
            styles.modeText,
            cameraMode === 'thermal' && styles.modeTextActive,
            !thermalAvailable && styles.modeTextDisabled,
          ]}>
            Thermal Camera
          </Text>
          <Text style={[
            styles.modeSubtext,
            cameraMode === 'thermal' && styles.modeTextActive,
            !thermalAvailable && styles.modeTextDisabled,
          ]}>
            {thermalAvailable ? 'Stress Detection' : 'Not Connected'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      {cameraMode === 'rgb' ? (
        <RGBCameraScreen />
      ) : (
        // Placeholder for thermal camera
        <View style={styles.placeholderContainer}>
          <Icon name="thermostat" size={60} color="#666" />
          <Text style={styles.placeholderText}>
            Thermal camera support coming soon
          </Text>
          <Text style={styles.placeholderSubtext}>
            Connect a USB-C thermal camera to enable
          </Text>
        </View>
      )}
      
      {renderModeSelector()}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  modeSelectorContainer: {
    position: 'absolute',
    bottom: 100,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  modeSelector: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 12,
    padding: 4,
    marginHorizontal: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  modeButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  modeButtonActive: {
    backgroundColor: '#4CAF50',
  },
  modeButtonDisabled: {
    opacity: 0.5,
  },
  modeText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginTop: 4,
  },
  modeTextActive: {
    color: '#fff',
  },
  modeTextDisabled: {
    color: '#999',
  },
  modeSubtext: {
    fontSize: 11,
    color: '#999',
    marginTop: 2,
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  placeholderText: {
    fontSize: 18,
    color: '#666',
    marginTop: 20,
  },
  placeholderSubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
  },
});

export default UnifiedCameraScreen;