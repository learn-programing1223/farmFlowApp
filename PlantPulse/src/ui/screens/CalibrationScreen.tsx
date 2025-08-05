import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { THERMAL_CONSTANTS } from '../../constants';

const CalibrationScreen: React.FC = () => {
  const [calibration, setCalibration] = useState({
    emissivity: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.EMISSIVITY.toString(),
    reflectedTemp: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.REFLECTED_TEMP.toString(),
    distance: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.DISTANCE.toString(),
    humidity: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.HUMIDITY.toString(),
  });

  const handleCalibrate = () => {
    // TODO: Implement calibration logic
    Alert.alert(
      'Calibration Complete',
      'Thermal camera has been calibrated successfully',
      [{ text: 'OK' }]
    );
  };

  const handleReset = () => {
    setCalibration({
      emissivity: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.EMISSIVITY.toString(),
      reflectedTemp: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.REFLECTED_TEMP.toString(),
      distance: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.DISTANCE.toString(),
      humidity: THERMAL_CONSTANTS.DEFAULT_CALIBRATION.HUMIDITY.toString(),
    });
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.infoCard}>
        <Icon name="information" size={24} color="#2196F3" />
        <Text style={styles.infoText}>
          Calibrate your thermal camera for accurate temperature measurements.
          Default values are optimized for plant leaves.
        </Text>
      </View>

      <View style={styles.form}>
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Emissivity (0.0 - 1.0)</Text>
          <Text style={styles.description}>
            Thermal radiation efficiency of plant leaves
          </Text>
          <TextInput
            style={styles.input}
            value={calibration.emissivity}
            onChangeText={(text) =>
              setCalibration(prev => ({ ...prev, emissivity: text }))
            }
            keyboardType="decimal-pad"
            placeholder="0.95"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Reflected Temperature (°C)</Text>
          <Text style={styles.description}>
            Ambient temperature reflected by the surface
          </Text>
          <TextInput
            style={styles.input}
            value={calibration.reflectedTemp}
            onChangeText={(text) =>
              setCalibration(prev => ({ ...prev, reflectedTemp: text }))
            }
            keyboardType="decimal-pad"
            placeholder="20.0"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Distance (meters)</Text>
          <Text style={styles.description}>
            Distance from camera to plant
          </Text>
          <TextInput
            style={styles.input}
            value={calibration.distance}
            onChangeText={(text) =>
              setCalibration(prev => ({ ...prev, distance: text }))
            }
            keyboardType="decimal-pad"
            placeholder="0.5"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Humidity (%)</Text>
          <Text style={styles.description}>
            Relative humidity of the environment
          </Text>
          <TextInput
            style={styles.input}
            value={calibration.humidity}
            onChangeText={(text) =>
              setCalibration(prev => ({ ...prev, humidity: text }))
            }
            keyboardType="number-pad"
            placeholder="50"
          />
        </View>

        <TouchableOpacity style={styles.calibrateButton} onPress={handleCalibrate}>
          <Icon name="tune" size={24} color="#fff" />
          <Text style={styles.buttonText}>Apply Calibration</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
          <Icon name="restore" size={24} color="#4CAF50" />
          <Text style={[styles.buttonText, { color: '#4CAF50' }]}>
            Reset to Defaults
          </Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  infoCard: {
    flexDirection: 'row',
    backgroundColor: '#E3F2FD',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    gap: 12,
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    color: '#1976D2',
    lineHeight: 20,
  },
  form: {
    padding: 16,
  },
  inputGroup: {
    marginBottom: 24,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  description: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#333',
  },
  calibrateButton: {
    backgroundColor: '#4CAF50',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
    gap: 8,
    marginTop: 24,
  },
  resetButton: {
    backgroundColor: '#fff',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 8,
    gap: 8,
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});

export default CalibrationScreen;