import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

const SettingsScreen: React.FC = () => {
  const [settings, setSettings] = useState({
    temperatureUnit: 'celsius',
    autoSaveAnalysis: true,
    enableGPUAcceleration: true,
    notificationsEnabled: true,
  });

  const toggleSetting = (key: string) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>General</Text>
        
        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Icon name="thermometer" size={24} color="#666" />
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>Temperature Unit</Text>
              <Text style={styles.settingDescription}>
                {settings.temperatureUnit === 'celsius' ? 'Celsius (°C)' : 'Fahrenheit (°F)'}
              </Text>
            </View>
          </View>
          <Switch
            value={settings.temperatureUnit === 'fahrenheit'}
            onValueChange={() =>
              setSettings(prev => ({
                ...prev,
                temperatureUnit: prev.temperatureUnit === 'celsius' ? 'fahrenheit' : 'celsius',
              }))
            }
            trackColor={{ false: '#767577', true: '#4CAF50' }}
            thumbColor={settings.temperatureUnit === 'fahrenheit' ? '#fff' : '#f4f3f4'}
          />
        </View>

        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Icon name="content-save" size={24} color="#666" />
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>Auto-save Analysis</Text>
              <Text style={styles.settingDescription}>
                Automatically save all scan results
              </Text>
            </View>
          </View>
          <Switch
            value={settings.autoSaveAnalysis}
            onValueChange={() => toggleSetting('autoSaveAnalysis')}
            trackColor={{ false: '#767577', true: '#4CAF50' }}
            thumbColor={settings.autoSaveAnalysis ? '#fff' : '#f4f3f4'}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Performance</Text>
        
        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Icon name="gpu" size={24} color="#666" />
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>GPU Acceleration</Text>
              <Text style={styles.settingDescription}>
                Use GPU for faster ML inference
              </Text>
            </View>
          </View>
          <Switch
            value={settings.enableGPUAcceleration}
            onValueChange={() => toggleSetting('enableGPUAcceleration')}
            trackColor={{ false: '#767577', true: '#4CAF50' }}
            thumbColor={settings.enableGPUAcceleration ? '#fff' : '#f4f3f4'}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notifications</Text>
        
        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Icon name="bell" size={24} color="#666" />
            <View style={styles.settingText}>
              <Text style={styles.settingLabel}>Push Notifications</Text>
              <Text style={styles.settingDescription}>
                Alerts for critical plant health issues
              </Text>
            </View>
          </View>
          <Switch
            value={settings.notificationsEnabled}
            onValueChange={() => toggleSetting('notificationsEnabled')}
            trackColor={{ false: '#767577', true: '#4CAF50' }}
            thumbColor={settings.notificationsEnabled ? '#fff' : '#f4f3f4'}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        <View style={styles.aboutItem}>
          <Text style={styles.aboutLabel}>Version</Text>
          <Text style={styles.aboutValue}>1.0.0</Text>
        </View>
        <View style={styles.aboutItem}>
          <Text style={styles.aboutLabel}>ML Model Version</Text>
          <Text style={styles.aboutValue}>plant_health_v1</Text>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  section: {
    backgroundColor: '#fff',
    marginVertical: 8,
    paddingVertical: 8,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4CAF50',
    textTransform: 'uppercase',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  settingInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  settingText: {
    flex: 1,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  settingDescription: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  aboutItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  aboutLabel: {
    fontSize: 16,
    color: '#333',
  },
  aboutValue: {
    fontSize: 16,
    color: '#666',
  },
});

export default SettingsScreen;