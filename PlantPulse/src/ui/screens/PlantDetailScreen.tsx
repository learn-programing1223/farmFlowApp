import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useRoute } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { PlantDetailScreenRouteProp } from '../../types';

const PlantDetailScreen: React.FC = () => {
  const route = useRoute<PlantDetailScreenRouteProp>();
  const { plantId } = route.params;

  // TODO: Load plant data based on plantId

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.plantImage}>
          <Icon name="flower" size={80} color="#4CAF50" />
        </View>
        <Text style={styles.plantName}>My Plant</Text>
        <Text style={styles.plantSpecies}>Species Name</Text>
      </View>

      <View style={styles.statusCard}>
        <Text style={styles.sectionTitle}>Current Status</Text>
        <View style={styles.statusGrid}>
          <View style={styles.statusItem}>
            <Icon name="water" size={24} color="#2196F3" />
            <Text style={styles.statusLabel}>Water Stress</Text>
            <Text style={styles.statusValue}>None</Text>
          </View>
          <View style={styles.statusItem}>
            <Icon name="bacteria" size={24} color="#4CAF50" />
            <Text style={styles.statusLabel}>Disease</Text>
            <Text style={styles.statusValue}>Healthy</Text>
          </View>
          <View style={styles.statusItem}>
            <Icon name="leaf" size={24} color="#FF9800" />
            <Text style={styles.statusLabel}>Nutrients</Text>
            <Text style={styles.statusValue}>Optimal</Text>
          </View>
          <View style={styles.statusItem}>
            <Icon name="thermometer" size={24} color="#F44336" />
            <Text style={styles.statusLabel}>Temperature</Text>
            <Text style={styles.statusValue}>--°C</Text>
          </View>
        </View>
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.sectionTitle}>Plant Information</Text>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Optimal Temperature</Text>
          <Text style={styles.infoValue}>18-27°C</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Optimal Humidity</Text>
          <Text style={styles.infoValue}>60-80%</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Water Stress Threshold</Text>
          <Text style={styles.infoValue}>0.40</Text>
        </View>
      </View>

      <TouchableOpacity style={styles.scanButton}>
        <Icon name="scan-helper" size={24} color="#fff" />
        <Text style={styles.scanButtonText}>Scan Now</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 32,
    backgroundColor: '#fff',
  },
  plantImage: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#E8F5E9',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  plantName: {
    fontSize: 24,
    fontWeight: '600',
    color: '#333',
  },
  plantSpecies: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
  },
  statusCard: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  statusGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
  },
  statusItem: {
    width: '50%',
    padding: 8,
    alignItems: 'center',
  },
  statusLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
  },
  statusValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginTop: 4,
  },
  infoCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    padding: 16,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  infoLabel: {
    fontSize: 14,
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  scanButton: {
    backgroundColor: '#4CAF50',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    gap: 8,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default PlantDetailScreen;