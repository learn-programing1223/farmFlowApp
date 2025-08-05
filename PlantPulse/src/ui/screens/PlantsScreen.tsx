import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { FAB } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { CameraScreenNavigationProp, PlantProfile } from '../../types';

const PlantsScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();
  const [plants, setPlants] = useState<PlantProfile[]>([]);

  const renderPlantItem = ({ item }: { item: PlantProfile }) => {
    const getStressColor = () => {
      if (!item.lastAnalysis) return '#4CAF50';
      switch (item.lastAnalysis.stressLevel) {
        case 'none':
          return '#4CAF50';
        case 'mild':
          return '#FFC107';
        case 'moderate':
          return '#FF9800';
        case 'severe':
          return '#F44336';
        default:
          return '#4CAF50';
      }
    };

    return (
      <TouchableOpacity
        style={styles.plantCard}
        onPress={() => navigation.navigate('PlantDetail', { plantId: item.id })}>
        <View style={styles.plantImage}>
          <Icon name="flower" size={40} color={getStressColor()} />
        </View>
        <View style={styles.plantInfo}>
          <Text style={styles.plantName}>{item.nickname}</Text>
          <Text style={styles.plantSpecies}>{item.species}</Text>
          <View style={styles.statusRow}>
            <Icon name="water" size={16} color={getStressColor()} />
            <Text style={[styles.statusText, { color: getStressColor() }]}>
              {item.lastAnalysis?.stressLevel || 'Not analyzed'}
            </Text>
          </View>
        </View>
        <Icon name="chevron-right" size={24} color="#ccc" />
      </TouchableOpacity>
    );
  };

  const EmptyPlantList = () => (
    <View style={styles.emptyContainer}>
      <Icon name="flower-outline" size={80} color="#ccc" />
      <Text style={styles.emptyTitle}>No plants yet</Text>
      <Text style={styles.emptyText}>
        Add your first plant to start monitoring its health
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={plants}
        renderItem={renderPlantItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={plants.length === 0 ? styles.emptyList : undefined}
        ListEmptyComponent={EmptyPlantList}
      />
      <FAB
        style={styles.fab}
        icon="plus"
        onPress={() => navigation.navigate('AddPlant')}
        color="#fff"
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  plantCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    marginHorizontal: 16,
    marginVertical: 8,
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
  plantImage: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  plantInfo: {
    flex: 1,
  },
  plantName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  plantSpecies: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
    gap: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  emptyList: {
    flex: 1,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#4CAF50',
  },
});

export default PlantsScreen;