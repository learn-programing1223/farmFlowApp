import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SectionList,
  TouchableOpacity,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { format } from 'date-fns';
import { CameraScreenNavigationProp, PlantHealthAnalysis } from '../../types';

interface HistoryItem {
  analysis: PlantHealthAnalysis;
  plantName: string;
  plantId: string;
}

interface HistorySection {
  title: string;
  data: HistoryItem[];
}

const HistoryScreen: React.FC = () => {
  const navigation = useNavigation<CameraScreenNavigationProp>();
  const sections: HistorySection[] = [];

  const getStressIcon = (stressLevel: string) => {
    switch (stressLevel) {
      case 'none':
        return { name: 'emoticon-happy', color: '#4CAF50' };
      case 'mild':
        return { name: 'emoticon-neutral', color: '#FFC107' };
      case 'moderate':
        return { name: 'emoticon-sad', color: '#FF9800' };
      case 'severe':
        return { name: 'emoticon-cry', color: '#F44336' };
      default:
        return { name: 'emoticon-outline', color: '#666' };
    }
  };

  const renderHistoryItem = ({ item }: { item: HistoryItem }) => {
    const icon = getStressIcon(item.analysis.stressLevel);
    
    return (
      <TouchableOpacity
        style={styles.historyItem}
        onPress={() =>
          navigation.navigate('Analysis', {
            plantId: item.plantId,
            analysisId: item.analysis.id,
          })
        }>
        <View style={[styles.iconContainer, { backgroundColor: `${icon.color}20` }]}>
          <Icon name={icon.name} size={24} color={icon.color} />
        </View>
        <View style={styles.itemContent}>
          <Text style={styles.plantName}>{item.plantName}</Text>
          <Text style={styles.analysisText}>
            Water Stress: {Math.round(item.analysis.waterStressIndex * 100)}%
          </Text>
          <Text style={styles.timeText}>
            {format(item.analysis.timestamp, 'h:mm a')}
          </Text>
        </View>
        <Icon name="chevron-right" size={24} color="#ccc" />
      </TouchableOpacity>
    );
  };

  const renderSectionHeader = ({ section }: { section: HistorySection }) => (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{section.title}</Text>
    </View>
  );

  const EmptyHistory = () => (
    <View style={styles.emptyContainer}>
      <Icon name="history" size={80} color="#ccc" />
      <Text style={styles.emptyTitle}>No analysis history</Text>
      <Text style={styles.emptyText}>
        Start scanning your plants to build up their health history
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <SectionList
        sections={sections}
        renderItem={renderHistoryItem}
        renderSectionHeader={renderSectionHeader}
        keyExtractor={(item) => item.analysis.id}
        contentContainerStyle={sections.length === 0 ? styles.emptyList : undefined}
        ListEmptyComponent={EmptyHistory}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  sectionHeader: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    textTransform: 'uppercase',
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  itemContent: {
    flex: 1,
  },
  plantName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  analysisText: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  timeText: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
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
});

export default HistoryScreen;