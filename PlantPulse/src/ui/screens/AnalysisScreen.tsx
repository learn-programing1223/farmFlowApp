import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { useRoute } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { AnalysisScreenRouteProp } from '../../types';

const AnalysisScreen: React.FC = () => {
  const route = useRoute<AnalysisScreenRouteProp>();
  const { plantId, analysisId } = route.params;

  // TODO: Load analysis data based on analysisId

  return (
    <ScrollView style={styles.container}>
      <View style={styles.summaryCard}>
        <View style={styles.summaryHeader}>
          <Icon name="checkbox-marked-circle" size={48} color="#4CAF50" />
          <Text style={styles.summaryTitle}>Plant is Healthy</Text>
        </View>
        <Text style={styles.summaryText}>
          No significant issues detected. Continue regular care routine.
        </Text>
      </View>

      <View style={styles.metricsCard}>
        <Text style={styles.sectionTitle}>Detailed Metrics</Text>
        
        <View style={styles.metricItem}>
          <View style={styles.metricHeader}>
            <Icon name="water" size={20} color="#2196F3" />
            <Text style={styles.metricLabel}>Water Stress Index</Text>
          </View>
          <View style={styles.metricBar}>
            <View style={[styles.metricFill, { width: '20%', backgroundColor: '#4CAF50' }]} />
          </View>
          <Text style={styles.metricValue}>0.20 (Optimal)</Text>
        </View>

        <View style={styles.metricItem}>
          <View style={styles.metricHeader}>
            <Icon name="bacteria" size={20} color="#4CAF50" />
            <Text style={styles.metricLabel}>Disease Detection</Text>
          </View>
          <Text style={styles.metricValue}>No pathogens detected</Text>
        </View>

        <View style={styles.metricItem}>
          <View style={styles.metricHeader}>
            <Icon name="leaf" size={20} color="#FF9800" />
            <Text style={styles.metricLabel}>Nutrient Status</Text>
          </View>
          <View style={styles.nutrientGrid}>
            <View style={styles.nutrientItem}>
              <Text style={styles.nutrientLabel}>N</Text>
              <Icon name="check-circle" size={16} color="#4CAF50" />
            </View>
            <View style={styles.nutrientItem}>
              <Text style={styles.nutrientLabel}>P</Text>
              <Icon name="check-circle" size={16} color="#4CAF50" />
            </View>
            <View style={styles.nutrientItem}>
              <Text style={styles.nutrientLabel}>K</Text>
              <Icon name="check-circle" size={16} color="#4CAF50" />
            </View>
          </View>
        </View>
      </View>

      <View style={styles.recommendationsCard}>
        <Text style={styles.sectionTitle}>Recommendations</Text>
        <View style={styles.recommendation}>
          <Icon name="check" size={20} color="#4CAF50" />
          <Text style={styles.recommendationText}>
            Continue current watering schedule
          </Text>
        </View>
        <View style={styles.recommendation}>
          <Icon name="check" size={20} color="#4CAF50" />
          <Text style={styles.recommendationText}>
            Maintain temperature between 18-27°C
          </Text>
        </View>
        <View style={styles.recommendation}>
          <Icon name="lightbulb" size={20} color="#FFC107" />
          <Text style={styles.recommendationText}>
            Consider fertilizing in 2 weeks
          </Text>
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
  summaryCard: {
    backgroundColor: '#fff',
    margin: 16,
    padding: 24,
    borderRadius: 8,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  summaryHeader: {
    alignItems: 'center',
    marginBottom: 16,
  },
  summaryTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#333',
    marginTop: 8,
  },
  summaryText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  metricsCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 16,
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
  metricItem: {
    marginBottom: 20,
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  metricLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: '#333',
  },
  metricBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginBottom: 4,
  },
  metricFill: {
    height: '100%',
    borderRadius: 4,
  },
  metricValue: {
    fontSize: 12,
    color: '#666',
  },
  nutrientGrid: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 8,
  },
  nutrientItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  nutrientLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  recommendationsCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 16,
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
  recommendation: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    color: '#333',
  },
});

export default AnalysisScreen;