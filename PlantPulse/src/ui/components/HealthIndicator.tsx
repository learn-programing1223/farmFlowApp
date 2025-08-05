import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { PlantHealthAnalysis } from '../../types/plant';

interface HealthIndicatorProps {
  analysis: PlantHealthAnalysis;
  compact?: boolean;
  showDetails?: boolean;
}

const HealthIndicator: React.FC<HealthIndicatorProps> = ({
  analysis,
  compact = false,
  showDetails = true,
}) => {
  const getHealthScore = (): number => {
    let score = 100;
    
    // Deduct for water stress
    score -= analysis.waterStressIndex * 30;
    
    // Deduct for disease
    if (analysis.diseaseDetection.type !== 'healthy') {
      score -= 20 + (analysis.diseaseDetection.affectedArea * 0.5);
    }
    
    // Deduct for nutrient deficiencies
    const nutrients = [
      analysis.nutrientStatus.nitrogen,
      analysis.nutrientStatus.phosphorus,
      analysis.nutrientStatus.potassium,
    ];
    const deficientCount = nutrients.filter(n => n === 'deficient').length;
    score -= deficientCount * 10;
    
    return Math.max(0, Math.min(100, score));
  };

  const getHealthColor = (score: number): string => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FFC107';
    if (score >= 40) return '#FF9800';
    return '#F44336';
  };

  const getStressIcon = (): string => {
    switch (analysis.stressLevel) {
      case 'none': return 'emoticon-happy';
      case 'mild': return 'emoticon-neutral';
      case 'moderate': return 'emoticon-sad';
      case 'severe': return 'emoticon-cry';
      default: return 'emoticon-neutral';
    }
  };

  const healthScore = getHealthScore();
  const healthColor = getHealthColor(healthScore);

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <View style={[styles.scoreCircle, { borderColor: healthColor }]}>
          <Text style={[styles.scoreText, { color: healthColor }]}>
            {healthScore.toFixed(0)}
          </Text>
        </View>
        <Icon name={getStressIcon()} size={24} color={healthColor} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={[styles.largeScorecircle, { borderColor: healthColor }]}>
          <Text style={[styles.largeScoreText, { color: healthColor }]}>
            {healthScore.toFixed(0)}
          </Text>
          <Text style={styles.scoreLabel}>Health Score</Text>
        </View>
        <Icon name={getStressIcon()} size={48} color={healthColor} />
      </View>

      {showDetails && (
        <View style={styles.details}>
          <View style={styles.detailRow}>
            <Icon name="water" size={20} color="#2196F3" />
            <Text style={styles.detailLabel}>Water Stress:</Text>
            <Text style={[
              styles.detailValue,
              { color: getStressColor(analysis.stressLevel) }
            ]}>
              {analysis.stressLevel.charAt(0).toUpperCase() + analysis.stressLevel.slice(1)}
            </Text>
          </View>

          <View style={styles.detailRow}>
            <Icon name="virus" size={20} color="#9C27B0" />
            <Text style={styles.detailLabel}>Disease:</Text>
            <Text style={[
              styles.detailValue,
              { color: analysis.diseaseDetection.type === 'healthy' ? '#4CAF50' : '#F44336' }
            ]}>
              {analysis.diseaseDetection.type === 'healthy' 
                ? 'None Detected' 
                : `${analysis.diseaseDetection.type} (${(analysis.diseaseDetection.confidence * 100).toFixed(0)}%)`}
            </Text>
          </View>

          <View style={styles.detailRow}>
            <Icon name="leaf" size={20} color="#4CAF50" />
            <Text style={styles.detailLabel}>Nutrients:</Text>
            <View style={styles.nutrientIcons}>
              <NutrientIndicator label="N" status={analysis.nutrientStatus.nitrogen} />
              <NutrientIndicator label="P" status={analysis.nutrientStatus.phosphorus} />
              <NutrientIndicator label="K" status={analysis.nutrientStatus.potassium} />
            </View>
          </View>

          {analysis.recommendations.length > 0 && (
            <View style={styles.recommendations}>
              <Text style={styles.recommendationsTitle}>Recommendations:</Text>
              {analysis.recommendations.slice(0, 3).map((rec, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Icon name="checkbox-marked-circle" size={16} color="#4CAF50" />
                  <Text style={styles.recommendationText}>{rec}</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      )}
    </View>
  );
};

const NutrientIndicator: React.FC<{
  label: string;
  status: 'deficient' | 'optimal' | 'excess';
}> = ({ label, status }) => {
  const getColor = () => {
    switch (status) {
      case 'deficient': return '#F44336';
      case 'optimal': return '#4CAF50';
      case 'excess': return '#FF9800';
    }
  };

  return (
    <View style={[styles.nutrientBadge, { backgroundColor: getColor() }]}>
      <Text style={styles.nutrientLabel}>{label}</Text>
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
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  scoreCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 3,
    justifyContent: 'center',
    alignItems: 'center',
  },
  largeScorecircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  largeScoreText: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  scoreLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  details: {
    gap: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  detailLabel: {
    flex: 1,
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  nutrientIcons: {
    flexDirection: 'row',
    gap: 8,
  },
  nutrientBadge: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  nutrientLabel: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  recommendations: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  recommendationsTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  recommendationItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 4,
  },
  recommendationText: {
    flex: 1,
    fontSize: 13,
    color: '#666',
  },
});

export default HealthIndicator;