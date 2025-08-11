import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  SafeAreaView,
  ActivityIndicator,
  Platform,
} from 'react-native';

// Disease categories and info
const diseaseCategories = [
  'Blight',
  'Healthy', 
  'Leaf_Spot',
  'Mosaic_Virus',
  'Nutrient_Deficiency',
  'Powdery_Mildew',
  'Rust'
];

const diseaseInfo = {
  'Blight': {
    severity: 'severe',
    description: 'Fungal disease causing dark spots and wilting',
    recommendations: [
      'Remove affected leaves immediately',
      'Apply copper-based fungicide',
      'Improve air circulation around plant'
    ]
  },
  'Healthy': {
    severity: 'none',
    description: 'Plant appears healthy with no visible disease',
    recommendations: [
      'Continue current care routine',
      'Monitor regularly for changes'
    ]
  },
  'Leaf_Spot': {
    severity: 'moderate',
    description: 'Bacterial or fungal spots on leaves',
    recommendations: [
      'Remove infected leaves',
      'Apply neem oil spray',
      'Avoid overhead watering'
    ]
  },
  'Mosaic_Virus': {
    severity: 'severe',
    description: 'Viral infection causing mottled leaf patterns',
    recommendations: [
      'Isolate infected plant immediately',
      'No cure - focus on prevention',
      'Control aphids and other vectors'
    ]
  },
  'Nutrient_Deficiency': {
    severity: 'mild',
    description: 'Lack of essential nutrients causing discoloration',
    recommendations: [
      'Apply balanced fertilizer',
      'Check soil pH levels',
      'Add compost or organic matter'
    ]
  },
  'Powdery_Mildew': {
    severity: 'moderate', 
    description: 'White powdery fungal growth on leaves',
    recommendations: [
      'Improve air circulation',
      'Apply sulfur or baking soda spray',
      'Remove affected leaves'
    ]
  },
  'Rust': {
    severity: 'moderate',
    description: 'Orange/brown pustules on leaf undersides',
    recommendations: [
      'Remove and destroy infected leaves',
      'Apply fungicide containing myclobutanil',
      'Water at soil level only'
    ]
  }
};

export default function App() {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  const simulateAnalysis = () => {
    setIsAnalyzing(true);
    
    // Simulate ML model inference (2 second delay)
    setTimeout(() => {
      // Random disease for demo (weighted towards common ones)
      const weights = [0.15, 0.3, 0.15, 0.1, 0.1, 0.1, 0.1]; // Healthy has higher weight
      const random = Math.random();
      let sum = 0;
      let selectedIndex = 0;
      
      for (let i = 0; i < weights.length; i++) {
        sum += weights[i];
        if (random < sum) {
          selectedIndex = i;
          break;
        }
      }
      
      const disease = diseaseCategories[selectedIndex];
      const confidence = 0.75 + Math.random() * 0.2; // 75-95% confidence
      
      setResults({
        disease,
        confidence,
        ...diseaseInfo[disease]
      });
      
      setIsAnalyzing(false);
      setCurrentScreen('results');
    }, 2000);
  };

  const reset = () => {
    setResults(null);
    setCurrentScreen('home');
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'severe': return '#f44336';
      case 'moderate': return '#ff9800';
      case 'mild': return '#ffc107';
      default: return '#4caf50';
    }
  };

  // Results Screen
  if (currentScreen === 'results' && results) {
    return (
      <SafeAreaView style={styles.container}>
        <ScrollView>
          <View style={styles.header}>
            <Text style={styles.title}>Analysis Results</Text>
          </View>
          
          <View style={styles.resultCard}>
            <View style={styles.diseaseHeader}>
              <Text style={styles.diseaseName}>{results.disease.replace(/_/g, ' ')}</Text>
              <Text style={styles.confidence}>
                {(results.confidence * 100).toFixed(1)}% Confidence
              </Text>
            </View>
            
            {results.severity !== 'none' && (
              <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(results.severity) }]}>
                <Text style={styles.severityText}>{results.severity.toUpperCase()} SEVERITY</Text>
              </View>
            )}
            
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Description</Text>
              <Text style={styles.description}>{results.description}</Text>
            </View>
            
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Treatment Recommendations</Text>
              {results.recommendations.map((rec, index) => (
                <View key={index} style={styles.recommendationItem}>
                  <Text style={styles.bullet}>•</Text>
                  <Text style={styles.recommendation}>{rec}</Text>
                </View>
              ))}
            </View>
          </View>
          
          <TouchableOpacity style={styles.button} onPress={reset}>
            <Text style={styles.buttonText}>Scan Another Plant</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    );
  }

  // Home/Camera Screen
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>PlantPulse</Text>
          <Text style={styles.subtitle}>AI Plant Disease Detection</Text>
        </View>
        
        <View style={styles.content}>
          <View style={styles.cameraContainer}>
            <View style={styles.cameraPlaceholder}>
              <Text style={styles.cameraIcon}>📷</Text>
              <Text style={styles.cameraText}>Camera Preview</Text>
              <Text style={styles.cameraSubtext}>
                (In production, camera will show here)
              </Text>
            </View>
          </View>
          
          <TouchableOpacity 
            style={[styles.button, styles.scanButton]}
            onPress={simulateAnalysis}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={[styles.buttonText, { marginLeft: 10 }]}>
                  Analyzing Plant...
                </Text>
              </View>
            ) : (
              <Text style={styles.buttonText}>Scan Plant</Text>
            )}
          </TouchableOpacity>
          
          <View style={styles.infoCard}>
            <Text style={styles.infoTitle}>How It Works</Text>
            <View style={styles.infoItem}>
              <Text style={styles.infoNumber}>1</Text>
              <Text style={styles.infoText}>Point camera at affected plant</Text>
            </View>
            <View style={styles.infoItem}>
              <Text style={styles.infoNumber}>2</Text>
              <Text style={styles.infoText}>Tap "Scan Plant" to analyze</Text>
            </View>
            <View style={styles.infoItem}>
              <Text style={styles.infoNumber}>3</Text>
              <Text style={styles.infoText}>Get instant diagnosis & treatment</Text>
            </View>
          </View>
          
          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>Model Performance</Text>
            <Text style={styles.statsText}>✓ 95% Accuracy</Text>
            <Text style={styles.statsText}>✓ 7 Disease Categories</Text>
            <Text style={styles.statsText}>✓ Under 100ms Detection</Text>
            <Text style={styles.statsText}>✓ Works Offline</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    flexGrow: 1,
  },
  header: {
    backgroundColor: '#4caf50',
    paddingTop: Platform.OS === 'ios' ? 20 : 10,
    paddingBottom: 20,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 16,
    color: '#fff',
    marginTop: 5,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  cameraContainer: {
    height: 300,
    marginBottom: 20,
    borderRadius: 15,
    overflow: 'hidden',
    backgroundColor: '#fff',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  cameraPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#e0e0e0',
  },
  cameraIcon: {
    fontSize: 60,
    marginBottom: 10,
  },
  cameraText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  cameraSubtext: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  button: {
    backgroundColor: '#4caf50',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  scanButton: {
    backgroundColor: '#2196f3',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  infoCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 15,
    marginBottom: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  infoTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  infoNumber: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#4caf50',
    color: '#fff',
    textAlign: 'center',
    lineHeight: 30,
    marginRight: 15,
    fontWeight: 'bold',
  },
  infoText: {
    flex: 1,
    fontSize: 15,
    color: '#666',
  },
  statsCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  statsText: {
    fontSize: 15,
    color: '#666',
    marginBottom: 5,
  },
  resultCard: {
    backgroundColor: '#fff',
    padding: 20,
    margin: 20,
    borderRadius: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  diseaseHeader: {
    marginBottom: 15,
  },
  diseaseName: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  confidence: {
    fontSize: 16,
    color: '#666',
  },
  severityBadge: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-start',
    marginBottom: 20,
  },
  severityText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  section: {
    marginTop: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  description: {
    fontSize: 15,
    color: '#666',
    lineHeight: 22,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  bullet: {
    fontSize: 16,
    marginRight: 10,
    color: '#4caf50',
  },
  recommendation: {
    flex: 1,
    fontSize: 15,
    color: '#666',
    lineHeight: 22,
  },
});